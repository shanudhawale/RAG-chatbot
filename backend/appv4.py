# importing libraries
import logging
from fastapi import FastAPI, HTTPException, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from datetime import datetime
from pathlib import Path
import re
from copy import deepcopy
from llama_index.core.query_engine import CustomQueryEngine, SimpleMultiModalQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, TextNode, ImageDocument
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
import base64
import shutil
from llama_index.core import Document
from llama_index.core.memory import ChatMemoryBuffer , ChatSummaryMemoryBuffer
import json
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.export import generate_multimodal_pages
from llama_index.core import SummaryIndex
import tiktoken
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="RAG API")

load_dotenv()
llm = OpenAI(model="gpt-4-0125-preview", api_key=os.getenv("OPENAI_API_KEY"))
Settings.llm = llm

# Summarizer chat history
summarizer_llm = llm
tokenizer_fn = tiktoken.encoding_for_model("gpt-4-0125-preview").encode

# Evaluation Criteria inbuilt from llama-index
relavancy_evaluator = RelevancyEvaluator(llm=llm)
faithfull_evaluator = FaithfulnessEvaluator(llm=llm)

# Initialize ChromaDB
CHROMA_DB_PATH = "/app/backend/chroma_db2"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    data_path: Optional[str] = Field(default="data", description="Path to document directory")
    user_id: str = Field(..., description="Unique user identifier")
    input_files: List[tuple] = Field(..., description="list of files")
    collection:str = Field(..., description="name of collection for vector db")


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

# Initialize global variables
index_cache: Dict[str, VectorStoreIndex] = {}
chat_history: Dict[str, ChatSummaryMemoryBuffer] = {}


pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.generate_page_images = True

# Docling Document Converter intialized
doc_converter = (
        DocumentConverter(  
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.XLSX,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, pipeline_options=pdf_pipeline_options
                )
            },
        )
    )

# Instructor Embeddings
class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-base",
        instruction: str = "Represent this document for semantic search across academic, report, and structured data formats around page numbers. If the content is from a research paper or presentation (PDF), focus on extracting key technical concepts, arguments, and any referenced figures or slides. If the content is from a DOCX file, capture the logical flow, section headings, and paragraph summaries relevant for understanding the main points and conclusions.",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the InstructorEmbeddings class.

        Args:
            instructor_model_name (str): The name of the Instructor model to use. Defaults to "hkunlp/instructor-base".
            instruction (str): The instruction to give to the Instructor model. Defaults to a string that asks the model to extract key technical concepts, arguments, and referenced figures or slides for PDFs and logical flow, section headings, and paragraph summaries for DOCX files.
            **kwargs: Additional keyword arguments to pass to the BaseEmbedding constructor.
        """
        super().__init__(**kwargs)
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

# Instantiate InstructorEmbeddings
embed_model = InstructorEmbeddings(embed_batch_size=2)
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Initialize Persistent ChromaDB
try:
    db = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.Settings(
            allow_reset=True,
            anonymized_telemetry=False  
        ))
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    db = None

current_index: Optional[VectorStoreIndex] = None

# Initialize function to index for the process_query() function
def initialize_index(doc_collection_name:str, docs, doc_type):
    """Initialize the index from ChromaDB"""
    global current_index
    
    try:
        if not docs:
            logger.warning("No documents provided for indexing")
                      
        collection_name = doc_collection_name
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Initialize or load the index
        if chroma_collection.count() == 0:
            print(f"Creating new ChromaDB index for document type: {doc_type}")
            if doc_type == "xlsx":
                # Use explicit service context with embed_model for SummaryIndex
                current_index = SummaryIndex.from_documents(
                    docs, 
                    storage_context=storage_context,
                )
            else:
                current_index = VectorStoreIndex.from_documents(
                    docs, 
                    storage_context=storage_context, 
                    embed_model=embed_model
                )
        else:
            print(f"Loading existing ChromaDB index for document type: {doc_type}")
            if doc_type == "xlsx":
                # Use load_from_disk directly with ChromaVectorStore
                current_index = SummaryIndex(
                    vector_store=vector_store,
                    storage_context=storage_context,
                )
                
                # Add new documents if provided
                if docs:
                    for doc in docs:
                        current_index.insert(doc)
            else:
                current_index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                if docs:
                    for doc in docs:
                        current_index.insert(doc)
                
        return current_index
    
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        raise

def process_documents(input_paths, output_dir):
    # print("Input files in document processing")
    """
    Process a list of input files and convert them to a list of Document objects

    Args:
        input_paths (list[tuple]): A list of tuples containing the input file path and the actual document name
        output_dir (str): The output directory for the processed documents

    Returns:
        tuple[list[Document], str]: A tuple containing a list of Document objects and the document type
    """
    docs = []
    inputfiles = [Path(ip[0]) for ip in input_paths]
    temp_file_names = [ip[0].split('/')[-1] for ip in input_paths]
    actual_filenames = [ip[-1] for ip in input_paths]
    conv_results = doc_converter.convert_all(inputfiles)
    for res in conv_results:
        if res.input.file.name.split('.')[-1] in ['docx','txt']:
            doc = res.document.export_to_markdown() 
            document = Document(text= doc ,metadata={"document_type": "docx",
                                                     "document_name": res.input.file.name,
                                                     "actual_doc_name": actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                     })
            docs.append(document)

        elif res.input.file.name.split('.')[-1] in ['xlsx']:
            doc = res.document.export_to_markdown()
            document = Document(text= doc ,metadata={"document_type": "xlsx",
                                                     "document_name": res.input.file.name,
                                                     "actual_doc_name":  actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                     })
            docs.append(document)

        elif res.input.file.name.split('.')[-1] == "pdf":
            # print(">>>>>>>>>> Processing pdf")
            for (content_text,content_md,content_dt,page_cells,page_segments,page,) in generate_multimodal_pages(res):
                page_no = page.page_no + 1
                # print(page_no,">>>>>>>", content_text)
                page_image_filename = Path(f'/app/backend/data_images/{output_dir}/{res.input.file.name}-{page_no}.jpg')
                with page_image_filename.open("wb") as fp:
                    page.image.save(fp, format="JPEG")
                # print("saved image", page_no)
                document = Document(text = f"page number:{page_no}\n"+content_md, metadata = {"document_type": "pdf",
                                                                   "pdf_name": res.input.file.name,
                                                                   "actual_doc_name": actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                                   "page_num": page.page_no + 1,
                                                                   "image_path":f'/app/backend/data_images/{output_dir}/{res.input.file.name}-{page_no}.jpg',
                                                                   })
                docs.append(document)

    return docs, document.metadata['document_type']

# Define the prompt template that is passed on to the OPENAI LLM
QA_PROMPT_TMPL = """\
Below we give parsed text from slides in two different formats, as well as the image.

We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
layout of the text.

Chat History:
{chat_history}

Use the image information first and foremost. ONLY use the text/markdown information 
if you can't understand the image.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.
If there's no match then do not provide the answer.

Query: {query_str}
Please provide a detailed response:
1. Directly all possible answers the question in great detail for upto 7-8 sentences.
2. Any other linkage to the possible answer in the whole document as bonus
3. References specific parts of the documents (line numbers and page numbers) in the context information in one line. Describes relevant images when they support the answer
4. Maintains continuity with previous conversation

Answer: 

Given below is XML that describes the information to extract from this document and the tags to extract it into.
`doc_type` can be only one of the 4 types mentioned [docx, xlsx, pdf, txt]
<output>
    <list name="result_respnse" description="Bullet points regarding the query">
        <object>
            <string name="explanation1"/>
            <string name="explanation2"/>
            <string name="explanation3"/>
            <string name="explanation4"/>
        </object>
    </list>
    <list name="refrence" description="multiple query references mentioned in the explaination">
        <object>
            <string name="page_number"/>
            <string name="pdf_name"/>
            <string name="actual_doc_name"/>
            <string namee="doc_type"/>
        </object>
    </list>
</output>

ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise.
"""

CONTEXT_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

class MultiModalConversationalEngine(CustomQueryEngine):
    """Custom query engine for multimodal conversational RAG"""
    
    def intialize_engine(
        self,
        retriever,
        multi_modal_llm: OpenAIMultiModal,
        memory_buffer: ChatSummaryMemoryBuffer = None,
        context_prompt: PromptTemplate = CONTEXT_PROMPT,
    ):
        """Initialize the multimodal conversational RAG engine

        Args:
            retriever: The retriever to use for the engine
            multi_modal_llm: The multimodal LLM to use for the engine
            memory_buffer: The chat summary memory buffer to use for the engine
            context_prompt: The prompt to use for the engine
        """
        self._retriever = retriever
        self._llm = multi_modal_llm
        self._memory = memory_buffer or ChatSummaryMemoryBuffer.from_defaults(chat_history=chat_history,
                                                                                llm=summarizer_llm,
                                                                                token_limit=2,
                                                                                tokenizer_fn=tokenizer_fn,)
        self._context_prompt = context_prompt


    def _create_image_documents(self, image_paths):
        """Create image documents for the given image paths to be displayed on the Chat UI"""
        image_documents = []
        #print("@@@@@@@@@",image_paths)
        for path in image_paths:
            try:
                logger.info(f'{path} {type(path)}')
                path = path.replace("\\","/")
                if path and os.path.exists(path):  # Check if path exists
                    with open(path, "rb") as f:
                        image_data = f.read()
                        image_doc = ImageDocument(
                            image_data=image_data,
                            image_path=path,
                        )
                        image_documents.append(image_doc)
                else:
                    logger.warning(f"Image path does not exist: {path}")
            except Exception as e:
                logger.error(f"Error reading image {path}: {e}")
        return image_documents

    def process_response(self, response_text: str) -> Dict[str, Any]:
        """
        Process and restructure the response to combine explanations and references from the LLM response
        """
        try:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            response_data = json.loads(cleaned_text)
            
            if "result_respnse" in response_data and isinstance(response_data["result_respnse"], list):
                if len(response_data["result_respnse"]) > 0:
                    explanations = response_data["result_respnse"][0]
                    combined_explanation = " ".join(filter(None, [
                        explanations.get("explanation1", ""),
                        explanations.get("explanation2", ""),
                        explanations.get("explanation3", ""),
                        explanations.get("explanation4", ""),
                    ]))
                else:
                    combined_explanation = "No explanation provided."
                
                processed_response = {
                    "result_response": {
                        "explanation": combined_explanation.strip()
                    },
                    "refrence": []
                }
                
                if "refrence" in response_data:
                    for ref in response_data["refrence"]:
                        if isinstance(ref, dict):# and "page_number, pdf_name, actual_doc_name, doc_type" in ref:
                            try:
                                page_num, pdf_name, actual_doc_name, doc_type = ref["page_number"], ref["pdf_name"], ref["actual_doc_name"], ref["doc_type"]
                                processed_response["refrence"].append({
                                    "page_number": page_num.strip(),
                                    "pdf_name": pdf_name.strip(),
                                    "actual_doc_name":actual_doc_name.strip(),
                                    "doc_type":doc_type.strip()
                                })
                            except ValueError:
                                logger.warning(f"Invalid reference format: {ref}")
                
                source_links = []
                for ref in processed_response["refrence"]:
                    if ref['doc_type'] =='pdf':
                        source_link = f"[Source: {ref['actual_doc_name']}, Page: {ref['page_number']}, Document_type: {ref['doc_type']}]"
                    else:
                        source_link = f"[Source: {ref['actual_doc_name']}, Document_type: {ref['doc_type']}]"
                    source_links.append(source_link)

                if source_links:
                    combined_explanation += "\n\nSources:\n" + "\n".join(source_links)
                
                processed_response = {
                    "result_response": {
                        "explanation": combined_explanation.strip()
                    },
                    "refrence": processed_response["refrence"],
                    "source_links": source_links
                }
                # print(">>>>>>>Processed REsponse???????????", processed_response)
                return processed_response
            
            return response_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}\nResponse text: {response_text}")
            return {
                "result_response": {
                    "explanation": str(response_text)
                },
                 "refrence": [],
                 "source_links": []
            }

    def custom_query(self, query_str: str) -> Response:
        """Process query with context text and chat history and user querys that are based out of documnets and images"""
        
        # Get chat history
        try:
            chat_history = self._memory.get()
            chat_history_str_content = "\n".join([
            f"{msg.role}:{msg.content}"
            for msg in chat_history
            ])
        
            print("Chat History:", chat_history_str_content)
        except Exception as e:
            chat_history_str_content = ''
            print("Error at chat chistory:", e)
        
        # Get relevant documents
        retrieved_nodes = self._retriever.retrieve(query_str)
        print("@#######@",retrieved_nodes)
        # Prepare context from nodes
        context_chunks = []
        image_nodes = []
        
        for node in retrieved_nodes:
            # Handle text content
            if isinstance(node, NodeWithScore):
                # print(">>>>>>>>>>>",node.node.metadata)
                text = node.node.get_content(metadata_mode=MetadataMode.ALL)
                metadata = node.node.metadata
                source_info = f"\nSource: {metadata.get('pdf_name', 'Unknown')}, Page: {metadata.get('page_num', 'Unknown')}"
                context_chunks.append(text + source_info)
                
                # Check for images in metadata
                if "image_path" in node.node.metadata:
                    image_nodes.append(node)
            
            # Handle image nodes
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
        # Combine text context
        context_text = "\n\n".join(context_chunks)
        print("Context Text:",context_text)
        
        # Format prompt with context and history
        prompt = self._context_prompt.format(
            chat_history=chat_history_str_content,
            context_str=context_text,
            query_str=query_str
        )
        print("Final Prompt:", prompt)
        # Collect all image paths
        all_image_paths = []
        for node in image_nodes:
            if isinstance(node.node, ImageNode):
                if hasattr(node.node, 'image_path'):
                    all_image_paths.append(node.node.image_path)
            elif "image_path" in node.node.metadata:
                paths = node.node.metadata["image_path"]
                if isinstance(paths, list):
                    all_image_paths.extend(paths)
                else:
                    all_image_paths.append(paths)
        
        # Remove duplicates while preserving order
        all_image_paths = list(dict.fromkeys(all_image_paths))
        
        # Create image documents
        image_documents = self._create_image_documents(all_image_paths)
        # print(image_documents)
        if not image_documents:
            logger.warning("No valid images found in the retrieved nodes")
            image_documents = []  # Ensure we have an empty list if no images
        

        # Add text response with images
        text_response = self._llm.complete(
            prompt=prompt,
            image_documents=image_documents
        )
        
        final_response = str(text_response)
        print("Final response::",final_response)
        # Process and restructure the response
        processed_response = self.process_response(final_response)
        final_response_str = str(processed_response["result_response"]["explanation"])

        eval_result = faithfull_evaluator.evaluate(response=final_response_str,
                                                   contexts=context_chunks)
        print("Faithfull eval result:",eval_result.passing, eval_result.score)

        relevance_result = relavancy_evaluator.evaluate(query=query_str,
                                                        response=final_response_str,
                                                        contexts=context_chunks)
        print("Relevance Result:",relevance_result.passing, relevance_result.score)
        
        # print(">>>>>>>>>",processed_response["refrence"])
        page_number_retrived = [str(processed_response["refrence"][i]["page_number"].split(' ')[-1]) for i in range(len(processed_response["refrence"]))]
        pdf_name_retrived = [str(processed_response["refrence"][i]["pdf_name"]) for i in range(len(processed_response["refrence"]))]
        actual_doc_name = [str(processed_response["refrence"][i]["actual_doc_name"]) for i in range(len(processed_response["refrence"]))]
        document_type = [str(processed_response["refrence"][i]["doc_type"]) for i in range(len(processed_response["refrence"]))]
        print(page_number_retrived, pdf_name_retrived, actual_doc_name,document_type)
        
        self._memory.put(ChatMessage(role="user", content=f"{query_str}", timestamp=datetime.now()))
        self._memory.put(ChatMessage(role="assistant",content=f"{final_response_str}", timestamp=datetime.now()))

        if document_type == []:
            return Response("Can you ask a specific question to which document are you referring to", source_nodes=[])
        
        # Process source nodes and include base64 images
        seen_nodes = set()
        source_nodes_with_images = []
        # print("@@@@@",retrieved_nodes)
        for node in retrieved_nodes:
            if isinstance(node, NodeWithScore):
                # Get the unique identifier tuple
                # doc_type = node.node.metadata.get("document_type", "")
                # actual_pdf_name = node.node.metadata.get("actual_doc_name", "")
                pdf_name = node.node.metadata.get("pdf_name", "")
                page_num = node.node.metadata.get("page_num", 0)
                node_id = (pdf_name, page_num)
                
                if node_id not in seen_nodes and str(page_num) in page_number_retrived and str(pdf_name) in pdf_name_retrived:
                    seen_nodes.add(node_id)
                    node_data = {
                        "text": node.node.get_content(),
                        "metadata": node.node.metadata,
                        "source_link": f"[Source: {pdf_name}, Page: {page_num}]",
                    }
                    if "image_path" in node.node.metadata:
                        node_data["image_base64"] = get_base64_image(node.node.metadata['image_path'])
                    
                    source_nodes_with_images.append(node_data)
        
        # run_tree.end()
        return Response(final_response_str, source_nodes=source_nodes_with_images)


def get_base64_image(image_path: str) -> str:
    """Encode an image file at the given path as a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image data.
    """
    try:
        image_path = image_path.replace("\\", "/")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return ""

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Process a query using the multimodal conversational engine.

    Given a set of documents and a user query, this endpoint processes the query
    using the multimodal conversational engine and returns the response as a
    StreamingResponse.

    The response is a JSON object with the following keys:

    *   `response`: The response text from the GPT-4o model.
    *   `source_nodes`: A list of dictionaries representing the source nodes
        used to generate the response. Each dictionary contains the following
        keys:

        *   `text`: The text content of the source node.
        *   `metadata`: A dictionary containing the metadata of the source
            node, including:

            *   `pdf_name`: The name of the PDF file containing the source
                node.
            *   `page_num`: The page number of the source node in the
                original PDF file.
            *   `actual_doc_name`: The name of the original PDF file that
                the source node was extracted from.
            *   `document_type`: The type of document that the source node
                was extracted from (e.g. PDF, DOCX, XLSX).
            *   `image_base64`: The base64 encoded image data of the source
                node, if applicable.

    :param request: The QueryRequest object containing the user query and
        documents to process.
    :return: A StreamingResponse object containing the response as a JSON
        object.
    """
    async def event_stream():
        """
        Asynchronously process a user query and document inputs, yielding status updates and the final result.

        This generator function handles the processing of document inputs, initializes the index, and 
        communicates with a multimodal conversational engine to generate a response. It yields status 
        updates at various stages of processing and ultimately yields the final response as a JSON object.

        Yields:
            str: Status updates and the final response in the form of JSON-formatted strings.

        Raises:
            HTTPException: If there are no indexed documents or if any errors occur during processing.
        """
        global doc_type
        try:
            # inputfiles = [Path(ip[0]) for ip in request.input_files]
            print("inputfiles:",request.input_files, request.collection)

            yield "event: status\ndata: Processing documents...\n\n"
            if request.input_files != []:
                docs, doc_type = process_documents(request.input_files, request.collection)
                # print("<<<<<<<<<", docs, doc_type)
            else:
                docs = []
            
            yield "event: status\ndata:  Initializing index...\n\n"
            current_index = initialize_index(request.collection,docs, doc_type)
            if current_index is None:
                raise HTTPException(
                    status_code=400,
                    detail="No indexed documents found. Please index documents first."
                )
            
            # Initialize multimodal LLM
            
            yield "event: status\ndata: Processing documents using GPT-4o...\n\n"
            gpt_4v = OpenAIMultiModal(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_new_tokens=4096
            )
            
            # Initialize memory buffer
            memory_key = f"memory_{request.user_id}"
            if memory_key not in chat_history:
                chat_history[memory_key] = ChatSummaryMemoryBuffer.from_defaults(llm=summarizer_llm,
                                                                                token_limit=2,
                                                                                tokenizer_fn=tokenizer_fn,)
            
            # Create custom query engine
            query_engine = MultiModalConversationalEngine()
            query_engine.intialize_engine(
                retriever=current_index.as_retriever(similarity_top_k=6),
                multi_modal_llm=gpt_4v,
                memory_buffer=chat_history[memory_key]
            )
            
            # Get response
            response = query_engine.custom_query(request.query)
            yield "event: status\ndata: Response received from GPT-4o...\n\n"
            final_result = {
                "response": str(response),
                "source_nodes": [
                    {
                        "text": node["text"],
                        "metadata": {
                            "pdf_name": node["metadata"].get("pdf_name", "Unknown"),
                            "page_num": node["metadata"].get("page_num", "Unknown"),
                            "actual_doc_name": node["metadata"].get("actual_doc_name", "Unknown"),
                            "document_type": node["metadata"].get("document_type", "Unknown"),
                            "image_base64": node.get("image_base64", None)
                        }
                    }
                    for node in response.source_nodes
                ]
            }

            yield f"event: final\ndata: {json.dumps(final_result, separators=(',', ':'))}\n\n"
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Get chat history for a specified user (personal use case)
@app.get("/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    """
    Retrieve chat history for a specified user.

    Args:
        user_id (str): The unique identifier for the user whose chat history is to be retrieved.

    Returns:
        dict: A dictionary containing a list of chat messages for the specified user. If the user
              does not have any chat history, an empty list is returned.
    """

    if user_id not in chat_history:
        return {"messages": []}
    return {"messages": chat_history[user_id]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
