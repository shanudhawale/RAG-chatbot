import os
import json
import pandas as pd
import base64
import tiktoken
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llama_index.core.query_engine import CustomQueryEngine, SimpleMultiModalQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode, TextNode, ImageDocument
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.memory import ChatMemoryBuffer , ChatSummaryMemoryBuffer
from llama_index.core import Settings
from backend.config import  logger, chat_history, summarizer_llm, tokenizer_fn

import mlflow
from mlflow.metrics import latency
from mlflow.metrics.genai import faithfulness, relevance
mlflow.llama_index.autolog()
mlflow.set_experiment("llama-index-pdf-qa-rag")
# mlflow.set_tracking_uri("http://0.0.0.0:5000")



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

This example can help you understand the output format.
```json
{
    "result_respnse": [
        {
            "explanation1": "The document discusses the process of annotating scientific articles, focusing on the methodology used to ensure clarity and specificity in labeling.",
            "explanation2": "It details the phases of data selection and preparation, emphasizing the importance of having a diverse and well-prepared dataset for effective annotation.",
            "explanation3": "The document outlines the roles of different teams involved in the annotation process, including a small team of experts and a larger group of dedicated annotators.",
            "explanation4": "It also highlights the challenges faced in distinguishing between documents and the criteria used to ensure that all documents are free to use."
        }
    ],
    "refrence": [
        {
            "page_number": "9",
            "pdf_name": "2408.09869v5.pdf",
            "actual_doc_name": "2408.09869v5.pdf",
            "doc_type": "pdf"
        }
    ]
}
```
ONLY return a valid JSON object (no other text is necessary). The JSON MUST conform to the XML format, including any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise.
"""

CONTEXT_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

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

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime

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
        # print("Final Prompt:", prompt)
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
        logger.info(f"Final Response: {final_response}")
        # Process and restructure the response
        processed_response = self.process_response(final_response)
        final_response_str = str(processed_response["result_response"]["explanation"])

        # eval_result = faithfull_evaluator.evaluate(response=final_response_str,
        #                                            contexts=context_chunks)
        # print("Faithfull eval result:",eval_result.passing, eval_result.score)

        # relevance_result = relavancy_evaluator.evaluate(query=query_str,
        #                                                 response=final_response_str,
        #                                                 contexts=context_chunks)
        # print("Relevance Result:",relevance_result.passing, relevance_result.score)
        try:
            logger.info("Logging the experiment to MLflow...")
            eval_data = pd.DataFrame({
                "inputs": [query_str],
                "context": [context_chunks],
                "predictions": [final_response_str],
                })
            
            with mlflow.start_run() as run:
                # mlflow.log_metric("Input Token", len(tokenizer_fn(prompt)))
                # mlflow.log_metric("Output_token", len(tokenizer_fn(text_response)))
                results = mlflow.evaluate(
                    data=eval_data,
                    targets="context",
                    predictions="predictions",
                    extra_metrics=[faithfulness(model="openai:/gpt-4o-mini"), relevance(model="openai:/gpt-4o-mini"), latency()],
                    evaluators="default",
                )
        finally:
            mlflow.end_run()

        # print(">>>>>>>>>",processed_response["refrence"])
        page_number_retrived = [str(processed_response["refrence"][i]["page_number"].split(' ')[-1]) for i in range(len(processed_response["refrence"]))]
        pdf_name_retrived = [str(processed_response["refrence"][i]["pdf_name"]) for i in range(len(processed_response["refrence"]))]
        actual_doc_name = [str(processed_response["refrence"][i]["actual_doc_name"]) for i in range(len(processed_response["refrence"]))]
        document_type = [str(processed_response["refrence"][i]["doc_type"]) for i in range(len(processed_response["refrence"]))]
        print(page_number_retrived, pdf_name_retrived, actual_doc_name,document_type)
        
        self._memory.put(ChatMessage(role="user", content=f"{query_str}", timestamp=datetime.now()))
        self._memory.put(ChatMessage(role="assistant",content=f"{final_response_str}", timestamp=datetime.now()))

        if document_type == []:
            logger.warning("No valid documents found in the retrieved nodes")
            return Response("Can you ask a specific question to which document are you referring to", source_nodes=[])
        
        # Process source nodes and include base64 images
        seen_nodes = set()
        source_nodes_with_images = []
        # print("@@@@@",retrieved_nodes)
        logger.info(f"Retrieved nodes: {retrieved_nodes}")
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
