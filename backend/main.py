import os
import json
from fastapi import FastAPI, HTTPException, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.memory import ChatSummaryMemoryBuffer
from backend.document_processing import process_documents
from backend.indexing import initialize_index
from backend.query_engine import MultiModalConversationalEngine
from backend.config import logger, chat_history, summarizer_llm, tokenizer_fn

class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="User query")
    data_path: Optional[str] = Field(default="data", description="Path to document directory")
    user_id: str = Field(..., description="Unique user identifier")
    input_files: List[tuple] = Field(..., description="list of files")
    collection: str = Field(..., description="name of collection for vector db")


# Initialize FastAPI app
app = FastAPI(title="RAG API")

gpt_4v = OpenAIMultiModal(model="gpt-4o-mini",
                          api_key=os.getenv("OPENAI_API_KEY"),max_new_tokens=4096)

# Create custom query engine
query_engine = MultiModalConversationalEngine()

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
        used from NodeWithScore. Each dictionary contains the following
        keys:

        *   `text`: The markdown content of the source node.
        *   `metadata`: A dictionary containing the metadata of the source
            node, including:

            *   `pdf_name`: The name of the source file if pdf. 
            *   `page_num`: The page number
            *   `actual_doc_name`: The name of the original PDF file when it was 
            uploaded
            *   `document_type`: The type of document that the source node
                was extracted from (e.g. PDF, DOCX, XLSX).
            *   `image_base64`: The base64 encoded image data of the source
                node, for pdf only.

    :param request: The QueryRequest object containing the user query and
        documents to process.
    :return: A StreamingResponse object containing the response as a JSON
        object at each step
    """
    async def event_stream():
        """
        Asynchronously process a user query and document inputs, yielding status updates and the final result.

        Yields:
            str: Status updates and the final response in the form of JSON-formatted strings.
        """
        global doc_type
        try:
            # inputfiles = [Path(ip[0]) for ip in request.input_files]
            print("inputfiles:",request.input_files, request.collection)

            yield "event: status\ndata: Processing documents...\n\n"
            if request.input_files != []:
                docs, doc_type = process_documents(request.input_files, request.collection)
                logger.info(f"Processed documents with input paths: {request.input_files} and output dir: {request.collection}")
                # print("<<<<<<<<<", docs, doc_type)
            else:
                docs = []
            
            yield "event: status\ndata:  Initializing index...\n\n"
            current_index = initialize_index(request.collection,docs, doc_type)
            logger.info(f"Initialized index for collection: {request.collection}")
            if current_index is None:
                raise HTTPException(
                    status_code=400,
                    detail="No indexed documents found. Please index documents first."
                )
            
            # Initialize multimodal LLM
            
            yield "event: status\ndata: Processing documents using GPT-4o...\n\n"
            # Initialize memory buffer
            memory_key = f"memory_{request.user_id}"
            if memory_key not in chat_history:
                logger.info(f"Initializing memory buffer for user: {request.user_id}")
                chat_history[memory_key] = ChatSummaryMemoryBuffer.from_defaults(llm=summarizer_llm,
                                                                                token_limit=2,
                                                                                tokenizer_fn=tokenizer_fn,)
                
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
            logger.info(f"Response received from GPT-4o: {final_result}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)