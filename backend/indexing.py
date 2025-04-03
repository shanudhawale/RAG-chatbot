import logging
from typing import Optional, List, Dict, Any
from llama_index.core import StorageContext, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import SummaryIndex
import chromadb
import os
from datetime import datetime
from pathlib import Path
from backend.config import db, embed_model, logger , node_parser, o1_llm



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
            logger.info(f"Creating new ChromaDB index for document type: {doc_type}")
            if doc_type == "xlsx":
                nodes = node_parser.get_nodes_from_documents(docs)
                base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
                current_index = VectorStoreIndex(nodes=base_nodes + objects, llm=o1_llm, storage_context=storage_context,)
                # Use explicit service context with embed_model for SummaryIndex
            else:
                current_index = VectorStoreIndex.from_documents(
                    docs, 
                    storage_context=storage_context, 
                    embed_model=embed_model
                )
        else:
            print(f"Loading existing ChromaDB index for document type: {doc_type}")
            logger.info(f"Loading existing ChromaDB index for document type: {doc_type}")
            if doc_type == "xlsx" and docs:
                nodes = node_parser.get_nodes_from_documents(docs)
                base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
                current_index = VectorStoreIndex(nodes=base_nodes + objects, llm=o1_llm, storage_context=storage_context,)

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