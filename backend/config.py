"""Configuration and environment settings."""
import os
import logging
import tiktoken
import chromadb
import nest_asyncio
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.openai import OpenAI
from backend.embeddings import InstructorEmbeddings
from llama_index.core import Settings

# Initialize logging
logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler("processing_logs.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Global variables
index_cache: Dict[str, VectorStoreIndex] = {}
chat_history: Dict[str, ChatSummaryMemoryBuffer] = {}
current_index: VectorStoreIndex = None

# Initialize ChromaDB
CHROMA_DB_PATH = "/app/backend/chroma_db2"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")

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

# Instantiate InstructorEmbeddings
embed_model = InstructorEmbeddings(embed_batch_size=2)
Settings.embed_model = embed_model
Settings.chunk_size = 512


# LLM initialization
llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.llm = llm

# Summarizer and tokenizer for chat history
summarizer_llm = llm
tokenizer_fn = tiktoken.encoding_for_model("gpt-4o-mini").encode
