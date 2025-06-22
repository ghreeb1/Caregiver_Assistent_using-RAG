# config.py
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory relative to BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "data")

# Path for the FAISS vector database (THIS SHOULD BE A DIRECTORY)
DB_FAISS_PATH = os.path.join(BASE_DIR, "faiss_index") # Changed from .bin to a directory name

# Ollama model to use
OLLAMA_MODEL = "llama3.2" # Make sure this model is downloaded with 'ollama pull llama3.2'

# Embedding model to use for RAG
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking parameters for data ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200