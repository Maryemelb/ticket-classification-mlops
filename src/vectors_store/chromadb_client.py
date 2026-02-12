import chromadb
import os
from chromadb.config import Settings

db_path = os.getenv("CHROMA_PATH", "./vector_db/chroma_storage/embedding_db")

client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(allow_reset=True)
    )
