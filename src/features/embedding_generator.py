from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from src.data.cleaning import cleaning
from src.vectors_store.chromadb_client import client
from sklearn.preprocessing import normalize
from src.data.encoding import encoding
from src.data.load_data import load_data
import numpy as np
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
collection_vect = client.get_or_create_collection(
        name="embedding_db"
    )
def generate_embeddings(df):
    columns = ['fusion_email']
    texts = df[columns].fillna("").apply(lambda row: " ".join([" ".join(x) if isinstance(x, list) else str(x) for x in row]), axis=1).tolist()

    batch_size = 128  # Increased from 50; 128-256 is usually the "sweet spot"
    print('start3')
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print('start')
        # Embed ONLY this batch
        batch_vectors = embedding_model.embed_documents(batch_texts)
        
        # Normalize only this batch
        batch_vectors = normalize(np.array(batch_vectors), norm="l2")
        
        batch_ids = [str(j) for j in range(i, i+len(batch_texts))]
        
        collection_vect.add(
            documents=batch_texts,
            embeddings=batch_vectors.tolist(),
            ids=batch_ids
        )
        print(f"Processed up to row {i + len(batch_texts)}")

df= load_data()
clean_df= cleaning(df)
encoded_df= encoding(clean_df)
embed= generate_embeddings(encoded_df)
