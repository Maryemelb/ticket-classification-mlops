from sklearn.discriminant_analysis import StandardScaler
# from src.data.encoding import encoding
# from src.data.cleaning import cleaning
# from src.data.load_data import load_data
from src.vectors_store.chromadb_client import client
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
collection_vect= client.get_or_create_collection('pdf_vectors2')
print(type(collection_vect))
print(collection_vect.get_indexing_status)

def split(df):
    results = client.get_collection('embedding_db').get(include=['embeddings'])
    embeddings= np.array(results['embeddings'])
    y= df['type']

    x_others= df.drop(['fusion_email','tags','answer','language', 'type'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).values

    X = np.hstack([embeddings, x_others]) 
    #Normalis
    scaler = StandardScaler()
    X= scaler.fit_transform(X)
    X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.3,train_size=0.7,shuffle=True, random_state=42, stratify=y)
    return X_train,X_test,y_train, y_test


# df= load_data()
# clean_df= cleaning(df)
# encoded_df= encoding(clean_df)
# print(split(encoded_df))