from src.data.encoding import encoding
from src.data.cleaning import cleaning
from src.data.load_data import load_data
from src.vectors_store.chromadb_client import client
from sklearn.model_selection import train_test_split
collection_vect= client.get_or_create_collection('pdf_vectors2')
print(type(collection_vect))
print(collection_vect.get_indexing_status)

def split(df):
    results = client.get_collection('embedding_db').get(include=['embeddings'])
    df['embeddings']= list(results['embeddings'])
    df.drop(['fusion_email','tags','answer','language'], axis=1, inplace=True)
    X= df.drop('type', axis=1)
    y= df['type']
    X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.3,train_size=0.7,shuffle=True, random_state=42)
    return X_train,X_test,y_train, y_test


df= load_data()
clean_df= cleaning(df)
encoded_df= encoding(clean_df)
print(split(encoded_df))