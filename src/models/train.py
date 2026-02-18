
import asyncio
from src.data.cleaning import cleaning
from src.data.encoding import encoding
from src.data.load_data import load_data
from src.features.embedding_generator import generate_embeddings
from src.models.split import split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
def train(X_train , X_test , y_train , y_test):
    model= LinearSVC(max_iter=1000 , class_weight="balanced")
    model.fit(X_train , y_train)
    prediction= model.predict(X_test)
    report = classification_report(y_test , prediction , output_dict=True)
    print(report)
    return report

# df= load_data()
# clean_df= cleaning(df)
# encoded_df= encoding(clean_df)
# asyncio.run(generate_embeddings(encoded_df))
# X_train,X_test,y_train, y_test= split(encoded_df)
# train(X_train,X_test,y_train, y_test)

async def main():
    df = load_data()
    clean_df = cleaning(df)
    encoded_df = encoding(clean_df)
    await generate_embeddings(encoded_df)  # async embedding generation
    X_train, X_test, y_train, y_test = split(encoded_df)
    train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())