from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.cleaning import cleaning
from src.data.encoding import encoding
from src.features.embedding_generator import generate_embeddings
from sklearn.svm import LinearSVC
def build_pipeline():
    pipeline= Pipeline([
        ('cleaning', FunctionTransformer(cleaning)),
        ('encoding', FunctionTransformer(encoding)),
        ('embedding', FunctionTransformer(generate_embeddings)),
        ('model', LinearSVC(max_iter=1000))
    ])
    return pipeline