from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np


embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

def create_doc_embeddings(chunks:list):
    embeddings = []

    for d in tqdm(chunks):
        v = embedding_model.encode(d['chunk'])
        embeddings.append(v)

    embeddings_array = np.array(embeddings)
    return embeddings_array

def create_query_embedding(query:str):
    return embedding_model.encode(query)
    
def vector_search(chunks:list, query:str):

    query_embedding = create_query_embedding(query)
    embeddings_array = create_doc_embeddings(chunks)

    vindex = VectorSearch(keyword_fields = [])
    
    vindex.fit(embeddings_array, chunks)
    return vindex.search(query_embedding)

    
    



def create_text_search(chunks:list):
    index = Index(
    text_fields=["chunk", "title", "description", "filename"],
    keyword_fields=[]
)
    return index.fit(chunks)