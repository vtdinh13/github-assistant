import io
import zipfile
import requests
import frontmatter


from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np


embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
v_index = VectorSearch(keyword_fields = [])



def read_repo_data(repo_owner:str, repo_name:str) -> list:
    """
    Download and parse all markdown files from a GitHub repository.
    
    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name
    
    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not filename.endswith(('.md', '.mdx')):
            continue
    
        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                data['filename'] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    zf.close()

    
    return repository_data   

def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result

def chunk_documents(docs:list, size=2000, step=1000):
    doc_chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        chunks = sliding_window(doc_content, size, step)
        for chunk in chunks:
            chunk.update(doc_copy)
        doc_chunks.extend(chunks)
    return doc_chunks



def create_doc_embeddings(chunks:list):
    embeddings = []

    for d in tqdm(chunks):
        v = embedding_model.encode(d['chunk'])
        embeddings.append(v)

    return np.array(embeddings)


def create_vector_index(chunks:list):
    emb_array = create_doc_embeddings(chunks)
    return v_index.fit(emb_array, chunks)



def text_embedding_search(query:str):
    query_embedding = embedding_model.encode(query)
    return v_index.search(query_embedding, num_results=5)

def index_data(repo_owner, repo_name, filter=None, chunk=False, chunking_params=None):
    docs = read_repo_data(repo_owner, repo_name)

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]
    
    if chunk:
        if chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        docs = chunk_documents(docs, **chunking_params)
    
    vector_index = create_vector_index(docs)
    return vector_index
    
