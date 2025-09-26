# Elastic Search AI Agent
Creating an AI agent that can answer questions about any given GitHub repository can help users better understand the codebase as well as retrieve examples and best practices without the user having to do endless manual search. This repository aims to create an AI agent that can any question about the Elastic Search repository.


# Step 0 : Set up

- Create virtual environment. I use conda.

```bash
conda create --name ai-agent python=3.11
```

- Create jupyter kernel and link it to the virtual environment of choice.

```bash
pip install ipykernel
python -m ipykernel install --user --name ai-agent --display-name ai-agent
```

- Install requirements file

```bash
pip install requirements.txt
```

- Install the `uv` package manager

```bash
pip install uv
```

- Initialize the `uv` package manager
```bash
uv init
```

# Step 1 : Ingestion

- Download and unzip a GitHub repository in the form of a zip file
- Select `.md` and `.mdx` files only
- You can download the Elastic Search repository either on the command line or in a Jupyter notebook, and the repository will be downloaded to your working directory.

CLI: 

```bash
python3 ingest.py --repo_owner='elastic' --repo_name='elasticsearch'
```

Jupyter notebook:

```python
from ingest import read_repo_data
read_repo_data(repo_owner='elastic', repo_name='elasticsearch')
```

# Step 2 : Processing

- Depending on the kind of data, there are various ways of processing the data.
    - Simple overlapping chunks, paragraph and section splitting, token-based, or AI-powered splitting using some LLM
    - Overlapping lexical split is one of the simplest methods and is often sufficient. It is recommended to try the simplest approach first before moving to a more complex method such as using a LLM.
- A sliding window method, or overlapping between chucks, was used to create a list of chunks for this project. To implement the processing step:

```python
from chunking import create_chunks
chunks_list = create_chunks(repo_docs)
```

# Step 3: Indexing

- Indexing the data and putting it in a search engine help the agent quickly retrieve relevant data when the user asks a question.
- Depending on the size of the data, there are three primary indexing methods:
    - Lexical: the agent matches keywords in the query to data in the search engine
    - Vector or semantic search: embeddings, or the numerical representation of text, are created from both the data and the question. Embeddings maintain semantic meaning, such that synonyms have similar embeddings while antonyms have dissimilar embeddings.
    - In most cases, lexical search suffices for small datasets while vector search is more suitable for medium to large datasets. Because the Elastic Search repository is rather large, vector search was implemented. At this point, the premature agent is already ready to answer questions in the following way:
    
    ```python
    from search import vector_search
    
    query = 'What is elastic net?'
    vector_search(chunks_list, query)
    ```
    
    The above results in the following JSON output:
    ![](json-example-output.png)