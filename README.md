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
    {'start': 0,
  'chunk': "When working on an AI system, you need test data to run automated evaluations for quality and safety. A test dataset is a structured set of test cases. It can contain:\n\n* Just the inputs, or\n* Both inputs and expected outputs (ground truth).\n\nYou can use this test dataset to:\n\n* Run **experiments** and track if changes improve or degrade system performance.\n* Run **regression testing** to ensure updates don’t break what was already working.\n* **Stress-test** your system with complex or adversarial inputs to check its resilience.\n\n![](/images/synthetic/synthetic_experiments_img.png)\n\nYou can create test datasets manually, collect them from real or historical data, or generate them synthetically. While real data is best, it is not always available or sufficient to cover all cases. Public LLM benchmarks help with general model comparisons but don’t reflect your specific use case. Manually writing test cases takes time and effort.\n\n**Synthetic data helps here**. It’s especially useful when you are:\n\n* You're starting from scratch and don’t have real data.\n* You need to scale a manually designed dataset with more variation.\n* You want to test edge cases, adversarial inputs, or system robustness.\n* You're evaluating complex AI systems like RAG and AI agents.\n\n![](/images/synthetic/synthetic_adversarial_img.png)\n\nSynthetic data is not a replacement for real data or expert-designed tests — it’s a way to add variety and speed up the process. With synthetic data you can:\n\n* Quickly generate hundreds structured test cases.\n* Fill gaps by adding missing scenarios and tricky inputs.\n* Create controlled variations to evaluate specific weaknesses.\n\nIt’s a practical way to expand your evaluation dataset efficiently while keeping human expertise focused on high-value testing.\n\nSynthetic data can also work for **complex AI systems** where designing test cases is simply difficult. For example, in RAG evaluation, synthetic data helps create input-output datasets from knowledge bases. I",
  'title': 'Why synthetic data?',
  'description': 'When do you need synthetic data in LLM evaluations.',
  'filename': 'docs-main/synthetic-data/why_synthetic.mdx'},
 {'start': 1000,
  'chunk': " you are:\n\n* You're starting from scratch and don’t have real data.\n* You need to scale a manually designed dataset with more variation.\n* You want to test edge cases, adversarial inputs, or system robustness.\n* You're evaluating complex AI systems like RAG and AI agents.\n\n![](/images/synthetic/synthetic_adversarial_img.png)\n\nSynthetic data is not a replacement for real data or expert-designed tests — it’s a way to add variety and speed up the process. With synthetic data you can:\n\n* Quickly generate hundreds structured test cases.\n* Fill gaps by adding missing scenarios and tricky inputs.\n* Create controlled variations to evaluate specific weaknesses.\n\nIt’s a practical way to expand your evaluation dataset efficiently while keeping human expertise focused on high-value testing.\n\nSynthetic data can also work for **complex AI systems** where designing test cases is simply difficult. For example, in RAG evaluation, synthetic data helps create input-output datasets from knowledge bases. In AI agent testing, it enables multi-turn interactions across different scenarios.",
  'title': 'Why synthetic data?',
  'description': 'When do you need synthetic data in LLM evaluations.',
  'filename': 'docs-main/synthetic-data/why_synthetic.mdx'},
 {'start': 1000,
  'chunk': "a set of Tests to evaluate your data or AI system. Each Report Preset has this option. \n\nEnable it by setting `include_tests=True` on the Report level. (Default: False).\n\n```python\nreport = Report([\n    DataSummaryPreset(),\n],\ninclude_tests=True)\n```\n\nFor example, while the `DataSummaryPreset()` Report simply shows descriptive stats of your data, adding the Tests will additionally run multiple checks on data quality and expected column statistics.\n\nThe automatic Test conditions can either\n* be derived from a reference dataset, or\n* use built-in heuristics.\n\n**Using reference**. When you provide a reference dataset, Tests compare the new data against it:\n\n```Python\nmy_eval = report.run(eval_data_1, eval_data_2) # eval_data_2 is reference\n```\n\nFor example, the check on missing values will validate if the current share of missing values is within +/-10% of the reference.\n\n<Note>\nNote that in this case the order matters: the first `eval_data_1` is the current data you evaluate, the second `eval_data_2` is the reference dataset you consider as a baseline and use to generate test conditions.\n</Note>\n\n**Using heuristics**. Without reference, Tests use predefined rules:\n\n```Python\nmy_eval = report.run(eval_data_1, None) # no reference data\n```\n\nIn this case, the missing values Test simply expects 0% missing values. Similarly, classification accuracy Test will compare the performance against a dummy model, etc. Some metrics (like min/max/mean values) don't have default heuristics.\n\n<Info>\n  **How to check Test defaults?** Consult the [All Metrics](/metrics/all_metrics) reference table.\n</Info>\n\n### Individual Tests with defaults\n\nPresets are great for a start or quick sanity checks, but often you'd want to select specific Tests. For example, instead of running checks on all value statistics, validate only mean or max.\n\nYou can pick the Tests while still using default conditions.\n\n**Select Tests**. List the individual Metrics, and choose the the `include_Tests` option:\n\n```Py",
  'title': 'Tests',
  'description': 'How to run conditional checks.',
  'filename': 'docs-main/docs/library/tests.mdx'},