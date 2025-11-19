# GitHub Assistant


Built to solve the challenge of navigating large codebases, this repo creates an AI agent that extracts knowledge directly from any given GitHub repo, making documentation instantly searchable and accessible. The codebase is designed to download the repo, index its Markdown content with embeddings, and use an LLM agent to answer your questions with file references.

## What it does
- Downloads a GitHub repo ZIP (main branch) and extracts only `.md`/`.mdx` files (`ingest.py`).
- Embeds content with `sentence-transformers` (`multi-qa-distilbert-cos-v1`) and builds a `minsearch` vector index.
- Exposes a search tool to a PydanticAI's agent class (using `gpt-4o-mini`) that cites GitHub file paths in responses (`search_agent.py`, `search_tools.py`).
- Offers both a CLI chat loop (`main.py`) and a Streamlit UI (`app.py`).
- Logs every interaction to JSON in `logs/` for review or evaluation (`logs.py`, `eval.py`).

## Setup
1) Create and activate a Python 3.11+ virtual environment. Example with `venv`:
```bash
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Provide an API key for the chat model (OpenAI-compatible). For example:
```bash
export OPENAI_API_KEY=your_key_here
```

## Running the agent
### CLI
Start an interactive session targeting a GitHub repo:
```bash
python main.py --repo_owner elastic --repo_name elasticsearch
```
Type questions; enter `stop` to exit. The script will download the repository, build the vector index, and answer using the search tool.

### Streamlit UI
Launch the web app:
```bash
streamlit run app.py
```
In the sidebar, set the repo owner and name (e.g., `elastic` / `elasticsearch`), click **Initialize / Rebuild Index**, then ask questions in the chat box.

## How it works
1) **Ingestion** – Downloads the repo ZIP from GitHub and parses Markdown files using frontmatter into records.
2) **Indexing** – Creates sentence-transformer embeddings and fits a `minsearch.VectorSearch` index (top‑5 results used by default).
3) **Agent** – Built with PydanticAI's `Agent` class using OpenAI's `gpt-4o-mini`; it calls the search tool before answering and injects GitHub blob links for cited files.
4) **Logging** – All conversations are written to timestamped JSON files in `logs/` for auditing or evaluation.

## Extras
- **Chunking:** `ingest.index_data(..., chunk=True, chunking_params={...})` will split documents with a sliding window before indexing.
- **Synthetic QA & eval:** `question_generation.py` can sample repo content to generate questions; `eval.py` scores logged responses against a checklist.
