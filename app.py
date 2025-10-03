# app.py
import asyncio
import contextlib
import inspect
import textwrap
from typing import AsyncIterator, Callable, Iterator, Optional

import streamlit as st

# Your modules
import ingest
import search_agent
import logs


# ---------- App setup ----------
st.set_page_config(page_title="GitHub Assistant", page_icon="🤖", layout="wide")
st.title("GitHub Assistant")
st.markdown(
    """
    <div style='font-size:16px; color: gray;'>
        Chat with an AI agent trained on a GitHub repository.  
        Enter the <b>owner</b> and <b>repository name</b> in the sidebar.  
        Example: <code>https://github.com/elastic/elasticsearch</code>, so <b>owner</b> = <code>elastic</code> and <b>name</b> = <code>elasticsearch</code>.
        Then ask your questions and have fun exploring 🎉 !
    </div>
    """,
    unsafe_allow_html=True
)



# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "user"|"assistant"|"system", "content": str}]
if "agent" not in st.session_state:
    st.session_state.agent = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "repo_owner" not in st.session_state:
    st.session_state.repo_owner = ""
if "repo_name" not in st.session_state:
    st.session_state.repo_name = ""


# ---------- Helpers ----------
def initialize_index(repo_owner: str, repo_name: str):
    st.write(f"🔧 Initializing index for **{repo_owner}/{repo_name}** …")
    index = ingest.index_data(repo_owner, repo_name)
    st.success("✅ Data indexing completed!")
    return index

def initialize_agent(index, repo_owner: str, repo_name: str):
    st.write("🧠 Initializing agent …")
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    st.success("✅ Agent ready!")
    return agent

def _chunk_text_for_streaming(text: str, chunk_size: int = 32) -> Iterator[str]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]

async def _run_agent_async(agent, prompt: str):
    """
    Run the agent asynchronously (non-streaming) and return the response object.
    Matches your CLI code (agent.run is awaited).
    """
    return await agent.run(user_prompt=prompt)

async def _try_native_stream(agent, prompt: str) -> Optional[AsyncIterator[str]]:
    """
    Try to obtain a native streaming iterator from the agent if it exists.
    Supports a few common names/variants. Returns None if not available.
    """
    # Candidate attribute/method names that might yield tokens or deltas.
    candidates = ["astream", "stream", "stream_run"]
    for name in candidates:
        if hasattr(agent, name):
            method = getattr(agent, name)
            if inspect.iscoroutinefunction(method):
                # async def astream(user_prompt=...): yields strings
                async def _aiter():
                    async for part in method(user_prompt=prompt):
                        # Allow tuple/dict events; try to extract string if present.
                        if isinstance(part, str):
                            yield part
                        elif isinstance(part, dict):
                            # Common shapes: {"delta": "..."} / {"text": "..."}
                            for key in ("delta", "text", "content", "token"):
                                if key in part and isinstance(part[key], str):
                                    yield part[key]
                                    break
                        else:
                            # Fallback to repr
                            yield str(part)
                return _aiter()
            elif callable(method):
                # Sync generator?
                result = method(user_prompt=prompt)
                if inspect.isgenerator(result):
                    async def _wrap_sync_gen(gen) -> AsyncIterator[str]:
                        for part in gen:
                            yield str(part)
                    return _wrap_sync_gen(result)
    return None

def _ui_stream_write(generator: Iterator[str] | AsyncIterator[str]) -> str:
    """
    Stream chunks into a single chat message area and return the final full text.
    Works with sync or async generators.
    """
    container = st.empty()
    acc = ""

    async def _async_consume(ait: AsyncIterator[str]):
        nonlocal acc
        async for chunk in ait:
            acc += chunk
            container.markdown(acc)

    def _sync_consume(it: Iterator[str]):
        nonlocal acc
        for chunk in it:
            acc += chunk
            container.markdown(acc)

    # If Streamlit's st.write_stream exists, prefer it for smoother UX
    if hasattr(st, "write_stream"):
        # st.write_stream expects a sync iterator of strings
        # If we got an async iterator, adapt to a sync generator that runs the loop
        if inspect.isasyncgen(generator):
            # Consume async gen into a thread-safe queue-like sync iterator
            # Simpler approach: run the async consumer to completion,
            # but we still want UI updates, so fall back to manual container updates
            # and return the accumulated text.
            asyncio.run(_async_consume(generator))  # blocks until done
            return acc
        else:
            # Sync generator: Streamlit will render progressively
            final_text = st.write_stream(generator)
            # st.write_stream returns the concatenated text
            return final_text if isinstance(final_text, str) else acc

    # Manual streaming fallback
    if inspect.isasyncgen(generator):
        asyncio.run(_async_consume(generator))
    else:
        _sync_consume(generator)
    return acc


# ---------- Sidebar: initialization ----------
with st.sidebar:
    st.header("⚙️ Setup")
    st.session_state.repo_owner = st.text_input("Repo owner", value=st.session_state.repo_owner)
    st.session_state.repo_name = st.text_input("Repo name", value=st.session_state.repo_name)
    init_clicked = st.button("Initialize / Rebuild Index", type="primary", use_container_width=True)

    if init_clicked:
        if not st.session_state.repo_owner or not st.session_state.repo_name:
            st.error("Please provide both **repo owner** and **repo name**.")
        else:
            with st.spinner("Indexing & initializing the agent…"):
                index = initialize_index(st.session_state.repo_owner, st.session_state.repo_name)
                st.session_state.agent = initialize_agent(index, st.session_state.repo_owner, st.session_state.repo_name)
                st.session_state.index_ready = True
                # Optional system message note
                st.session_state.messages = [{"role": "system", "content": f"Agent initialized for {st.session_state.repo_owner}/{st.session_state.repo_name}."}]

    st.markdown("---")
    st.caption("Tip: You can rebuild the index anytime after editing owner and name of the repo.")


# ---------- Chat history render ----------
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "assistant" else ("user" if m["role"] == "user" else "system")):
        st.markdown(m["content"])


# ---------- Input ----------
prompt = st.chat_input("Ask a question about the repository…")
if prompt:
    if not st.session_state.index_ready or st.session_state.agent is None:
        st.error("Please initialize the agent in the sidebar first.")
        st.stop()

    # Show the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent

    # Assistant message container (will receive streamed content)
    with st.chat_message("assistant"):
        # 1) Try native streaming if the agent supports it
        native_stream: Optional[AsyncIterator[str]] = asyncio.run(_try_native_stream(agent, prompt))
        if native_stream is not None:
            final_text = _ui_stream_write(native_stream)
            # We can’t assume native stream yields a "response object", so do a non-stream read after for logging if needed
            # If logging requires new_messages(), do a follow-up non-stream run in a best-effort, non-blocking way.
            with contextlib.suppress(Exception):
                resp = asyncio.run(_run_agent_async(agent, prompt))
                logs.log_interaction_to_file(agent, resp.new_messages())
        else:
            # 2) Fallback: run once (non-stream) and stream UI chunks
            with st.spinner("Thinking…"):
                resp = asyncio.run(_run_agent_async(agent, prompt))
            final_text = _ui_stream_write(_chunk_text_for_streaming(str(resp.output)))
            # Persist logs using your helper
            with contextlib.suppress(Exception):
                logs.log_interaction_to_file(agent, resp.new_messages())

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": final_text})
