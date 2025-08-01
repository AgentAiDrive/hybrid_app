import streamlit as st
import openai, json, os
import numpy as np
from pydantic import BaseModel
from time import sleep

# Optional PDF parsing
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# --------------------
# 📦 Config & State
# --------------------
SOURCES_FILE = "sources.json"
PROFILE_FILE = "profiles.json"
MEMORY_FILE  = "memory.json"
CHAT_FILE    = "chat_history.json"
INDEX_FILE   = "index.json"

# Load OpenAI key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# --------------------
# 🔧 Helpers: JSON Persistence
# --------------------
@st.cache_data
def load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON from {path}, resetting to default.")
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# --------------------
# 📚 1. Domain Sources
# --------------------
sources = load_json(SOURCES_FILE, {"domains": []})
st.sidebar.header("1. Domain Sources")
with st.sidebar.expander("Edit Domain-Agnostic Sources", expanded=True):
    new_src = st.sidebar.text_input("Add a source (book/expert/style/URL)")
    if st.sidebar.button("➕ Add Source") and new_src:
        sources["domains"].append(new_src)
        save_json(SOURCES_FILE, sources)
        st.rerun()
    st.sidebar.markdown("---")
    for i, src in enumerate(sources["domains"]):
        cols = st.sidebar.columns((0.85, 0.15))
        cols[0].write(src)
        if cols[1].button("✖", key=f"del_src_{i}"):
            sources["domains"].pop(i)
            save_json(SOURCES_FILE, sources)
            st.rerun()

# --------------------
# 🧱 2. Profile Builder
# --------------------
st.sidebar.header("2. Build Persona & System Prompt")
profile = load_json(PROFILE_FILE, {})
profile.setdefault("agent_type", "Other")
profile.setdefault("persona", {"name": "", "tone": "Helpful", "sources": []})

profile["agent_type"] = st.sidebar.selectbox(
    "Agent Type", ["Parent","Teacher","Other"],
    index=["Parent","Teacher","Other"].index(profile["agent_type"]) if profile["agent_type"] in ["Parent","Teacher","Other"] else 2
)
profile["persona"]["name"] = st.sidebar.text_input(
    "Persona Name", profile["persona"]["name"]
)
profile["persona"]["tone"] = st.sidebar.selectbox(
    "Tone", ["Helpful","Encouraging","Neutral"],
    index=["Helpful","Encouraging","Neutral"].index(profile["persona"]["tone"]) if profile["persona"]["tone"] in ["Helpful","Encouraging","Neutral"] else 0
)
profile["persona"]["sources"] = st.sidebar.multiselect(
    "Pick Sources to Ground Persona", sources["domains"],
    default=profile["persona"]["sources"]
)
save_json(PROFILE_FILE, profile)

# --------------------
# 📑 3. RAG Indexing
# --------------------
st.sidebar.header("3. RAG & Indexing")
index = load_json(INDEX_FILE, [])
uploaded = st.sidebar.file_uploader(
    "Upload Reference Docs (PDF, TXT)", accept_multiple_files=True
)
if uploaded:
    with st.spinner("Indexing documents..."):
        total = len(uploaded)
        progress = st.sidebar.progress(0)
        for idx, f in enumerate(uploaded, start=1):
            try:
                # PDF handling
                if PdfReader and f.name.lower().endswith('.pdf'):
                    reader = PdfReader(f)
                    text = "\n\n".join(page.extract_text() or '' for page in reader.pages)
                else:
                    raw = f.read()
                    try:
                        text = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        text = raw.decode('latin-1', errors='ignore')
                chunks = [c for c in text.split("\n\n") if c.strip()]
                for chunk in chunks:
                    res = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
                    emb = res['data'][0]['embedding']
                    index.append({"text": chunk, "embedding": emb})
            except Exception as e:
                st.sidebar.error(f"Error processing {f.name}: {e}")
            progress.progress(idx / total)
            sleep(0.1)
        save_json(INDEX_FILE, index)
        st.sidebar.success(f"Indexed {len(index)} chunks.")

# RAG retrieval function
def retrieve_documents(query, top_k=3):
    with st.spinner("Retrieving relevant contexts..."):
        try:
            q_res = openai.Embedding.create(input=query, model="text-embedding-ada-002")
            q_emb = q_res['data'][0]['embedding']
            scores = [np.dot(q_emb, item['embedding']) for item in index]
            top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [index[i]['text'] for i in top_idxs]
        except Exception as e:
            st.error(f"RAG retrieval error: {e}")
            return []

# --------------------
# 🧠 4. Memory Management
# --------------------
st.sidebar.header("4. Memory")
memory = load_json(MEMORY_FILE, [])
mem_mode = st.sidebar.radio("Memory Mode", ["session","persistent"], index=0)
session_memory = [] if mem_mode == "session" else memory

# --------------------
# 🔧 5. Function-Calling Hooks
# --------------------
functions = [
    {
        "name": "retrieve_documents",
        "description": "Retrieve relevant document chunks for RAG",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"}
            },
            "required": ["query"]
        }
    }
]

# --------------------
# 💬 Chat Interface
# --------------------
st.title("🚀 Hybrid Context-Engineered Agent")
chat = load_json(CHAT_FILE, [])
for m in chat:
    if m['role'] == 'user':
        st.markdown(f"**You:** {m['content']}")
    else:
        st.markdown(f"**Agent:** {m['content']}")

query = st.text_input("Your question:")
if st.button("Send") and query:
    with st.spinner("Thinking..."):
        messages = []
        # Layer 1: system prompt
        sys = [
            f"Agent Type: {profile['agent_type']}",
            f"Persona Name: {profile['persona']['name']}",
            f"Tone: {profile['persona']['tone']}",
            "Sources: " + ", ".join(profile['persona']['sources'])
        ]
        messages.append({"role":"system","content":"\n".join(sys)})
        # Memory injection
        if mem_mode == "persistent":
            for entry in memory:
                messages.append(entry)
        # Add user turn
        messages.append({"role":"user","content": query})
        # ChatCompletion with function-calling
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                functions=functions,
                function_call="auto"
            )
        except Exception as e:
            st.error(f"API error: {e}")
            return
        msg = resp.choices[0].message
        # Handle function call
        if msg.get("function_call"):
            args = json.loads(msg["function_call"]["arguments"])
            docs = retrieve_documents(args.get("query", query), top_k=args.get("top_k",3))
            messages.append({"role":"function","name":"retrieve_documents","content": json.dumps(docs)})
            resp2 = openai.ChatCompletion.create(model="gpt-4o-mini",messages=messages)
            reply = resp2.choices[0].message.content
        else:
            reply = msg.content
        # Display and save history
        st.markdown(f"**Agent:** {reply}")
        chat.extend([
            {"role":"user","content": query},
            {"role":"assistant","content": reply}
        ])
        save_json(CHAT_FILE, chat)
        if mem_mode == "persistent":
            memory.extend(chat[-2:])
            save_json(MEMORY_FILE, memory)
