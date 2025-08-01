# hybrid_app.py

import streamlit as st
import openai, json, os
import numpy as np
from pydantic import BaseModel

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
            return default
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
    new_src = st.text_input("Add a source (book/expert/style/URL)")
    if st.button("➕ Add Source") and new_src:
        sources["domains"].append(new_src)
        save_json(SOURCES_FILE, sources)
        st.experimental_rerun()
    st.markdown("---")
    for i, src in enumerate(sources["domains"]):
        cols = st.columns((0.85, 0.15))
        cols[0].write(src)
        if cols[1].button("✖", key=f"del_src_{i}"):
            sources["domains"].pop(i)
            save_json(SOURCES_FILE, sources)
            st.experimental_rerun()

# --------------------
# 🧱 2. Profile Builder
# --------------------
st.sidebar.header("2. Build Persona & System Prompt")
profile = load_json(PROFILE_FILE, {})
profile.setdefault("agent_type", "Other")
profile.setdefault("persona", {"name": "", "tone": "Helpful", "sources": []})

profile["agent_type"] = st.sidebar.selectbox(
    "Agent Type", ["Parent","Teacher","Other"],
    index=["Parent","Teacher","Other"].index(profile["agent_type"]) 
          if profile["agent_type"] in ["Parent","Teacher","Other"] else 2
)
profile["persona"]["name"] = st.sidebar.text_input(
    "Persona Name", profile["persona"]["name"]
)
profile["persona"]["tone"] = st.sidebar.selectbox(
    "Tone", ["Helpful","Encouraging","Neutral"],
    index=["Helpful","Encouraging","Neutral"].index(profile["persona"]["tone"])
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
    # Simple chunk and embed
    for f in uploaded:
        text = f.read().decode('utf-8')
        # naive split by paragraphs
        chunks = [c for c in text.split("\n\n") if c.strip()]
        for chunk in chunks:
            res = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
            emb = res['data'][0]['embedding']
            index.append({"text": chunk, "embedding": emb})
    save_json(INDEX_FILE, index)
    st.sidebar.success(f"Indexed {len(index)} chunks.")

# RAG retrieval function
def retrieve_documents(query, top_k=3):
    q_res = openai.Embedding.create(input=query, model="text-embedding-ada-002")
    q_emb = q_res['data'][0]['embedding']
    # cosine similarity
    scores = [np.dot(q_emb, item['embedding']) for item in index]
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [index[i]['text'] for i in top_idxs]

# --------------------
# 🧠 4. Memory Management
# --------------------
st.sidebar.header("4. Memory")
memory = load_json(MEMORY_FILE, [])
mem_mode = st.sidebar.radio("Memory Mode", ["session","persistent"], index=0)
if mem_mode == "session":
    session_memory = []
else:
    # persistent memory across restarts
    session_memory = memory

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
                "query": {"type": "string", "description": "Query to retrieve docs for"},
                "top_k": {"type": "integer", "description": "Number of docs"}
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
    # Build messages
    messages = []
    sys = [
        f"Agent Type: {profile['agent_type']}",
        f"Persona Name: {profile['persona']['name']}",
        f"Tone: {profile['persona']['tone']}",
        "Sources: " + ", ".join(profile['persona']['sources'])
    ]
    messages.append({"role":"system","content":"\n".join(sys)})
    # Memory layer
    if mem_mode == "persistent":
        for entry in memory:
            messages.append(entry)
    # RAG via function calling
    messages.append({"role":"user","content": query})
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    msg = completion.choices[0].message
    if msg.get("function_call"):
        args = json.loads(msg["function_call"]["arguments"])
        docs = retrieve_documents(args["query"], top_k=args.get("top_k",3))
        messages.append({
            "role":"function",
            "name":"retrieve_documents",
            "content": json.dumps(docs)
        })
        second = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = second.choices[0].message.content
    else:
        reply = msg.content
    # Display & save
    st.markdown(f"**Agent:** {reply}")
    chat.append({"role":"user","content": query})
    chat.append({"role":"assistant","content": reply})
    save_json(CHAT_FILE, chat)
    if mem_mode == "persistent":
        memory.extend([
            {"role":"user","content": query},
            {"role":"assistant","content": reply}
        ])
        save_json(MEMORY_FILE, memory)
