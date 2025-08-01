# README.md

# Hybrid Context-Engineered Agent

A Streamlit app combining mph-2025’s domain-agnostic source lists with Pairents’ deep customization: RAG indexing, memory persistence, and function-calling hooks. It provides a two-layer context-engineering pipeline, enabling true end-to-end control over AI chat interactions.

---

## Table of Contents
1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [File Structure](#file-structure)  
6. [Usage](#usage)  
   - [1. Manage Sources](#1-manage-sources)  
   - [2. Build Persona & System Prompt](#2-build-persona--system-prompt)  
   - [3. RAG & Indexing](#3-rag--indexing)  
   - [4. Memory Management](#4-memory-management)  
   - [5. Function-Calling Hooks](#5-function-calling-hooks)  
   - [6. Chat Interface](#6-chat-interface)  
7. [Extending the App](#extending-the-app)  
8. [Dependencies](#dependencies)  
9. [License](#license)  

---

## Features

- **Domain-Agnostic Sources**: Add/edit lists of books, experts, URLs or any domain-specific sources.  
- **Profile Builder**: Configure `agent_type`, persona name, tone, and select which sources ground the system prompt.  
- **Retrieval-Augmented Generation (RAG)**: Upload PDFs/TXT files, chunk into paragraphs, embed with OpenAI embeddings, and retrieve relevant text at query time.  
- **Memory Persistence**: Toggle between session-only memory or persistent memory saved across restarts.  
- **Function-Calling Hooks**: Define functions (e.g., `retrieve_documents`) that the model can invoke to fetch additional context dynamically.  
- **Two-Layer Context**:  
  - *Layer 1 (System Prompt)*: Curated sources & persona settings.  
  - *Layer 2 (Middleware)*: RAG retrieval, memory summaries, and function-call results.  
- **Chat History**: Displays full conversation history stored locally in JSON.  

---

## Architecture

1. **Streamlit Frontend**: Sidebar controls for each stage (Sources, Profile, RAG, Memory, Tools) and a main chat interface.  
2. **Persistence**: JSON files (`sources.json`, `profiles.json`, `index.json`, `memory.json`, `chat_history.json`) for stateful data.  
3. **OpenAI Integration**:  
   - **Embeddings** for RAG.  
   - **ChatCompletion** with `functions` for dynamic retrieval.  
4. **Middleware Logic**: Orchestrates system messages, retrieval calls, and memory injection before sending to the model.  

---

## Installation

```bash
git clone <repo-url>
cd hybrid-context-agent
pip install -r requirements.txt
streamlit run hybrid_app.py
