🌿 My Parent Helpers (MPH)

Test drive the app → https://mph-2025.streamlit.app/

My Parent Helpers (MPH) is a mobile-friendly, AI-powered application that enables parents to create personalized digital parenting assistants — called Agents — based on parenting books, experts, or styles. These agents provide contextual, age-appropriate guidance tailored to your child’s developmental needs and your preferred parenting approach.


---

🚀 Features

🧠 Contextual Prompt Engineering

MPH uses a multi-layered context injection technique to shape how the AI responds. Every query sent to the model dynamically injects:

✅ Persona Description – Synthesized from your chosen parenting source

✅ Parent and Child Names – For personalization

✅ Child’s Age – For developmentally appropriate language

✅ Structured Prompt Type – Defines format, tone, and purpose


This ensures every response is context-aware, customized, and aligned with your parenting goals.


---

👨‍👩‍👧‍👦 Agent Creation & Persona Generation

Create up to 99 unique parenting agents, each with:

A selected parenting source:

📚 Books

🧑‍ Experts

✨ Styles


A dynamically generated persona using GPT-4o

Custom values: Parent name, child name, age, profile name

Editable persona summaries before saving



---

💬 AI Chat With Purposeful Shortcuts

Each query can be refined using a response Shortcut that adjusts:

Format

Instruction to the model

Comprehension level


Shortcut	Purpose	Format	Language Level

💬 Default	Generic AI chat	Natural	Adaptive
🤝 Connect	Explain complex ideas	3 Examples	Child age
🌱 Grow	Parenting strategies	3 Advanced Tips	Adult
🔍 Explore	Interactive Q&A	Guided Answer	Child age
🛠 Resolve	Conflict resolution	Step-by-step	Child age
❤ Support	Emotional guidance	Supportive Advice	Child age



---

🔄 History & Management

Save and manage AI responses per profile

View and edit saved chats

Edit or delete agent profiles

Full export and persistence using local .json files



---

🔐 Requirements

Python 3.9+

OpenAI API key



---

📦 Installation

git clone https://github.com/yourusername/mph2025-app.git
cd mph2025-app
pip install -r requirements.txt
streamlit run app.py


---

🧬 Streamlit Secrets Setup

🔹 On Streamlit Cloud

1. Open your Streamlit Dashboard.


2. Go to your app → Secrets tab.


3. Add:



openai_key = "sk-xxxxxxxxxxxxxxxxxxxxx"

4. Save & rerun.



🔹 Locally

Create .streamlit/secrets.toml:

[general]
openai_key = "sk-xxxxxxxxxxxxxxxxxxxxx"

Or update directly in code:

openai.api_key = st.secrets.get("openai_key", "YOUR_OPENAI_API_KEY")


---

📂 Key Files

File	Purpose

app.py	Main application logic
parent_helpers_profiles.json	Saved agent profiles
parent_helpers_responses.json	Saved chat history
requirements.txt	Python dependencies



---

📖 Documentation

In-app User Manual and Quick Start Guide

Full walk-through for first-time users

Dynamic tooltips and responsive design

Editable persona fields and saved state management



---

✍️ Custom Sources & Editable Agents

MPH gives parents full control over the knowledge base powering each agent.

🔧 Source Management

You can edit the parenting Books, Experts, and Styles used to generate personas:

Add custom entries to the agent source lists

Remove or update existing parenting sources

Saved sources persist across sessions in a local JSON file

Changes are reflected immediately in new agent creation


🧑‍🎨 Editable Agent Personas

Once a persona is generated:

Review and approve the AI-generated summary

Edit the persona’s tone, principles, or values to match your parenting philosophy

Combine your custom source entries with a tailored persona for fully personalized parenting agents


This flexibility allows you to:

Reproduce advice aligned with your favorite parenting authors or mentors

Build unique agent voices and tones for different children

Continuously evolve your parenting strategies as your child grows


MPH isn't just AI-assisted — it's AI-shaped by you.


---

🛡️ License

MIT License


---

🤝 Contributing

We welcome contributions! Fork this repo and submit a PR for improvements or bug fixes.


---

📫 Contact

Questions or feedback? Visit www.myparenthelpers.com for support, resources, and contact options.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

My Parent Helpers 2025 (mph 2025) – Version README
Overview
My Parent Helpers (2025) is a Streamlit application that generates customized parenting agents using context engineering and ChatGPT 4.5. It is designed for parents and educators who want AI powered advice tailored to their parenting style, favorite books, or trusted experts. The app uses a smartphone style UI with tabs for creating agents, chatting, editing sources, reviewing saved items and clearing data. It emphasizes transparency—users can inspect or modify the sources (books, experts and styles) that shape the agent’s responses—so that parenting advice is always grounded in user approved material.
Key features
1. Create new agents
•	Agent types – When creating a new agent, users choose whether the persona is a Parent, Teacher or Other (free form) profile. Each type influences available source lists. For example, the “Parent” type offers parenting books and experts.
•	Source selection – After selecting the agent type, users pick a Book, Expert or Parenting Style. Default options are provided (e.g. books The Whole Brain Child and Peaceful Parent, Happy Kids ; experts Dr. Laura Markham and Dr. Daniel Siegel ; styles Authoritative and Gentle Parenting ). An “Other…” option allows users to supply their own book, expert or style names .
•	Customization – Users can create multiple agents and give each a name. Agents are stored in the Saved Agents list and may be selected later when starting new chats .
2. Chat with agents
•	Response modes – When starting a new chat, the user selects a previously created agent profile and chooses a response mode via icons:
o	Default: plain text answers.
o	Connect (🤝): offers actionable strategies and empathy to improve parenting .
o	Grow (📈): age appropriate Q&A with concise guidance .
o	Explore (🔍): step by step advice and deeper discussion .
Other icons represent analogies or creative formats. The selected mode is indicated under the Selected label.
•	Interactive chat – Users type questions and receive responses generated by the agent. Chats can be saved for later review; saved chats show the agent’s name, question and detailed answer with numbered tips .
3. Manage sources
•	Edit source lists – The Edit Sources button leads to a dedicated screen where users can view or modify the lists of books, experts and styles that define agent personas. Each list is displayed as a JSON like array and can be edited to add or remove entries  .
•	Dynamic filtering – Changing the agent type automatically updates the available sources (e.g. switching from Parent to Teacher displays teacher specific books and experts). This ensures that each agent draws from relevant reference material.
4. Data and saved items
•	Saved Agents & Chats – The app maintains lists of saved agent profiles and previous chat sessions. Agents show the profile name and can be selected to start new chats; chats show a preview of the question and answer and can be deleted .
•	Data counts and clearing – Under the Data section, a drop down labelled Counts reveals how many profiles and chats exist (e.g. “Profiles: 6” and “Chats: 2” ). The Clear Data button resets both lists and displays “All data cleared” .
5. Help & About
•	About – A small “About” panel notes that My Parent Helpers is “powered by context engineering messages dynamically ChatGPT 4.5” and markets itself as providing “Personalized helpers for parents” .
•	Help – The help panel gives usage instructions: users should edit agent source types and names, use those sources to build agent personas, create custom agents, and then chat. This clarifies the workflow .
How to use the app
1.	Add or edit sources: Use the Edit Sources button to tailor the lists of books, experts and parenting styles. Delete any items you don’t want and add new ones relevant to your parenting approach .
2.	Create a new agent: Click New Agent, choose the agent type (Parent/Teacher/Other) and select one or more sources (book, expert or style). Provide an agent name when prompted and save it to the Saved Agents list .
3.	Start a chat: Click New Chat, choose the agent you want to talk to and select a response mode using the icons. Enter your parenting question; the app will generate a tailored response with the tone/format defined by the mode. Save chats you want to revisit .
4.	Review and manage data: Expand the Counts dropdown under Data to see how many profiles and chats you have. If you want to start over, click Clear Data to remove all saved items .
5.	Learn more: Read the About section to understand that the app is powered by context engineered ChatGPT and aims to provide personalized assistance. Use the Help panel as a quick reminder of the workflow  .
Notes & limitations
•	No account required – All data is stored locally in the web session. Clearing browser data will remove profiles and chats.
•	Manual source curation – The quality of responses depends on the books, experts and styles you select. Ensure the source lists reflect your values and parenting philosophy.
•	Non medical advice – The app is for educational purposes. For serious behavioral or medical concerns, consult qualified professionals.
