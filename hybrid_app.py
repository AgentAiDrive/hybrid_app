import streamlit as st
import openai, json, os
from pydantic import BaseModel

# ---------------------------------------------------------------------------
#  📐  GLOBAL STYLE SHEET
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    body{background:linear-gradient(135deg,#2fe273 0%,#09742a 100%)!important;min-height:100vh;}
    .stApp {
      background: linear-gradient(335deg,#2fe273 0%,#09742a 100%)!important;
      border-radius: 32px;
      max-width: 400px;
      min-height: 100vh;
      height: 100vh;
      overflow-y: auto;
      margin: 32px auto;
      box-shadow: 0 8px 32px rgba(60,60,60,.25), 0 1.5px 8px rgba(30,90,40,.06);
      border: 3px solid #ffffff;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 10px 10px 10px;
    }

    .biglabel {
      font-size: 1.1em;
      font-weight: 800;
      color: #ffffff;
      margin: 4px 0 10px;
      text-align: center;
      letter-spacing: 0.5px;
      background: rgba(255, 255, 255, 0.15);  /* Light translucent white */
      padding: 6px 12px;
      border-radius: 12px;
    }
    .frame-avatar{font-size:1.4em;margin:6px 0 6px;display:flex;justify-content:center;color:#ffffff;}
    
    .stButton>button{
      border-radius:26px!important;
      font-weight:700!important;
      font-size:.7em!important;
      padding:.4em 0!important;
      background:#ffffff!important;
      color:#000000!important;
      margin:6px 0!important;
      width:100%!important;
    }
    .top-nav-container {
      padding: 12px 12px 12px 12px !important;
      border-radius: 32px !important;
      margin: -10px -10px 24px -10px !important;
      width: calc(100% + 20px) !important;
    }
    /* --- Top nav button colors: HIGH SPECIFICITY! --- */
    .top-nav-container > div[data-testid="stHorizontalBlock"] > div > div[data-testid="stButton"][data-key="nav_home"] > button { background: #e63946 !important; }
    .top-nav-container > div[data-testid="stHorizontalBlock"] > div > div[data-testid="stButton"][data-key="nav_chat"] > button { background: #27e67a !important; }
    .top-nav-container > div[data-testid="stHorizontalBlock"] > div > div[data-testid="stButton"][data-key="nav_saved"] > button { background: #1d3557 !important; }
   
    /* --- Answer bubble --- */
    .answer-box{background:#23683c;border-radius:12px;padding:14px 18px;color:#fff;white-space:pre-wrap;margin-top:8px;}
   
    /* --- Home cards --- */
    .home-card{background:rgba(255,255,255,0.15);border-radius:16px;padding:12px;margin:6px;color:#fff;}
    .home-card-title{font-weight:800;margin-bottom:6px;}
    .home-small{font-size:0.8em;opacity:0.85;background:white;border:3px solid #000000;padding:4px;;margin:6px}
    .home-button{font-size:0.8em;opacity:0.85;background:blue;border:3px solid #000000;}
    @media (max-height:750px){.stApp{min-height:640px;}}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
#  TOP NAVIGATION
# ---------------------------------------------------------------------------
def render_top_nav():
    st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.step = 0
            st.rerun()
    with col2:
        if st.button("💬 Chat", key="nav_chat"):
            st.session_state.step = 7 if st.session_state.profiles else 1
            st.rerun()
    with col3:
        if st.button("📂 Saved", key="nav_saved"):
            if st.session_state.saved_responses:
                st.session_state.step = 8
            else:
                st.warning("No saved responses yet.")
            st.rerun()

# ---------------------------------------------------------------------------
#  HELPER FUNCTIONS & CONSTANTS
# ---------------------------------------------------------------------------
PROFILES_FILE = "parent_helpers_profiles.json"
RESPONSES_FILE = "parent_helpers_responses.json"
SOURCES_FILE = "parent_helpers_sources.json"

def load_json(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return []

def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error writing {path}: {e}")

for key, default in {
    "profiles":        load_json(PROFILES_FILE),
    "saved_responses": load_json(RESPONSES_FILE),
    "last_answer":     "",
}.items():
    st.session_state.setdefault(key, default)

step = st.session_state.get("step", 0)
openai.api_key = st.secrets.get("openai_key", "YOUR_OPENAI_API_KEY")
# Agent types and default sources
AGENT_TYPES = ["Parent", "Teacher", "Other"]

PARENT_SOURCES = {
    "Book": ["The Whole-Brain Child", "Peaceful Parent, Happy Kids"],
    "Expert": ["Dr. Laura Markham", "Dr. Daniel Siegel"],
    "Style": ["Authoritative", "Gentle Parenting"]
}

TEACHER_SOURCES = {
    "Book": ["Teach Like a Champion", "Mindset"],
    "Expert": ["Carol Dweck", "Doug Lemov"],
    "Style": ["Project-Based Learning", "SEL"]
}

OTHER_SOURCES = {
    "Book": ["Custom Book (enter manually)"],
    "Expert": ["Custom Expert (enter manually)"],
    "Style": ["Custom Style (enter manually)"]
}

def get_source_options(agent_type):
    return st.session_state.get("sources", {}).get(agent_type, {})
    
if 'sources' not in st.session_state:
    st.session_state['sources'] = {
        "Parent": {
            "Book": ["The Whole-Brain Child", "Peaceful Parent, Happy Kids"],
            "Expert": ["Dr. Laura Markham", "Dr. Daniel Siegel"],
            "Style": ["Authoritative", "Gentle Parenting"]
        },
        "Teacher": {
            "Book": ["Teach Like a Champion", "Mindset"],
            "Expert": ["Carol Dweck", "Doug Lemov"],
            "Style": ["Project-Based Learning", "SEL"]
        },
        "Other": {
            "Book": ["Custom Book (enter manually)"],
            "Expert": ["Custom Expert (enter manually)"],
            "Style": ["Custom Style (enter manually)"]
        }
    }

class PersonaProfile(BaseModel):
    profile_name: str
    parent_name: str
    child_name: str
    child_age: int
    agent_type: str
    source_type: str
    source_name: str
    persona_description: str

SHORTCUTS = ["💬 DEFAULT","🤝 CONNECT","🌱 GROW","🔍 EXPLORE","🛠 RESOLVE","❤ SUPPORT"]
EMOJIS = {"💬 DEFAULT":"💬","🤝 CONNECT":"🤝","🌱 GROW":"🌱","🔍 EXPLORE":"🔍","🛠 RESOLVE":"🛠","❤ SUPPORT":"❤"}
TOOLTIPS = {
    "💬 DEFAULT":"No formatting",
    "🤝 CONNECT":"Help explain complex ideas with examples",
    "🌱 GROW":"Strategies to improve parenting",
    "🔍 EXPLORE":"Age-appropriate Q&A",
    "🛠 RESOLVE":"Step-by-step advice",
    "❤ SUPPORT":"Empathetic guidance"
}
# ---------------------------------------------------------------------------
#  STEP LOGIC
# ---------------------------------------------------------------------------
def render_home_card(title, subtitle=None, buttons=None, expander_label=None, expander_body=None):
    # Title
    st.markdown(f'<div class="biglabel">{title}</div>', unsafe_allow_html=True)

    # Optional subtitle
    if subtitle:
        st.markdown(subtitle, unsafe_allow_html=True)

    # Buttons
    if buttons:
        for label, key, condition, action in buttons:
            if st.button(label, key=key):
                if condition is None or condition():
                    action()

    # Expander
    if expander_label and expander_body:
        with st.expander(expander_label):
            expander_body()

if step == 0:
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # --- Card: AGENTS ---
    with row1_col1:
        render_home_card(
            "AGENTS",
            subtitle='<p class="home-small"> -> Manage Agents  </p>',
            buttons=[
                ("SAVED AGENTS", "home_profiles", lambda: st.session_state.profiles,
                 lambda: (st.session_state.__setitem__('step', 9), st.rerun())),
                ("NEW AGENT", "home_create", None,
                 lambda: (st.session_state.__setitem__('step', 1), st.rerun()))
            ],
            expander_label="Profiles",
            expander_body=lambda: [
                st.markdown(f"<p class='home-small'>{p['profile_name']}</p>", unsafe_allow_html=True)
                for p in st.session_state.profiles
            ] if st.session_state.profiles else st.markdown(
                '<p class="home-small"> No profiles yet.</p>', unsafe_allow_html=True
            )
        )
    # --- Card: CHATS ---
    with row1_col2:
        render_home_card(
            "CHATS",
            subtitle='<p class="home-small">  View and Delete Chats </p>',
            buttons=[
                ("SAVED CHATS", "home_saved", lambda: st.session_state.saved_responses,
                 lambda: (st.session_state.__setitem__('step', 8), st.rerun())),
                ("NEW CHAT", "home_chat", None, lambda: (
                    st.session_state.__setitem__('step', 7 if st.session_state.profiles else 1),
                    st.warning('No profiles – create one first.') if not st.session_state.profiles else None,
                    st.rerun()
                ))
            ],
            expander_label="Saved Count",
            expander_body=lambda: st.markdown(
                f"<p class='home-small'>{len(st.session_state.saved_responses)} saved</p>",
                unsafe_allow_html=True
            )
        )

    # --- Card: SOURCES ---
    with row2_col1:
        render_home_card(
            "SOURCES",
            buttons=[
                ("EDIT SOURCES", "edit_sources", None,
                 lambda: (st.session_state.__setitem__('step', 10), st.rerun()))
            ],
            expander_label="Counts",
            expander_body=lambda: [
                st.markdown(
                    f"<p class='home-small'>{atype}: {sum(len(st.session_state['sources'].get(atype, {}).get(t, [])) for t in ['Book','Expert','Style'])}</p>",
                    unsafe_allow_html=True
                ) for atype in AGENT_TYPES
            ]
        )

elif step == 1:
    render_top_nav()
    st.markdown(
            """
            <div style="text-align:center;">
              <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
            </div>
            """,
            unsafe_allow_html=True,)
    st.markdown('<div class="biglabel">Select An Agent Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="frame-avatar"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👪  Parent", key="btn_agent_parent"):
            st.session_state.agent_type = "Parent"
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🧑‍🏫  Teacher", key="btn_agent_teacher"):
            st.session_state.agent_type = "Teacher"
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("✨  Other", key="btn_agent_other"):
            st.session_state.agent_type = "Other"
            st.session_state.step = 2
            st.rerun()

elif step == 2:
    render_top_nav()
    st.markdown(
                """
                <div style="text-align:center;">
                  <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
                </div>
                """,
                unsafe_allow_html=True,)
    st.markdown('<div class="biglabel">Select Agent Source Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="frame-avatar"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📚  Book", key="btn_book"):
            st.session_state.source_type = "Book"
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("🧑‍  Expert", key="btn_expert"):
            st.session_state.source_type = "Expert"
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("🌟  Style", key="btn_style"):
            st.session_state.source_type = "Style"
            st.session_state.step = 3
            st.rerun()

elif step == 3:
    st.markdown(
        """
        <div style="text-align:center;">
          <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Get current agent type
    agent_type = st.session_state.get("agent_type", "Parent")
    # Get source options for agent type
    sources = get_source_options(agent_type)
    source_type = st.session_state.get("source_type", "Book")
    # Available options for the selected source_type
    options = sources.get(source_type, ["Other..."])
    # Always append "Other..." for custom entry
    if "Other..." not in options:
        options = options + ["Other..."]
    emoji = "📚" if source_type == "Book" else "🧑‍" if source_type == "Expert" else "🌟"
    st.markdown(f'<div class="biglabel">Choose a {source_type}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="frame-avatar">{emoji}</div>', unsafe_allow_html=True)
    choice = st.selectbox("Select or enter your own:", options)
    custom = st.text_input("Enter custom name") if choice == "Other..." else ""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("BACK", key="btn_back_step2"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("CREATE", key="btn_create_step2"):
            src_name = custom if choice == "Other..." else choice
            if not src_name:
                st.warning("Please provide a name.")
            else:
                st.session_state.source_name = src_name
                # clear any old persona so step 4 always regenerates
                st.session_state.pop("persona_description", None)
                st.session_state.step = 4
                st.rerun()
                
elif step == 4:
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    import time
    st.markdown('<div class="biglabel">GENERATING YOUR AGENT PERSONA</div>', unsafe_allow_html=True)
    st.markdown('<div class="frame-avatar">🧠✨</div>', unsafe_allow_html=True)
    placeholder = st.empty()
    for msg in ["Assimilating Knowledge…", "Synthesizing Information…", "Assessing Results…", "Generating Persona…"]:
        placeholder.info(msg)
        time.sleep(0.5)
    if "persona_description" not in st.session_state:
        with st.spinner("Thinking…"):
            try:
                prompt = (
                    f"Summarize the philosophy, core principles, and practices of "
                    f"the {st.session_state.source_type} '{st.session_state.source_name}' in under 200 words to describe the persona. "
                    "Respond in a JSON object with 'persona_description'."
                )
                out = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type":"json_object"}
                )
                raw = out.choices[0].message.content
                st.session_state.persona_description = json.loads(raw)["persona_description"]
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
    placeholder.empty()
    desc = st.session_state.get("persona_description")
    if desc:
        st.info(desc)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("RETRY", key="btn_retry"):
            st.session_state.pop("persona_description", None)
            st.rerun()
    with col2:
        if st.button("SAVE", key="btn_save_persona"):
            st.session_state.step = 5
            st.rerun()

elif step == 5:
    st.markdown(
                """
                <div style="text-align:center;">
                  <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
                </div>
                """,
                unsafe_allow_html=True,)
    st.markdown('<div class="biglabel">PERSONALIZE AGENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="frame-avatar">📷</div>', unsafe_allow_html=True)
    with st.form("profile"):
        p_name = st.text_input("Parent first name")
        c_age  = st.number_input("Child age", 1, 21)
        c_name = st.text_input("Child first name")
        prof_nm= st.text_input("Profile name")
        saved  = st.form_submit_button("SAVE")
    if saved:
        if not all([p_name, c_age, c_name, prof_nm]):
            st.warning("Please fill every field.")
        else:
            profile = PersonaProfile(
                profile_name=prof_nm,
                parent_name=p_name,
                child_name=c_name,
                child_age=int(c_age),
                source_type=st.session_state.source_type,
                source_name=st.session_state.source_name,
                persona_description=st.session_state.persona_description,
                agent_type=st.session_state.agent_type
            )
            st.session_state.profiles.append(profile.dict())
            save_json(PROFILES_FILE, st.session_state.profiles)
            st.success("Profile saved!")
            st.session_state.step = 6
            st.rerun()
    if st.button("BACK", key="btn_back_details"):
        st.session_state.step = 4
        st.rerun()

elif step == 6:
    # Top nav + header
    render_top_nav()
    st.markdown(
        """
        <div style="text-align:center;">
          <img src="https://img1.wsimg.com/isteam/ip/e13cd0a5-b867-446e-af2a-268488bd6f38/myparenthelpers%20logo%20round.png" width="80" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="biglabel">AGENT PROFILE CREATED!</div>', unsafe_allow_html=True)
    st.markdown('<div class="frame-avatar"></div>', unsafe_allow_html=True)

    # Pull the latest profile
    profile = st.session_state.profiles[-1]

    # Render it in a colored “card”
    st.markdown(
        f"""
        <div class="home-card">
          <p><strong>Profile Name:</strong> {profile['profile_name']}</p>
          <p><strong>Parent:</strong> {profile['parent_name']}</p>
          <p><strong>Child:</strong> {profile['child_name']} (Age {profile['child_age']})</p>
          <p><strong>Agent Type:</strong> {profile['agent_type']}</p>
          <p><strong>Source ({profile['source_type']}):</strong> {profile['source_name']}</p>
          <p><strong>Persona Description:</strong></p>
          <div class="answer-box" style="margin-top:4px;">{profile['persona_description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif step == 7:
    render_top_nav() 
    # -- Row: Section Title and Agent Selector in One Line --
    col1, col2 = st.columns([3, 5])

    # Column 1: Header
    with col1:
        st.markdown('<div class="biglabel">1. SELECT AN AGENT</div>', unsafe_allow_html=True)

    # Column 2: Selectbox and Info Icon
    names = [p["profile_name"] for p in st.session_state.profiles]
    col_dd, col_icon = col2.columns([4, 1])

    # Selectbox
    idx = col_dd.selectbox(
        "Agent Profiles:",
        range(len(names)),
        format_func=lambda i: names[i],
        key="chat_profile"
    )

    # Info Tooltip
    sel = st.session_state.profiles[idx]
    tooltip = (
        f"Profile: {sel['profile_name']} | "
        f"Agent: {sel.get('agent_type','Parent')} | "
        f"Type: {sel['source_type']} | "
        f"Source: {sel['source_name']} | "
        f"Child: {sel['child_name']} | "
        f"Age: {sel['child_age']} | "
        f"Parent: {sel['parent_name']} | "
        f"Persona: {sel['persona_description']}"
    )

    col_icon.markdown(
        f'<span title="{tooltip}" style="font-size:1.5em; cursor:help;">ℹ️</span>',
        unsafe_allow_html=True,
    )


    # -- Ensure session state key exists first --
    st.session_state.setdefault("shortcut", "💬 DEFAULT")

    # -- Row 1: Section Label and Selection Box --
    col1, col2 = st.columns([3, 5])

    with col1:
        st.markdown('<div class="biglabel">2. SELECT A RESPONSE TYPE</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
            <div style="background:#fff;color:#000;padding:12px;border-radius:8px;margin-top:4px;margin-bottom:12px;">
              <strong>Selected:</strong> {st.session_state.shortcut}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- Row 2: Emoji Buttons (Full Width) --
    button_cols = st.columns(len(SHORTCUTS))
    for i, sc in enumerate(SHORTCUTS):
        with button_cols[i]:
            if st.button(EMOJIS[sc], key=f"type_{sc}", help=TOOLTIPS[sc]):
                st.session_state.shortcut = sc

    st.markdown('<div class="home-small">3. WHAT DO YOU WANT TO ASK?</div>', unsafe_allow_html=True)
    query = st.text_area("Type here", key="chat_query")
    if st.session_state.last_answer:
        st.markdown(f"<div class='answer-box'>{st.session_state.last_answer}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("SAVE RESPONSE", key="save_response"):
            record = {
                "profile": sel["profile_name"],
                "shortcut": st.session_state.shortcut,
                "question": query,
                "answer":   st.session_state.last_answer
            }
            if record not in st.session_state.saved_responses:
                st.session_state.saved_responses.append(record)
                save_json(RESPONSES_FILE, st.session_state.saved_responses)
            st.session_state.step = 8
            st.rerun()
    with col2:
        if st.button("SEND", key="send_btn"):
            base = (
              f"Adopt the persona described here: {sel['persona_description']}. You are conversing with:"
              f" Parent: {sel['parent_name']}, Child: {sel['child_name']}, Age: {sel['child_age']}."
            )
            extra_map = {
              "🤝 CONNECT":" Help explain with examples.",
              "🌱 GROW":" Offer advanced strategies.",
              "🔍 EXPLORE":" Facilitate age-appropriate Q&A.",
              "🛠 RESOLVE":" Provide step-by-step resolution.",
              "❤ SUPPORT":" Offer empathetic support."
            }
            prompt = base + extra_map.get(st.session_state.shortcut, "") + "\n" + query + "\nRespond as JSON with 'answer'."
            try:
                out = openai.chat.completions.create(
                  model="gpt-4o",
                  messages=[{"role":"system","content":prompt}],
                  response_format={"type":"json_object"}
                )
                st.session_state.last_answer = json.loads(out.choices[0].message.content)["answer"]
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
            st.rerun()

elif step == 8:
    render_top_nav()
    st.markdown('<div class="biglabel">SELECT A SAVED CHAT</div>', unsafe_allow_html=True)
    if not st.session_state.saved_responses:
        st.info("No saved responses."); st.session_state.step = 0; st.rerun()
    titles = [f"{i+1}. {r['profile']} – {r['shortcut']}" for i, r in enumerate(st.session_state.saved_responses)]
    sel_idx = st.selectbox("Saved Chats:", range(len(titles)), format_func=lambda i: titles[i], key="saved_select")
    item = st.session_state.saved_responses[sel_idx]
    for field in ("profile","shortcut"):
        st.markdown(f'''
          <p style="color:#fff;margin:4px 0;">
            <strong>{field.title()}:</strong> {item[field]}
          </p>''', unsafe_allow_html=True)
    st.markdown('''
      <p style="color:#fff;margin:4px 0;"><strong>Question:</strong></p>''',
      unsafe_allow_html=True)
    st.markdown(f'''
      <blockquote style="color:#fff;border-left:4px solid #27e67a;
                        padding-left:8px;margin:4px 0;">
        {item["question"]}
      </blockquote>''', unsafe_allow_html=True)
    st.markdown('''
      <p style="color:#fff;margin:4px 0;"><strong>Answer:</strong></p>''',
      unsafe_allow_html=True)
    st.markdown(f'''
      <div class="answer-box" style="color:#fff;">
        {item["answer"]}
      </div>''', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("DELETE", key="btn_delete_saved"):
            st.session_state.saved_responses.pop(sel_idx)
            save_json(RESPONSES_FILE, st.session_state.saved_responses)
            st.rerun()
    with c2:
        if st.button("CLOSE", key="btn_close_saved"):
            st.session_state.step = 0
            st.rerun()

elif step == 9:
    render_top_nav()
    st.markdown('<div class="biglabel">AGENT PROFILES</div>', unsafe_allow_html=True)
    if not st.session_state.profiles:
        st.info("No profiles stored."); st.session_state.step = 0; st.rerun()
    titles = [f"{i+1}. {p['profile_name']}" for i,p in enumerate(st.session_state.profiles)]
    idx = st.selectbox("Select a profile to view / edit", range(len(titles)), format_func=lambda i: titles[i], key="profile_select")
    prof = st.session_state.profiles[idx]
    with st.form("edit_profile"):
        p_name = st.text_input("Parent first name", value=prof.get("parent_name", ""))
        c_age  = st.number_input("Child age", 1, 21, value=prof.get("child_age", 1))
        c_name = st.text_input("Child first name", value=prof.get("child_name", ""))
        prof_nm= st.text_input("Profile name", value=prof.get("profile_name", ""))
        a_type = st.selectbox("Agent type", ["Parent","Teacher","Other"], index=["Parent","Teacher","Other"].index(prof.get("agent_type","Parent")))
        desc   = st.text_area("Persona description", value=prof.get("persona_description",""), height=150)
        saved  = st.form_submit_button("SAVE CHANGES")
    if saved:
        prof.update(parent_name=p_name, child_age=int(c_age), child_name=c_name, profile_name=prof_nm, persona_description=desc, agent_type=a_type)
        st.session_state.profiles[idx] = prof
        save_json(PROFILES_FILE, st.session_state.profiles)
        st.success("Profile updated!")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("DELETE PROFILE", key="btn_delete_profile"):
            st.session_state.profiles.pop(idx)
            save_json(PROFILES_FILE, st.session_state.profiles)
            st.rerun()
    with c2:
        if st.button("CLOSE", key="btn_close_profile"):
            st.session_state.step = 0
            st.rerun()

elif step == 10:
    render_top_nav()
    st.markdown('<div class="biglabel">EDIT SOURCE LISTS</div>', unsafe_allow_html=True)

    # Source persistence file
    SOURCES_FILE = "parent_helpers_sources.json"

    # Define save_sources function
    def save_sources(sources):
        try:
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2)
        except Exception as e:
            st.error(f"Error saving sources: {e}")

    # Load sources from file if not already
    def load_sources():
        return load_json(SOURCES_FILE)

    # Initialize sources if needed
    if 'sources' not in st.session_state:
        st.session_state['sources'] = load_sources() or {
            "Parent": PARENT_SOURCES,
            "Teacher": TEACHER_SOURCES,
            "Other": OTHER_SOURCES
        }

    sources = st.session_state["sources"]
    agent_type = st.selectbox("Agent Type", AGENT_TYPES, key="edit_agent_type")
    source_type = st.selectbox("Source Type", ["Book", "Expert", "Style"], key="edit_source_type")

    items = sources.get(agent_type, {}).get(source_type, [])
    st.write(f"**Current {source_type}s for {agent_type}:**")
    to_remove = st.multiselect("Select to remove", items, key="remove_sources")
    new_item = st.text_input(f"Add new {source_type}:", key="add_source")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Remove Selected"):
            sources[agent_type][source_type] = [x for x in items if x not in to_remove]
            st.session_state["sources"] = sources
            save_sources(sources)
            st.success("Removed selected!")
            st.rerun()

    with c2:
        if st.button("Add"):
            if new_item and new_item not in items:
                sources[agent_type][source_type].append(new_item)
                st.session_state["sources"] = sources
                save_sources(sources)
                st.success(f"Added '{new_item}'!")
                st.rerun()
            elif new_item:
                st.warning("Already in list.")

    with c3:
        if st.button("Back to Home"):
            st.session_state.step = 0
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Current list:**")
    st.write(sources.get(agent_type, {}).get(source_type, []))
