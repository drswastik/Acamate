import os
import tempfile
import streamlit as st
from datetime import datetime

# Import custom backend modules
from db.db_manager import DatabaseManager
from core.pdf_handler import PDFHandler
from core.response_generator import ResponseGenerator

#page configuration and custom CSS for chat interface
st.set_page_config(page_title="AcaMate Research Engine", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .stChatMessage { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

#initialise the database and backend components
if "backend_loaded" not in st.session_state:
    st.session_state.db = DatabaseManager()
    st.session_state.db.create_tables()
    
    # Create a dummy paper in the DB to unify the session architecture
    if not st.session_state.db.get_paper_id("GENERAL_KNOWLEDGE"):
        st.session_state.db.insert_paper("🌍 General AI Chat (No PDF)", "GENERAL_KNOWLEDGE", "N/A")
    
    st.session_state.pdf_handler = PDFHandler(st.session_state.db)
    st.session_state.response_gen = ResponseGenerator(st.session_state.db)
    st.session_state.backend_loaded = True
    
    st.session_state.paper_data = None
    st.session_state.active_session_id = None
    st.session_state.active_session_name = None
    st.session_state.chat_history = []
    st.session_state.last_query_hash = None
    st.session_state.last_uploaded_file_id = None

db = st.session_state.db
pdf_handler = st.session_state.pdf_handler
response_gen = st.session_state.response_gen

def load_session_history(session_id):
    raw_history = db.get_chat_history(session_id, limit=30)
    st.session_state.chat_history = []
    for msg in raw_history:
         st.session_state.chat_history.append({"role": msg["role"], "content": msg["content"], "trace": None})

#sidebar ui
with st.sidebar:
    st.title("🤖 AcaMate Library")
    st.markdown("---")

    st.subheader("📂 1. Select Mode / Paper")
    
    # Force a quick commit to ensure we are reading the freshest DB state
    db.conn.commit() 
    saved_papers = db.get_all_papers()
    
    paper_options = {"Select an option...": None}
    
    # Force the General Chat mode to the top of the dropdown
    general_id = db.get_paper_id("GENERAL_KNOWLEDGE")
    paper_options["🌍 General AI Chat (No PDF)"] = {"paper_id": general_id, "file_name": "General AI Chat", "file_hash": "GENERAL_KNOWLEDGE"}
    
    for p in saved_papers:
        if p[2] != "GENERAL_KNOWLEDGE":
            paper_options[f"[{p[0]}] {p[1]}"] = {"paper_id": p[0], "file_name": p[1], "file_hash": p[2]}

    options_list = list(paper_options.keys())

    # Safe UI State Mutation 
    # We intercept the forced selection BEFORE the selectbox is instantiated to prevent crashes
    if "force_paper_selection" in st.session_state:
        if st.session_state.force_paper_selection in options_list:
            st.session_state.paper_selectbox = st.session_state.force_paper_selection
        del st.session_state.force_paper_selection

    selected_paper_key = st.selectbox("Choose from Library:", options=options_list, key="paper_selectbox")

    uploaded_file = st.file_uploader("Or Upload New PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file_id != uploaded_file.file_id:
            with st.spinner("Ingesting PDF (FAISS & MiA-Map)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                paper_info = pdf_handler.process_pdf(tmp_path)
                if paper_info:
                    st.session_state.last_uploaded_file_id = uploaded_file.file_id
                    
                    new_paper_id = db.get_paper_id(paper_info['file_hash'])
                    new_key = f"[{new_paper_id}] {paper_info['file_name']}"
                    
                    # Store the command safely for the next rerun
                    st.session_state.force_paper_selection = new_key
                    
                    # Reset the chat session states
                    st.session_state.paper_data = {"paper_id": new_paper_id, "file_name": paper_info['file_name'], "file_hash": paper_info['file_hash']}
                    st.session_state.active_session_id = None
                    st.session_state.chat_history = []
                    
                    st.success(f"Successfully processed {paper_info['file_name']}!")
                    st.rerun()

    # Handle Paper Switching 
    current_paper = paper_options.get(selected_paper_key)
    if current_paper != st.session_state.paper_data:
        st.session_state.paper_data = current_paper
        st.session_state.active_session_id = None
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    if st.session_state.paper_data:
        st.subheader("💬 2. Chat Sessions")
        sessions = db.get_sessions_for_paper(st.session_state.paper_data["paper_id"])
        
        session_options = {"➕ Start New Session": "NEW"}
        for s in sessions:
            session_options[f"#{s[0]}: {s[1]}"] = s[0]

        selected_session_key = st.selectbox("Active Session:", options=list(session_options.keys()))

        if session_options[selected_session_key] == "NEW":
            new_sess_name = st.text_input("New Session Name", placeholder="e.g., Analysis")
            if st.button("Create Session"):
                name = new_sess_name if new_sess_name else f"Session {datetime.now().strftime('%b %d, %H:%M')}"
                new_id = db.create_chat_session(st.session_state.paper_data["paper_id"], name)
                st.session_state.active_session_id = new_id
                st.session_state.active_session_name = name
                st.session_state.chat_history = []
                st.rerun()
        else:
            session_id = session_options[selected_session_key]
            if st.session_state.active_session_id != session_id:
                st.session_state.active_session_id = session_id
                st.session_state.active_session_name = selected_session_key.split(": ")[1]
                load_session_history(session_id)

# main chat interface
if not st.session_state.paper_data:
    st.title("Welcome to AcaMate 🤖")
    st.info("👈 Please select General Chat or upload a research paper from the sidebar to begin.")
elif not st.session_state.active_session_id:
    st.title(f"{'🌍' if st.session_state.paper_data['file_hash'] == 'GENERAL_KNOWLEDGE' else '📄'} {st.session_state.paper_data['file_name']}")
    st.info("👈 Please create or select a chat session from the sidebar.")
else:
    st.title(f"{'🌍' if st.session_state.paper_data['file_hash'] == 'GENERAL_KNOWLEDGE' else '📄'} {st.session_state.paper_data['file_name']}")
    st.caption(f"💬 Active Session: {st.session_state.active_session_name}")
    st.markdown("---")
    
    # Render History
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("🔍 Backend Workflow & Evidence (Execution Trace)"):
                    st.json(msg["trace"])

    if user_query := st.chat_input("Ask a question..."):
        
        st.session_state.chat_history.append({"role": "user", "content": user_query, "trace": None})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing intent, extracting context, & generating response..."):
                
                result = response_gen.get_response(
                    paper_id=st.session_state.paper_data["paper_id"],
                    file_hash=st.session_state.paper_data["file_hash"],
                    session_id=st.session_state.active_session_id,
                    user_query=user_query,
                    paper_text=""  
                )
                
                st.session_state.last_query_hash = result["query_hash"]
                trace = result.get("trace", {})
                
                st.markdown(result["answer"])
                
                col1, col2 = st.columns(2)
                col1.caption(f"🔀 **Route:** {result['provider']}")
                col2.caption(f"🛡️ **Status:** {result['badge']}")

                if trace:
                    with st.expander("🔍 Backend Workflow & Evidence (Execution Trace)"):
                        st.markdown(f"**🧠 Semantic Intent:** `{trace.get('routing', {}).get('rag_intent', 'N/A')}`")
                        st.markdown(f"**🚦 RAG Gate Triggered:** `{trace.get('routing', {}).get('use_rag', False)}`")
                        st.markdown(f"**⚙️ Selected Engine:** `{trace.get('routing', {}).get('engine', 'N/A')}`")
                        st.markdown(f"**🎯 Refined Query:** `{trace.get('refined_query', 'N/A')}`")
                        
                        chunks = trace.get('pruned_chunks', [])
                        if chunks:
                            st.markdown("**📑 Citations / Snippets Retrieved:**")
                            for idx, chunk in enumerate(chunks):
                                st.info(f"**Snippet {idx+1}:** {chunk}")

                formatted_response = f"{result['answer']}\n\n---\n*🔀 Route: {result['provider']} | 🛡️ Status: {result['badge']}*"
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": formatted_response,
                    "trace": trace
                })
