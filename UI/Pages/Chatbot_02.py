def app():
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    import streamlit as st
    import random

    from dotenv import load_dotenv
    load_dotenv()

    from Preprocessing.vector_store import CVVectorStore
    from RAG.rag_engine import EnhancedRAGEngine

    PRIMARY_COLOR = "#017691"
    ACCENT_COLOR = "#FF9F1C"
    BACKGROUND_COLOR = "#f5f7fa"

    # --- Load Vector Store with data ---
    @st.cache_resource
    def load_rag_engine(initial_docs):
        vector_store = CVVectorStore()
        if initial_docs:
            vector_store.add_cvs(initial_docs)
        rag_engine = EnhancedRAGEngine(vector_store)
        return rag_engine

    # --- Get processed CVs from session_state ---
    initial_docs = st.session_state.get("all_cv_chunks", [])
    rag_engine = load_rag_engine(initial_docs)

    # --- Global Styles ---
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {BACKGROUND_COLOR}; font-family: 'Poppins', sans-serif; }}
        .main-title {{ color: {PRIMARY_COLOR}; font-size: 38px; font-weight: bold; text-align: center; margin-top: 20px; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #444; font-size: 16px; margin-bottom: 20px; }}
        .chat-container {{ max-width: 800px; margin: 0 auto 10px auto; }}
        .message {{ display: flex; margin-bottom: 12px; align-items: flex-start; }}
        .user-msg {{ justify-content: flex-start; }}
        .bot-msg {{ justify-content: flex-end; }}
        .bubble {{ max-width: 70%; padding: 14px 18px; border-radius: 16px; font-size: 16px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }}
        .user-bubble {{ background-color: #DCF8C6; color: #000; border-bottom-left-radius: 0; }}
        .bot-bubble {{ background-color: {ACCENT_COLOR}; color: #000; border-bottom-right-radius: 0; }}
        .user-icon {{ font-weight: bold; margin-right: 10px; color: {PRIMARY_COLOR}; min-width: 30px; text-align: center; }}
        .bot-icon {{ font-weight: bold; margin-left: 10px; color: #5a4b00; min-width: 30px; text-align: center; }}
        .stButton>button {{ background-color: {PRIMARY_COLOR}; color: white; border-radius: 10px; font-weight: bold; border: none; padding: 0.5rem 1.2rem; }}
        .stButton>button:hover {{ background-color: #015566; transform: scale(1.03); }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🤖 Smart Recruiter Chat Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask any question about candidates\' CVs and get instant answers!</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Ask your question here:", key="input")
        submitted = st.form_submit_button("🔍 Ask")

    def render_message(user_msg, bot_msg):
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="message user-msg">
                    <div class="user-icon">👤</div>
                    <div class="bubble user-bubble">{user_msg}</div>
                </div>
                <div class="message bot-msg">
                    <div class="bubble bot-bubble">🤖 {bot_msg}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted and user_input.strip():
        with st.spinner("Thinking... 🤔"):
            response = rag_engine.query(user_input)
        st.session_state.chat_history.append((user_input, response))

    for user_msg, bot_msg in st.session_state.chat_history[::-1]:
        render_message(user_msg, bot_msg)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
