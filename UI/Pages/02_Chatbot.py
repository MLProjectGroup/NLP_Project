# Pages/02_Chatbot.py

import streamlit as st
from main import process_custom_query
from RAG.rag_engine import EnhancedRAGEngine
from Preprocessing.vector_store import CVVectorStore

def app():
    st.markdown('<div class="main-title">🔍 Ask Anything About the Candidates</div>', unsafe_allow_html=True)
    st.write("Type your hiring question (e.g., *Who has experience in Python?*)")

    query = st.text_input("Enter your query:", placeholder="e.g., Who has experience in machine learning?")

    if query:
        # Setup engine
        vector_store = CVVectorStore()
        rag_engine = EnhancedRAGEngine(vector_store)

        try:
            with st.spinner("Processing your query... ⏳"):
                top_text, all_relevant_text, top_names = process_custom_query(rag_engine, query)

            st.success("Done ✅")

            st.subheader("🎯 Top 5 Candidates")
            st.code(top_text, language="markdown")

            st.subheader("📋 All Relevant Candidates")
            st.code(all_relevant_text, language="markdown")

        except Exception as e:
            st.error(f"Error: {str(e)}")
