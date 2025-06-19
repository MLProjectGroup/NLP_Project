import sys
import os

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# App Config
st.set_page_config(
    page_title="Smart Recruiter Assistant",
    page_icon="🤖",
    layout="wide",
)

# Sidebar
st.sidebar.title("Smart Recruiter Assistant")
tab = st.sidebar.radio(
    "Go to",
    [
        "📂 Upload CVs",
        "💬 Chatbot Q&A",
        "📝 Matching & Summarizer",
        "🎯 Recommender",
        "📊 Dashboard",
    ]
)

# Pages
if tab == "📂 Upload CVs":
    st.header("📂 Upload CVs")
    st.info("Upload your CVs here. Supported formats: PDF, DOCX, TXT.")
    # (هنضيف هنا كود upload قريباً)

elif tab == "💬 Chatbot Q&A":
    st.header("💬 Chatbot Q&A")
    st.info("Ask questions about the candidates. Example: 'Who has time series experience?'")
    # (هنضيف هنا كود chatbot قريباً)

elif tab == "📝 Matching & Summarizer":
    st.header("📝 Matching & Summarizer")
    st.info("Upload a job description and view top matching candidates.")
    # (هنضيف هنا كود matcher + feedback قريباً)

elif tab == "🎯 Recommender":
    st.header("🎯 Recommender")
    st.info("See job recommendations for each candidate.")
    # (هنضيف هنا كود recommender قريباً)

elif tab == "📊 Dashboard":
    st.header("📊 Dashboard")
    st.info("Overview of all candidates.")
    # (هنضيف هنا كود dashboard قريباً)

# Footer
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <center>Created by MLProjectGroup • <a href="https://github.com/MLProjectGroup/NLP_Project" target="_blank">GitHub Repo</a></center>
    """,
    unsafe_allow_html=True
)
