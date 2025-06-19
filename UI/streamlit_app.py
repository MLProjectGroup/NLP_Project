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

# --- Sidebar ---
st.sidebar.image(
    "https://i.pinimg.com/originals/5b/d6/4a/5bd64ad031e917fcb1b4ad267b3e16d5.gif", 
    width=220
)
st.sidebar.title("Smart Recruiter Assistant 🤖")
st.sidebar.markdown("**with your helpful AI partner:**")
st.sidebar.markdown("### Roz 🤓")
st.sidebar.markdown("_Always watching... always recruiting!_")

# --- Tabs Navigation (with Professional + Funny icons) ---
tab = st.sidebar.radio(
    "Navigate to:",
    [
        "📂 Upload CVs",
        "💬 Chatbot Q&A",
        "📝 Matching & Summarizer",
        "🎯 Recommender",
        "📊 Dashboard",
    ]
)

# --- Background Styling ---
if tab == "📂 Upload CVs":
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://media.giphy.com/media/xTiTnpVDp86JbFCdzi/giphy.gif");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #0E1117;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Tabs Content ---
if tab == "📂 Upload CVs":
    st.header("📂 Upload CVs")
    st.info("Upload your CVs here. Supported formats: PDF, DOCX, TXT.")
    st.success("Roz is watching... 👀 Ready to analyze your CVs!")

elif tab == "💬 Chatbot Q&A":
    st.header("💬 Chatbot Q&A")
    st.info("Ask questions about the candidates.")
    st.markdown("Example: *'Who has time series experience?'*")
    st.success("Roz is listening carefully... 🎧")

elif tab == "📝 Matching & Summarizer":
    st.header("📝 Matching & Summarizer")
    st.info("Upload a job description and view top matching candidates.")
    st.success("Roz is preparing her matches... 🔍")

elif tab == "🎯 Recommender":
    st.header("🎯 Recommender")
    st.info("See job recommendations for each candidate.")
    st.success("Roz's recommendation engine is running... 🧠")

elif tab == "📊 Dashboard":
    st.header("📊 Dashboard")
    st.info("Overview of all candidates.")
    st.success("Roz is generating analytics... 📊")

# --- Footer ---
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <center>
    🤖 Powered by AI • Watched by Roz 👀  
    Created by MLProjectGroup • <a href="https://github.com/MLProjectGroup/NLP_Project" target="_blank">GitHub Repo</a>
    </center>
    """,
    unsafe_allow_html=True
)

