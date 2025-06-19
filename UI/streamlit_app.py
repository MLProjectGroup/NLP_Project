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

# --- Tabs ---
PAGES = {
    "📂 Upload CVs": "Pages/01_Upload.py",
    "💬 Chatbot Q&A": "Pages/02_Chatbot.py",
    "📝 Matching & Summarizer": "Pages/03_Matcher.py",
    "🎯 Recommender": "Pages/04_Recommender.py",
    "📊 Dashboard": "Pages/05_Dashboard.py",
}

tab = st.sidebar.radio("Navigate to:", list(PAGES.keys()))

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

# --- Import and run selected page ---
with open(PAGES[tab]) as f:
    code = f.read()
    exec(code, globals())

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
