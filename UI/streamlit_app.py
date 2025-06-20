import streamlit as st
import random

# --- App Config ---
st.set_page_config(
    page_title="Smart Recruiter Assistant 🤖",
    layout="wide",
    page_icon="https://raw.githubusercontent.com/MLProjectGroup/NLP_Project/main/UI/assets/hr_man.png"
)

# --- Theme Colors ---
theme = {
    "primary": "#017691",
    "secondary": "#FF9F1C",
    "accent": "#dce3e4",
    "background": "#dce3e4",
    "text": "#222222"
}

# --- Daily Tips ---
daily_tips = [
    "Always personalize your hiring message!",
    "Look beyond keywords, consider potential.",
    "Soft skills matter as much as experience.",
    "Diversity is a strength in hiring!",
    "Hiring is like dating... look for culture fit!",
]

# --- Pages Dictionary ---
pages = {
    "Home": None,
    "Upload CVs": "Pages.01_Upload",
    "Chatbot Q&A": "Pages.02_Chatbot",
    "Matcher": "Pages.03_Matcher",
    "Summarizer": None,
    "Recommender": "Pages.04_Recommender",
    "Dashboard": "Pages.05_Dashboard"
}

# ✅ --- Get current page using modern Streamlit method ---
query_params = st.query_params
current_page = query_params.get("page", ["Home"])[0]

if current_page not in pages:
    current_page = "Home"

# --- Load Page Dynamically ---
def load_page(page_key):
    mod_name = pages.get(page_key)
    if mod_name:
        mod = __import__(mod_name, fromlist=['app'])
        mod.app()

# --- Load Google Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Styling ---
st.markdown(f"""
<style>
    body, .stApp {{
        background-color: {theme['background']};
        direction: ltr;
        font-family: 'Poppins', sans-serif;
    }}

    .fade-in {{
        animation: fadeIn 0.8s ease-in-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .main-title {{
        color: {theme['primary']};
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0 10px;
    }}

    .quote {{
        font-size: 22px;
        color: {theme['primary']};
        text-align: center;
        font-style: italic;
        font-weight: 600;
        margin: 30px 0 30px 0;
    }}

    .centered-image img {{
        width: 400px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        margin: 40px auto 20px auto;
        display: block;
    }}
    .centered-image img:hover {{
        transform: scale(1.05);
    }}

    .nav-button {{
        background-color: {theme['primary']};
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .nav-button:hover {{
        background-color: {theme['secondary']};
        color: black;
    }}
</style>
""", unsafe_allow_html=True)

# --- Fixed Header ---
st.markdown(f"""
<div style="
    background-color:{theme['primary']}; 
    padding: 15px; 
    color:white; 
    font-weight:bold; 
    font-size:26px; 
    position: fixed; 
    top:0; 
    width:100%; 
    z-index: 1000; 
    display: flex; 
    justify-content: center; 
    align-items: center;
">
    🤖 Smart Recruiter Assistant
</div>
""", unsafe_allow_html=True)

# --- Fade-In Container Start ---
st.markdown('<div class="fade-in" style="margin-top:80px;">', unsafe_allow_html=True)

# --- Render Page Content ---
if current_page == "Home":
    st.markdown('<div class="main-title"> Reclaim Your Time, Recruit Smarter.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="centered-image">
        <img src="https://raw.githubusercontent.com/MLProjectGroup/NLP_Project/main/UI/assets/hr_man.png" alt="HR Assistant">
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="quote"><b>Daily Tip:</b> <br> {random.choice(daily_tips)}</div>', unsafe_allow_html=True)
else:
    load_page(current_page)

# --- Fade-In Container End ---
st.markdown('</div>', unsafe_allow_html=True)

# --- Bottom Navigation ---
with st.container():
    nav_cols = st.columns(len(pages))
    for i, (page_name, page_module) in enumerate(pages.items()):
        with nav_cols[i]:
            if st.button(page_name, use_container_width=True, key=f"nav_{page_name}"):
                st.query_params.page = page_name
                st.experimental_rerun()
