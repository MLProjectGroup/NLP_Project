import streamlit as st
import random

# --- App Config ---
st.set_page_config(
    page_title="Smart Recruiter Assistant 🤖",
    layout="wide",
    page_icon="https://i.pinimg.com/originals/5b/d6/4a/5bd64ad031e917fcb1b4ad267b3e16d5.gif"
)

# --- Theme Colors ---
theme = {
    "primary": "#490b4a",         
    "secondary": "#FF6EC7",        
    "accent": "#fefefe",          
    "background": "#F8EAF9",      
    "text": "#2B002C"             
}


# --- Daily Tips ---
daily_tips = [
    "Always personalize your hiring message! 🎯",
    "Look beyond keywords, consider potential. 💡",
    "Soft skills matter as much as experience. 🤝",
    "Diversity is a strength in hiring! 🌍",
    "Hiring is like dating... look for culture fit! 💌",
]

# --- Pages ---
pages = {
    "🏠 Home": None,
    "  Upload CVs": "Pages.01_Upload",
    "💬 Chatbot Q&A": "Pages.02_Chatbot",
    " Matcher": "Pages.03_Matcher",
    " Summarizer": None , 
    "Recommender": "Pages.04_Recommender",
    "Dashboard": "Pages.05_Dashboard"
}

# --- Current Page ---
query_params = st.query_params
current_page = query_params.get("page", "🏠 Home")

if current_page not in pages:
    current_page = "🏠 Home"

# --- Load Page ---
def load_page(page_key):
    mod_name = pages.get(page_key)
    if mod_name:
        mod = __import__(mod_name, fromlist=['app'])
        mod.app()

# --- CSS Styling ---
st.markdown(f"""
<style>
    body, .stApp {{
        background-color: {theme['background']};
        direction: ltr;
        font-family: 'Open Sans', sans-serif;
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
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0 10px;
        text-shadow: 0 0 8px {theme['accent']};
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
    .bottom-nav {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: {theme['primary']};
        display: flex;
        justify-content: center;
        padding: 12px 0;
        border-top: 3px solid {theme['accent']};
        z-index: 999;
    }}
    .bottom-nav a {{
        color: white;
        margin: 0 15px;
        text-decoration: none;
        font-weight: bold;
        font-size: 14px;
        padding: 6px 12px;
        border-radius: 8px;
        transition: background-color 0.3s;
        cursor: pointer;
    }}
    .bottom-nav a:hover {{
        background-color: {theme['accent']};
        color: black;
    }}
    .bottom-nav a.active {{
        background-color: {theme['accent']};
        color: black;
    }}
</style>
""", unsafe_allow_html=True)

# --- Header Fixed ---
st.markdown(f"""
<div style="background-color:{theme['primary']}; padding: 15px; color:white; text-align:center; font-weight:bold; font-size:26px; position: fixed; top:0; width:100%; z-index: 1000;">
    🤖 Smart Recruiter Assistant - powered by Roz 👀
</div>
""", unsafe_allow_html=True)

# --- Fade In Start ---
st.markdown('<div class="fade-in" style="margin-top:80px;">', unsafe_allow_html=True)

# --- Pages Content ---
if current_page == "🏠 Home":
    st.markdown('<div class="main-title"> Reclaim Your Time, Recruit Smarter: The Future of HR is Here. </div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="centered-image">
        <img src="https://raw.githubusercontent.com/MLProjectGroup/NLP_Project/main/UI/assets/hr.jpg" alt="HR Assistant">

    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="quote">✨ Daily  Tip: <br> {random.choice(daily_tips)}</div>', unsafe_allow_html=True)
else:
    load_page(current_page)

# --- Fade In End ---
st.markdown('</div>', unsafe_allow_html=True)

# --- Bottom Nav ---
footer_html = ""
for page_name in pages.keys():
    active = "active" if page_name == current_page else ""
    footer_html += f'<a href="/?page={page_name}" class="{active}">{page_name}</a>'

st.markdown(f'<div class="bottom-nav">{footer_html}</div>', unsafe_allow_html=True)
