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
    "primary": "#017691",          # Main header, buttons
    "secondary": "#FF9F1C",        # Highlights, accents
    "accent": "#e0e0e0",           # Lines, borders
    "background": "#dce3e4",       # App background
    "text": "#222222"              # Base text
}

# Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# --- Daily Tips ---
daily_tips = [
    "Always personalize your hiring message!",
    "Look beyond keywords, consider potential.",
    "Soft skills matter as much as experience.",
    "Diversity is a strength in hiring!",
    "Hiring is like dating... look for culture fit!",
]

# --- Pages ---
pages = {
    "Home": None,
    "Upload CVs": "Pages.01_Upload",
    "Chatbot Q&A": "Pages.02_Chatbot",
    "Matcher": "Pages.03_Matcher",
    "Summarizer": None , 
    "Recommender": "Pages.04_Recommender",
    "Dashboard": "Pages.05_Dashboard"
}

# --- Current Page ---
query_params = st.query_params
current_page = query_params.get("page", "Home")   

if current_page not in pages:
    current_page = "Home"

# --- Load Page ---
def load_page(page_key):
    mod_name = pages.get(page_key)
    if mod_name:
        mod = __import__(mod_name, fromlist=['app'])
        mod.app()

# --- Google Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Global Styles (CSS) ---
st.markdown(f"""
<style>

p, label {{
    color: {theme['text']};
}}

.stFileUploaderFileName {{
    color: #000 !important;
}}

.stButton > button {{
    background-color: white !important;  /* or any color */
    color: white !important;
}}

.st-emotion-cache-1wbqy5l.e17qgqm80 {{
    color: #222222 !important;  /* or use theme['text'] value */
}}

.section-box {{
    background: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    border: none !important;
}}

/* General App Background & Font */
body, .stApp {{
    background-color: {theme['background']};
    font-family: 'Poppins', sans-serif;
    direction: ltr;
}}

/* Animation */
.fade-in {{
    animation: fadeIn 0.8s ease-in-out;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Main Title */
.main-title {{
    color: {theme['primary']};
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    margin: 20px 0 10px;
}}

/* Quote or Daily Tip Box */
.quote {{
    font-size: 22px;
    color: {theme['primary']};
    text-align: center;
    font-style: italic;
    font-weight: 600;
    margin: 30px 0;
}}

/* Image Block */
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

/* Header Bar */
.header {{
    background-color: {theme['primary']};
    padding: 15px;
    color: white;
    font-weight: bold;
    font-size: 26px;
    top: 0;
    width: 100%;
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
}}

/* Bottom Navigation */
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

.bottom-nav a:hover,
.bottom-nav a.active {{
    background-color: {theme['accent']};
    color: black;
}}

/* Force white background on file uploader box */
div[role="presentation"] {{
    background-color: white !important;
    border: 2px dashed #ccc !important;
    border-radius: 12px !important;
    color: #222 !important;
    padding: 20px !important;
}}


/* Responsive Styles for Small Screens */
@media (max-width: 768px) {{
    .main-title {{
        font-size: 26px;
    }}
    .quote {{
        font-size: 18px;
    }}
    .centered-image img {{
        width: 90%;
    }}
    .header {{
        font-size: 20px;
        padding: 10px;
    }}
    .bottom-nav a {{
        font-size: 12px;
        padding: 4px 8px;
    }}
}}

</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f'<div class="header">🤖 Smart Recruiter Assistant</div>', unsafe_allow_html=True)

# --- Fade In Start ---
st.markdown('<div class="fade-in" style="margin-top:80px;">', unsafe_allow_html=True)

# --- Pages Content ---
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


# --- Fade In End ---
st.markdown('</div>', unsafe_allow_html=True)


# --- Bottom Nav ---
footer_html = ""
for page_name in pages.keys():
    active = "active" if page_name == current_page else ""
    footer_html += f'<a href="/?page={page_name}" class="{active}">{page_name}</a>'

st.markdown(f'<div class="bottom-nav">{footer_html}</div>', unsafe_allow_html=True)
