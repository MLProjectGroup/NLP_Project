# === Main streamlit.py ===
import streamlit as st
import random
import urllib.parse

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
    "Start Recruiting": "Pages.recruiter",   # هنا لازم تتأكدي الملف اسمه recruiter.py
    "About Us": "Pages.about"
}

# --- Get Current Page ---
query_params = st.query_params
current_page_param = query_params.get("page", "Home")

# normalize the page name to match dictionary keys exactly
page_names = list(pages.keys())
if current_page_param in page_names:
    current_page = current_page_param
else:
    current_page = "Home"

# --- Load Page ---
def load_page(page_key):
    mod_name = pages.get(page_key)
    if mod_name:
        mod = __import__(mod_name, fromlist=["app"])
        mod.app()

# --- Google Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- CSS Styling ---
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
        margin: 30px 0;
    }}
    .centered-image img {{
        width: 400px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        margin: 40px auto;
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
        flex-wrap: wrap;
        padding: 10px 0 4px 0;
        border-top: 3px solid {theme['accent']};
        z-index: 999;
    }}
    .bottom-nav a {{
        color: white;
        margin: 0 10px;
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
    .footer-text {{
        text-align: center;
        font-size: 13px;
        color: #444;
        margin-top: 8px;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div style="background-color:{theme['primary']}; padding: 15px; color:white; font-weight:bold; font-size:26px; position: fixed; top:0; width:100%; z-index: 1000; display: flex; justify-content: center; align-items: center;">
    🤖 Smart Recruiter Assistant
</div>
""", unsafe_allow_html=True)

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
for page_name in page_names:
    active = "active" if page_name == current_page else ""
    page_param = urllib.parse.quote(page_name)  # encode to avoid issues with spaces
    footer_html += f'<a href="/?page={page_param}" class="{active}">{page_name.strip()}</a>'

st.markdown(f"""
<div class="bottom-nav">
    {footer_html}
</div>
<p class="footer-text">
    © 2025 Smart Recruiter Assistant. All rights reserved.
</p>
""", unsafe_allow_html=True)
