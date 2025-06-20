import streamlit as st

def app():
    theme = {
        "primary": "#017691",
        "secondary": "#FF9F1C",
        "accent": "#dce3e4",
        "background": "#dce3e4",
        "text": "#222222"
    }

    st.markdown(f"""
    <style>
        html, body, .main {{
            height: 100%;
            background-color: {theme['background']};
            margin: 0;
            padding: 0;
            direction: ltr;
            font-family: 'Poppins', sans-serif;
        }}
        .fade-in {{
            animation: fadeIn 0.9s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(25px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto 60px auto;
            padding: 40px 40px 50px 40px;
            color: {theme['text']};
            line-height: 1.65;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            min-height: 100vh;
            justify-content: center;
        }}
        .main-title {{
            color: {theme['primary']};
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 20px;
        }}
        .subtitle {{
            color: {theme['secondary']};
            font-size: 1.3rem;
            font-style: italic;
            margin-bottom: 30px;
            max-width: 100%;
        }}
        .features {{
            list-style: none;
            padding: 0;
            max-width: 100%;
        }}
        .features li {{
            background: {theme['accent']};
            color: {theme['primary']};
            font-weight: 600;
            font-size: 1.1rem;
            margin: 12px 0;
            padding: 14px 22px;
            border-radius: 10px;
            box-shadow: 0 3px 8px rgba(1, 118, 145, 0.15);
            position: relative;
            transition: background-color 0.3s ease;
        }}
        .features li:hover {{
            background-color: {theme['secondary']};
            color: white;
            cursor: default;
        }}
        .features li::before {{
            content: "✔";
            position: absolute;
            left: 18px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: 900;
            font-size: 1.3rem;
            color: {theme['primary']};
        }}
        .why-choose h3 {{
            color: {theme['primary']};
            font-size: 2rem;
            margin-bottom: 20px;
            font-weight: 800;
        }}
        .why-choose p {{
            font-size: 1.1rem;
            color: {theme['secondary']};
            line-height: 1.6;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fade-in container">', unsafe_allow_html=True)

    st.markdown("""
    <div class="main-title">Smart Recruiter Assistant</div>
    <div class="subtitle">Empowering Recruiters with AI-driven insights and tools</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul class="features">
        <li>AI-powered candidate screening</li>
        <li>Smart interview scheduling and follow-ups</li>
        <li>Real-time insights on candidate fit</li>
        <li>Automated communication and engagement</li>
        <li>Data-driven hiring recommendations</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="why-choose">
        <h3>Why Choose Us?</h3>
        <p>Because we blend cutting-edge AI with human intuition to streamline your hiring process, save you time, and help you find the best candidates faster and more effectively.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="contact">
        For more info: <a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer Navigation ---
    pages = {
        "Home  ": None,
        "   Start Recruiting   ": "Pages.Chatbot_02",
        "   About Us": "Pages.About"
    }

    query_params = st.query_params
    current_page = query_params.get("page", "   About Us")

    footer_html = ""
    for page_name in pages.keys():
        active = "active" if page_name == current_page else ""
        footer_html += f'<a href="/?page={page_name}" class="{active}">{page_name.strip()}</a>'

    st.markdown(f"""
    <div class="bottom-nav" style="
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
        z-index: 999;">
        {footer_html}
    </div>
    <p class="footer-text" style="text-align: center; font-size: 13px; color: #444; margin-top: 8px; margin-bottom: 8px;">
        © 2025 Smart Recruiter Assistant. All rights reserved.
    </p>
    """, unsafe_allow_html=True)
