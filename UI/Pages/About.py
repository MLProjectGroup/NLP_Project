import streamlit as st

def app():
    theme = {
        "primary": "#2E7D32",
        "secondary": "#00796B",
        "accent": "#FFC107",
        "background": "#F9F9F9",
        "text": "#333333",
        "highlight": "#AED581"
    }

    st.markdown(f"""
    <style>
        html, body, .main {{
            height: 100%;
            background-color: {theme['background']};
            margin: 0;
            padding: 0;
        }}
        .fade-in {{
            animation: fadeIn 0.9s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(25px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .container {{
            max-width: 700px;
            margin: 0 auto 60px auto;
            padding: 40px 40px 50px 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
            margin-bottom: 10px;
        }}
        .decor-line {{
            width: 100px;
            height: 4px;
            background-color: {theme['accent']};
            border-radius: 4px;
            margin: 10px auto 40px auto;
        }}
        .subtitle {{
            color: {theme['secondary']};
            font-size: 1.3rem;
            font-style: italic;
            margin-bottom: 40px;
            max-width: 100%;
        }}
        .features {{
            list-style: none;
            padding: 0;
            max-width: 100%;
            text-align: left;
        }}
        .features li {{
            background: {theme['highlight']};
            color: {theme['primary']};
            font-weight: 600;
            font-size: 1.1rem;
            margin: 14px 0;
            padding: 14px 22px;
            border-radius: 10px;
            box-shadow: 0 3px 8px rgba(46, 125, 50, 0.15);
            position: relative;
            transition: background-color 0.3s ease;
        }}
        .features li:hover {{
            background-color: {theme['accent']};
            color: #222;
            cursor: default;
            box-shadow: 0 5px 15px rgba(255, 193, 7, 0.3);
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
        .why-choose {{
            max-width: 100%;
            margin-top: 80px;
            text-align: center;
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
        .contact {{
            margin-top: 30px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            color: {theme['secondary']};
            letter-spacing: 0.05em;
        }}
        .contact a {{
            color: {theme['accent']};
            text-decoration: none;
            font-weight: 700;
            transition: color 0.3s ease;
        }}
        .contact a:hover {{
            color: {theme['primary']};
            text-decoration: underline;
            cursor: pointer;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fade-in container">', unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div class="main-title">Smart Recruiter Assistant</div>
    <div class="decor-line"></div>
    """, unsafe_allow_html=True)

    # Subtitle
    st.markdown('<p class="subtitle">Your AI-powered partner for intelligent CV analysis, job matching, and HR insights</p>', unsafe_allow_html=True)

    # Features list
    st.markdown("""
    <ul class="features">
        <li>Upload, process, and analyze multiple CVs with ease</li>
        <li>Advanced search for skills, experience, and qualifications</li>
        <li>AI-powered job matching and ranking</li>
        <li>Generate professional CV summaries</li>
        <li>Tailored HR interview questions for candidates</li>
        <li>Visual dashboards and insights for recruiters</li>
    </ul>
    """, unsafe_allow_html=True)

    # Why choose section
    st.markdown("""
    <div class="why-choose">
        <h3>Why choose Smart Recruiter Assistant?</h3>
        <p>Designed to streamline recruitment workflows, save time, and uncover top talent effortlessly. Combining the latest in AI-driven document analysis with an intuitive interface, it's the smart way to hire!</p>
    </div>
    """, unsafe_allow_html=True)

    # Contact info
    st.markdown("""
    <div class="contact">
        For inquiries & support: 
        <a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
