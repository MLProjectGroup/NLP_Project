import streamlit as st

def app():
    theme = {
        "primary": "#017691",
        "secondary": "#333",
        "accent": "#dce3e4",
        "background": "#dce3e4",
        "text": "#222222",
        "highlight": "#abc2c7"
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
            max-width: 600px;
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
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 10px;
        }}
        .decor-line {{
            width: 80px;
            height: 4px;
            background-color: {theme['accent']};
            border-radius: 4px;
            margin: 10px auto 40px auto;
        }}
        .subtitle {{
            color: {theme['secondary']};
            font-size: 1.2rem;
            font-style: italic;
            margin-bottom: 40px;
            max-width: 100%;
        }}
        .features {{
            list-style: none;
            padding: 0;
            max-width: 100%;
            text-align: right;
        }}
        .features li {{
            background: {theme['highlight']};
            color: {theme['primary']};
            font-weight: 600;
            font-size: 1.15rem;
            margin: 14px 0;
            padding: 14px 22px 14px 50px;
            border-radius: 10px;
            box-shadow: 0 3px 8px rgba(46, 125, 50, 0.15);
            position: relative;
            transition: background-color 0.3s ease;
            direction: ltr;
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
            direction: ltr;
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
        @media (max-width: 768px) {{
            .container {{
                margin: 0 20px 40px 20px;
                padding: 30px 20px 30px 20px;
            }}
            .main-title {{
                font-size: 1.7rem;
            }}
            .subtitle {{
                font-size: 1rem;
                margin-bottom: 28px;
            }}
            .features li {{
                font-size: 1rem;
                padding: 12px 18px 12px 45px;
            }}
            .why-choose h3 {{
                font-size: 1.5rem;
            }}
            .why-choose p {{
                font-size: 1rem;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fade-in container">', unsafe_allow_html=True)

    st.markdown("""
    <div class="why-choose">
        <h3>Smart Recruiter Assistant</h3>
    </div>
    <div class="decor-line"></div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="subtitle">Reclaim Your Time, Recruit Smarter.</p>', unsafe_allow_html=True)

    # Features list
    st.markdown("""
    <ul class="features">
        <li>AI helps you quickly find the right candidates.</li>
        <li>A smart assistant guides you through hiring.</li>
        <li>Clear reports help you make better decisions.</li>
        <li>Automates repetitive tasks to save you time.</li>
        <li>Easy-to-use and secure platform for recruiters.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Why choose section
    st.markdown("""
    <div class="why-choose">
        <h3>Why Choose Smart Recruiter Assistant?</h3>
        <p>We combine advanced AI with an easy-to-use design to make your hiring process faster, smarter, and more efficient. Our platform saves you time, improves decision-making, and helps you build stronger teams — all with security and simplicity at its core.</p>
    </div>
    """, unsafe_allow_html=True)

 
  
