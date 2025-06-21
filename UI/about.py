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
            max-width: 800px;
            margin: 0 auto 60px auto;
            padding: 40px 40px 50px 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: {theme['text']};
            line-height: 1.75;
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
            font-weight: 900;
            margin-bottom: 12px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        .decor-line {{
            width: 90px;
            height: 4px;
            background-color: {theme['accent']};
            border-radius: 4px;
            margin: 12px auto 40px auto;
        }}
        .subtitle {{
            color: {theme['secondary']};
            font-size: 1.3rem;
            font-style: italic;
            margin-bottom: 40px;
            max-width: 90%;
        }}
        .features {{
            list-style: none;
            padding: 0;
            margin: 0 auto;
            max-width: 600px;
            text-align: left;
        }}
        .features li {{
            display: flex;
            align-items: flex-start;
            font-size: 1.15rem;
            color: {theme['text']};
            margin-bottom: 18px;
            line-height: 1.6;
        }}
        .features li::before {{
            content: "✔";
            color: {theme['primary']};
            font-weight: bold;
            margin-right: 12px;
            font-size: 1.3rem;
            flex-shrink: 0;
        }}
        .why-choose {{
            max-width: 100%;
            margin-top: 80px;
            text-align: center;
            direction: ltr;
        }}
        .why-choose h3 {{
            color: {theme['primary']};
            font-size: 2.2rem;
            margin-bottom: 20px;
            font-weight: 900;
        }}
        .why-choose p {{
            font-size: 1.15rem;
            color: {theme['secondary']};
            line-height: 1.7;
        }}
        .contact {{
            margin-top: 50px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            color: {theme['secondary']};
            letter-spacing: 0.05em;
        }}
        .contact a {{
            color: {theme['primary']};
            text-decoration: none;
            font-weight: 700;
            transition: color 0.3s ease;
        }}
        .contact a:hover {{
            color: #000;
            text-decoration: underline;
            cursor: pointer;
        }}
        @media (max-width: 768px) {{
            .container {{
                margin: 0 20px 40px 20px;
                padding: 30px 20px 30px 20px;
            }}
            .main-title {{
                font-size: 2rem;
            }}
            .subtitle {{
                font-size: 1.1rem;
                margin-bottom: 28px;
            }}
            .features li {{
                font-size: 1rem;
            }}
            .why-choose h3 {{
                font-size: 1.7rem;
            }}
            .why-choose p {{
                font-size: 1.05rem;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Main container start
    st.markdown('<div class="container fade-in">', unsafe_allow_html=True)

    # Title + subtitle + features
    st.markdown("""
    <h1 class="main-title">Smart Recruiter Assistant</h1>
    <div class="decor-line"></div>
    <p class="subtitle">Reclaim Your Time, Recruit Smarter.</p>
    <ul class="features">
        <li>AI helps you quickly find the right candidates.</li>
        <li>A smart assistant guides you through hiring.</li>
        <li>Clear reports help you make better decisions.</li>
        <li>Automates repetitive tasks to save you time.</li>
        <li>Easy-to-use and secure platform for recruiters.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Why Choose section
    st.markdown("""
    <div class="why-choose">
        <h3>Why Choose Smart Recruiter Assistant?</h3>
        <p>We combine advanced AI with an easy-to-use design to make your hiring process faster, smarter, and more efficient. Our platform saves you time, improves decision-making, and helps you build stronger teams — all with security and simplicity at its core.</p>
    </div>
    """, unsafe_allow_html=True)

    # Contact section
    st.markdown("""
    <div class="contact">
        <p>Contact us: 
            <a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a> | 
            <a href="mailto:israaabdelghany9@gmail.com">israaabdelghany9@gmail.com</a> | 
            <a href="mailto:nagwammatia919@gmail.com">nagwammatia919@gmail.com</a> | 
            <a href="mailto:mohamedsalama152019@gmail.com">mohamedsalama152019@gmail.com</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main container end
    st.markdown('</div>', unsafe_allow_html=True)
