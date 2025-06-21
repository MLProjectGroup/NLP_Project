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
            max-width: 1000px;
            margin: 0 auto;
            padding: 60px 40px 80px 40px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: {theme['text']};
            line-height: 1.75;
        }}
        .hero {{
            text-align: center;
            padding-top: 60px;
            padding-bottom: 60px;
        }}
        .main-title {{
            color: {theme['primary']};
            font-size: 3rem;
            font-weight: 900;
            margin-bottom: 16px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        .decor-line {{
            width: 100px;
            height: 4px;
            background-color: {theme['accent']};
            border-radius: 4px;
            margin: 14px auto 40px auto;
        }}
        .subtitle {{
            color: {theme['secondary']};
            font-size: 1.4rem;
            font-style: italic;
            margin-bottom: 50px;
            max-width: 80%;
            margin-left: auto;
            margin-right: auto;
        }}
        .features {{
            list-style: none;
            padding: 0;
            margin: 0 auto 60px auto;
            max-width: 700px;
            text-align: left;
        }}
        .features li {{
            display: flex;
            align-items: flex-start;
            font-size: 1.2rem;
            color: {theme['text']};
            margin-bottom: 20px;
            line-height: 1.6;
        }}
        .features li::before {{
            content: "✔";
            color: {theme['primary']};
            font-weight: bold;
            margin-right: 12px;
            font-size: 1.4rem;
            flex-shrink: 0;
        }}
        .why-choose {{
            text-align: center;
            margin-bottom: 60px;
        }}
        .why-choose h3 {{
            color: {theme['primary']};
            font-size: 2.4rem;
            margin-bottom: 20px;
            font-weight: 900;
        }}
        .why-choose p {{
            font-size: 1.2rem;
            color: {theme['secondary']};
            line-height: 1.8;
            max-width: 800px;
            margin: 0 auto;
        }}
        .cta {{
            text-align: center;
            margin-top: 60px;
        }}
        .cta button {{
            background-color: {theme['primary']};
            color: white;
            border: none;
            border-radius: 50px;
            padding: 14px 28px;
            font-size: 1.2rem;
            font-weight: 700;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }}
        .cta button:hover {{
            background-color: #015a71;
            transform: translateY(-2px);
        }}
        .contact {{
            margin-top: 80px;
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
            .main-title {{
                font-size: 2.2rem;
            }}
            .subtitle {{
                font-size: 1.15rem;
            }}
            .features li {{
                font-size: 1.05rem;
            }}
            .why-choose h3 {{
                font-size: 2rem;
            }}
            .why-choose p {{
                font-size: 1.05rem;
            }}
            .cta button {{
                font-size: 1.05rem;
                padding: 12px 24px;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Main container start
    st.markdown('<div class="container fade-in">', unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 class="main-title">Smart Recruiter Assistant</h1>
        <div class="decor-line"></div>
        <p class="subtitle">Reclaim Your Time, Recruit Smarter.</p>
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
    <ul class="features">
        <li>AI helps you quickly find the right candidates.</li>
        <li>A smart assistant guides you through hiring.</li>
        <li>Clear reports help you make better decisions.</li>
        <li>Automates repetitive tasks to save you time.</li>
        <li>Easy-to-use and secure platform for recruiters.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Why Choose Section
    st.markdown("""
    <div class="why-choose">
        <h3>Why Choose Smart Recruiter Assistant?</h3>
        <p>We combine advanced AI with an easy-to-use design to make your hiring process faster, smarter, and more efficient. Our platform saves you time, improves decision-making, and helps you build stronger teams — all with security and simplicity at its core.</p>
    </div>
    """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div class="cta">
        <button onclick="window.location.href='mailto:menatarek04@gmail.com'">Request a Demo</button>
    </div>
    """, unsafe_allow_html=True)

    # Contact Section
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
