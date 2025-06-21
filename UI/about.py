import streamlit as st

def app():    
    landing_page_html = """
    <style>
        /* Reset + Base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #017691;
            background-color: #dce3e4;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        /* Hero */
        .hero {
            background: linear-gradient(135deg, #017691 0%, #015a70 100%);
            color: white;
            padding: 100px 0;
            text-align: center;
        }
        .hero h1 {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .hero p {
            font-size: 1.3rem;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        /* Features */
        .features {
            padding: 80px 0;
            background: white;
            color: #017691;
        }
        .section-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 3rem;
            color: #017691;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }
        .feature-item {
            display: flex;
            align-items: flex-start;
            padding: 1.5rem;
            border-radius: 12px;
            background: #abc2c7;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .feature-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(1, 118, 145, 0.15);
        }
        .checkmark {
            width: 24px;
            height: 24px;
            background: #017691;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
        }
        .checkmark::after {
            content: '✓';
            color: white;
            font-weight: bold;
            font-size: 14px;
        }
        .feature-content h3 {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .feature-content p {
            font-size: 0.95rem;
        }
        /* Why Choose */
        .why-choose-us {
            padding: 80px 0;
            background: #dce3e4;
        }
        .why-choose-us .section-title {
            color: #017691;
        }
        .why-content {
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
        }
        .why-content p {
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 1.5rem;
            color: #000000; /* BLACK */
        }
        /* Contact */
        .contact {
            padding: 80px 0;
            background: #abc2c7;
        }
        .contact-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 3rem;
        }
        .contact-item {
            background: white;
            padding: 2rem;
            border-radius: 14px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            word-wrap: break-word;
        }
        .contact-item:hover {
            transform: translateY(-5px);
        }
        .contact-item h4 {
            color: #017691;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .contact-item a {
            display: inline-block;
            color: #017691;
            text-decoration: underline;
            font-size: 0.95rem;
            word-break: break-word;
            margin-top: 0.5rem;
        }
        .contact-item a:hover {
            color: #015a70;
        }
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 2.5rem; }
            .hero p { font-size: 1.1rem; }
            .section-title { font-size: 2rem; }
            .features-grid { grid-template-columns: 1fr; }
            .contact-grid { grid-template-columns: 1fr; }
        }
    </style>

    <section class="hero">
        <div class="container">
            <h1>Recruite Smarter</h1>
            <p>Innovative solutions designed to streamline your operations and accelerate growth with cutting-edge technology</p>
        </div>
    </section>

    <section class="features">
        <div class="container">
            <h2 class="section-title">Key Features</h2>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Fast</h3>
                        <p>AI helps you quickly find the right candidates.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Advanced Analytics</h3>
                        <p>Clear reports help you make better decisions.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Smart</h3>
                        <p>A smart assistant guides you through hiring</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Save Time</h3>
                        <p>Automates repetitive tasks to save you time.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Easy Usage</h3>
                        <p>Easy-to-use and secure platform for recruiters.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="why-choose-us">
        <div class="container">
            <h2 class="section-title">Why Choose Us</h2>
            <div class="why-content">
                <p>We combine advanced AI with an easy-to-use design to make your hiring process faster, smarter, and more efficient.</p>
                <p>Our platform saves you time, improves decision-making, and helps you build stronger teams — all with security and simplicity at its core.</p>
            </div>
        </div>
    </section>

    <section class="contact">
        <div class="container">
            <h2 class="section-title">Get in Touch</h2>
            <div class="contact-grid">
                <div class="contact-item">
                    <h4>Mennatullah Tarek</h4>
                    <a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a>
                </div>
                <div class="contact-item">
                    <h4>Israa Abdelghany</h4>
                    <a href="mailto:israaabdelghany9@gmail.com">israaabdelghany9@gmail.com</a>
                </div>
                <div class="contact-item">
                    <h4>Nagwa Mohamed</h4>
                    <a href="mailto:nagwammatia919@gmail.com">nagwammatia919@gmail.com</a>
                </div>
                <div class="contact-item">
                    <h4>Mohamed Salama</h4>
                    <a href="mailto:mohamedsalama152019@gmail.com">mohamedsalama152019@gmail.com</a>
                </div>
            </div>
        </div>
    </section>
    """

    st.markdown(landing_page_html, unsafe_allow_html=True)
