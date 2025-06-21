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
            color: #222222;
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
            position: relative;
            overflow: hidden;
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
        }
        .section-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 600;
            color: #222222;
            margin-bottom: 3rem;
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
            flex-shrink: 0;
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
            color: #222222;
        }
        .feature-content p {
            color: #333;
            font-size: 0.95rem;
        }
        /* Why Choose */
        .why-choose-us {
            padding: 80px 0;
            background: #dce3e4;
        }
        .why-content {
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
        }
        .why-content p {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #333;
            margin-bottom: 1.5rem;
        }
        /* CTA */
        .cta-section {
            padding: 80px 0;
            background: white;
            text-align: center;
        }
        .cta-button {
            display: inline-block;
            background: linear-gradient(135deg, #017691 0%, #015a70 100%);
            color: white;
            padding: 18px 40px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(1, 118, 145, 0.3);
            margin-top: 1rem;
            text-decoration: none;
        }
        .cta-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(1, 118, 145, 0.4);
            color: white;
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
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
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
            color: #333;
            text-decoration: none;
            font-size: 0.95rem;
        }
        .contact-item a:hover {
            color: #017691;
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
            <h1>Transform Your Business</h1>
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
                        <h3>Advanced Analytics</h3>
                        <p>Deep insights into your data with real-time reporting capabilities.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Seamless Integration</h3>
                        <p>Connect with your tools and workflows via our API.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Enterprise Security</h3>
                        <p>Bank-level security with advanced threat protection.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>24/7 Support</h3>
                        <p>Expert support team ready to help you around the clock.</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="checkmark"></div>
                    <div class="feature-content">
                        <h3>Scalable Architecture</h3>
                        <p>Built to grow with your business, from startup to enterprise.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="why-choose-us">
        <div class="container">
            <h2 class="section-title">Why Choose Us</h2>
            <div class="why-content">
                <p>We combine cutting-edge technology with intuitive design, making complex processes simple and efficient.</p>
                <p>99.9% uptime, enterprise-grade security, and a team dedicated to your success.</p>
            </div>
        </div>
    </section>

    <section class="cta-section">
        <div class="container">
            <h2 class="section-title">Ready to Get Started?</h2>
            <p>Experience our platform with a personalized demo for your business needs.</p>
            <a href="#" class="cta-button">Request a Demo</a>
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
