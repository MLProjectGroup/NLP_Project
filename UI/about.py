def app():    
    landing_page_html = """[
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professional Landing Page</title>
        <style>
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
    
            /* Hero Section */
            .hero {
                background: linear-gradient(135deg, #017691 0%, #015a70 100%);
                color: white;
                padding: 100px 0;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
    
            .hero::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.1;
            }
    
            .hero-content {
                position: relative;
                z-index: 1;
                opacity: 0;
                transform: translateY(30px);
                animation: fadeInUp 1s ease-out forwards;
            }
    
            .hero h1 {
                font-size: 3.5rem;
                font-weight: 700;
                margin-bottom: 1rem;
                letter-spacing: -0.02em;
            }
    
            .hero p {
                font-size: 1.3rem;
                margin-bottom: 2rem;
                opacity: 0.9;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }
    
            /* Features Section */
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
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.8s ease-out 0.2s forwards;
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
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.8s ease-out forwards;
            }
    
            .feature-item:nth-child(1) { animation-delay: 0.3s; }
            .feature-item:nth-child(2) { animation-delay: 0.4s; }
            .feature-item:nth-child(3) { animation-delay: 0.5s; }
            .feature-item:nth-child(4) { animation-delay: 0.6s; }
            .feature-item:nth-child(5) { animation-delay: 0.7s; }
    
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
    
            /* Why Choose Us Section */
            .why-choose-us {
                padding: 80px 0;
                background: #dce3e4;
            }
    
            .why-content {
                max-width: 800px;
                margin: 0 auto;
                text-align: center;
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.8s ease-out 0.3s forwards;
            }
    
            .why-content p {
                font-size: 1.1rem;
                line-height: 1.8;
                color: #333;
                margin-bottom: 1.5rem;
            }
    
            /* CTA Section */
            .cta-section {
                padding: 80px 0;
                background: white;
                text-align: center;
            }
    
            .cta-content {
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.8s ease-out 0.4s forwards;
            }
    
            .cta-button {
                display: inline-block;
                background: linear-gradient(135deg, #017691 0%, #015a70 100%);
                color: white;
                padding: 18px 40px;
                text-decoration: none;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(1, 118, 145, 0.3);
                margin-top: 1rem;
            }
    
            .cta-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(1, 118, 145, 0.4);
                text-decoration: none;
                color: white;
            }
    
            /* Contact Section */
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
                opacity: 0;
                transform: translateY(20px);
                animation: fadeInUp 0.8s ease-out forwards;
            }
    
            .contact-item:nth-child(1) { animation-delay: 0.5s; }
            .contact-item:nth-child(2) { animation-delay: 0.6s; }
            .contact-item:nth-child(3) { animation-delay: 0.7s; }
            .contact-item:nth-child(4) { animation-delay: 0.8s; }
    
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
                word-break: break-word;
                transition: color 0.3s ease;
            }
    
            .contact-item a:hover {
                color: #017691;
            }
    
            /* Animations */
            @keyframes fadeInUp {
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
    
            /* Responsive Design */
            @media (max-width: 768px) {
                .hero h1 {
                    font-size: 2.5rem;
                }
    
                .hero p {
                    font-size: 1.1rem;
                }
    
                .section-title {
                    font-size: 2rem;
                }
    
                .features-grid {
                    grid-template-columns: 1fr;
                    gap: 1.5rem;
                }
    
                .contact-grid {
                    grid-template-columns: 1fr;
                    gap: 1rem;
                }
    
                .container {
                    padding: 0 15px;
                }
    
                .hero, .features, .why-choose-us, .cta-section, .contact {
                    padding: 60px 0;
                }
            }
    
            @media (max-width: 480px) {
                .hero h1 {
                    font-size: 2rem;
                }
    
                .hero p {
                    font-size: 1rem;
                }
    
                .section-title {
                    font-size: 1.8rem;
                }
    
                .cta-button {
                    padding: 16px 30px;
                    font-size: 1rem;
                }
            }
        </style>
    </head>
    <body>
        <!-- Hero Section -->
        <section class="hero">
            <div class="container">
                <div class="hero-content">
                    <h1>Transform Your Business</h1>
                    <p>Innovative solutions designed to streamline your operations and accelerate growth with cutting-edge technology</p>
                </div>
            </div>
        </section>
    
        <!-- Features Section -->
        <section class="features">
            <div class="container">
                <h2 class="section-title">Key Features</h2>
                <div class="features-grid">
                    <div class="feature-item">
                        <div class="checkmark"></div>
                        <div class="feature-content">
                            <h3>Advanced Analytics</h3>
                            <p>Get deep insights into your data with powerful analytics tools and real-time reporting capabilities</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="checkmark"></div>
                        <div class="feature-content">
                            <h3>Seamless Integration</h3>
                            <p>Connect with your existing tools and workflows through our comprehensive API and integration platform</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="checkmark"></div>
                        <div class="feature-content">
                            <h3>Enterprise Security</h3>
                            <p>Bank-level security with end-to-end encryption, compliance standards, and advanced threat protection</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="checkmark"></div>
                        <div class="feature-content">
                            <h3>24/7 Support</h3>
                            <p>Round-the-clock customer support from our expert team to ensure your success at every step</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="checkmark"></div>
                        <div class="feature-content">
                            <h3>Scalable Architecture</h3>
                            <p>Built to grow with your business, from startup to enterprise with flexible scaling options</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    
        <!-- Why Choose Us Section -->
        <section class="why-choose-us">
            <div class="container">
                <h2 class="section-title">Why Choose Us</h2>
                <div class="why-content">
                    <p>We've helped thousands of businesses transform their operations and achieve remarkable growth. Our platform combines cutting-edge technology with intuitive design, making complex processes simple and efficient.</p>
                    <p>With over 99.9% uptime, enterprise-grade security, and a team of dedicated experts, we're not just a service provider – we're your technology partner committed to your success.</p>
                    <p>Join industry leaders who trust us to power their most critical operations and drive innovation across their organizations.</p>
                </div>
            </div>
        </section>
    
        <!-- CTA Section -->
        <section class="cta-section">
            <div class="container">
                <div class="cta-content">
                    <h2 class="section-title">Ready to Get Started?</h2>
                    <p style="font-size: 1.1rem; color: #333; margin-bottom: 2rem;">Experience the power of our platform with a personalized demo tailored to your business needs</p>
                    <a href="#" class="cta-button">Request a Demo</a>
                </div>
            </div>
        </section>
    
        <!-- Contact Section -->
        <section class="contact">
            <div class="container">
                <h2 class="section-title">Get in Touch</h2>
                <div class="contact-grid">
                    <div class="contact-item">
                        <h4>Mennatullah Tarek </h4>
                        <a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a>
                    </div>
                    <div class="contact-item">
                        <h4>Israa Abdelghany</h4>
                        <a href="mailto:israaabdelghany9@gmail.com">israaabdelghany9@gmail.com</a>
                    </div>
                    <div class="contact-item">
                        <h4>Nagwa Mohamed </h4>
                        <a href="mailto:nagwammatia919@gmail.com">nagwammatia919@gmail.com</a>
                    </div>
                    <div class="contact-item">
                        <h4>Mohamed Salama</h4>
                        <a href="mailto:mohamedsalama152019@gmail.com">mohamedsalama152019@gmail.com</a>
                    </div>
                </div>
            </div>
        </section>
    </body>
    </html>]
    """
    st.markdown(landing_page_html, unsafe_allow_html=True)
