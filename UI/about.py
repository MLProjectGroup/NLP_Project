import streamlit as st

def app():
    theme = {
        "primary": "#017691",
        "primary_dark": "#015a71",
        "secondary": "#333",
        "accent": "#dce3e4",
        "background": "#dce3e4",
        "text": "#222222",
        "text_light": "#333",
        "highlight": "#abc2c7",
        "success": "#017691",
        "gradient": "linear-gradient(135deg, #017691 0%, #015a71 100%)"
    }
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        html, body, .main {{
            height: 100%;
            background-color: {theme['background']};
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        .stApp {{
            background: {theme['background']};
        }}
        
        .fade-in {{
            animation: fadeIn 1.2s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .slide-up {{
            animation: slideUp 0.8s ease-out forwards;
            opacity: 0;
            transform: translateY(50px);
        }}
        
        @keyframes slideUp {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        /* Header */
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            border-bottom: 1px solid #e2e8f0;
            padding: 15px 0;
        }}
        
        .nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .logo {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {theme['primary']};
            text-decoration: none;
        }}
        
        .nav-links {{
            display: flex;
            gap: 30px;
            align-items: center;
        }}
        
        .nav-links a {{
            text-decoration: none;
            color: {theme['text']};
            font-weight: 500;
            transition: color 0.3s ease;
        }}
        
        .nav-links a:hover {{
            color: {theme['primary']};
        }}
        
        /* Hero Section */
        .hero {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['primary_dark']} 100%);
            color: white;
            padding: 140px 0 100px 0;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="white" opacity="0.1"><polygon points="0,0 1000,100 1000,0"/></svg>');
            background-size: cover;
        }}
        
        .hero-content {{
            position: relative;
            z-index: 2;
        }}
        
        .hero-title {{
            font-size: 3.5rem;
            font-weight: 900;
            margin-bottom: 20px;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }}
        
        .hero-subtitle {{
            font-size: 1.3rem;
            margin-bottom: 40px;
            opacity: 0.9;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            font-weight: 400;
        }}
        
        .hero-buttons {{
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 40px;
        }}
        
        .btn-primary {{
            background: white;
            color: {theme['primary']};
            border: none;
            border-radius: 50px;
            padding: 16px 32px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }}
        
        .btn-secondary {{
            background: transparent;
            color: white;
            border: 2px solid white;
            border-radius: 50px;
            padding: 14px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}
        
        .btn-secondary:hover {{
            background: white;
            color: {theme['primary']};
            transform: translateY(-2px);
        }}
        
        /* Features Section */
        .features-section {{
            padding: 100px 0;
            background: white;
        }}
        
        .section-title {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            color: {theme['text']};
            margin-bottom: 20px;
        }}
        
        .section-subtitle {{
            text-align: center;
            font-size: 1.2rem;
            color: {theme['text_light']};
            margin-bottom: 60px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 40px;
            margin-top: 60px;
        }}
        
        .feature-card {{
            background: white;
            padding: 40px 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 4px 30px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 40px rgba(0,0,0,0.12);
        }}
        
        .feature-icon {{
            width: 80px;
            height: 80px;
            background: {theme['highlight']};
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px auto;
            font-size: 2rem;
        }}
        
        .feature-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {theme['text']};
            margin-bottom: 15px;
        }}
        
        .feature-description {{
            color: {theme['text_light']};
            line-height: 1.6;
            font-size: 1rem;
        }}
        
        /* Why Choose Section */
        .why-choose-section {{
            padding: 100px 0;
            background: {theme['accent']};
        }}
        
        .why-choose-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 80px;
            align-items: center;
        }}
        
        .why-choose-text h3 {{
            font-size: 2.5rem;
            font-weight: 800;
            color: {theme['text']};
            margin-bottom: 25px;
            line-height: 1.2;
        }}
        
        .why-choose-text p {{
            font-size: 1.1rem;
            color: {theme['text_light']};
            line-height: 1.7;
            margin-bottom: 30px;
        }}
        
        .stats {{
            display: flex;
            gap: 40px;
            margin-top: 40px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2.5rem;
            font-weight: 900;
            color: {theme['primary']};
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: {theme['text_light']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }}
        
        .benefits-list {{
            list-style: none;
            padding: 0;
        }}
        
        .benefits-list li {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            font-size: 1.1rem;
            color: {theme['text']};
        }}
        
        .benefits-list li::before {{
            content: "✓";
            background: {theme['success']};
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
            font-size: 0.9rem;
            flex-shrink: 0;
        }}
        
        /* CTA Section */
        .cta-section {{
            background: {theme['primary']};
            color: white;
            padding: 80px 0;
            text-align: center;
        }}
        
        .cta-title {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 20px;
        }}
        
        .cta-subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 40px;
        }}
        
        /* Footer */
        .footer {{
            background: #1e293b;
            color: white;
            padding: 60px 0 30px 0;
        }}
        
        .footer-content {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 60px;
            margin-bottom: 40px;
        }}
        
        .footer-brand {{
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 20px;
        }}
        
        .footer-description {{
            color: #94a3b8;
            line-height: 1.6;
            margin-bottom: 30px;
        }}
        
        .footer-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        
        .footer-links {{
            list-style: none;
            padding: 0;
        }}
        
        .footer-links li {{
            margin-bottom: 10px;
        }}
        
        .footer-links a {{
            color: #94a3b8;
            text-decoration: none;
            transition: color 0.3s ease;
        }}
        
        .footer-links a:hover {{
            color: white;
        }}
        
        .footer-bottom {{
            border-top: 1px solid #374151;
            padding-top: 30px;
            text-align: center;
            color: #94a3b8;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .hero-title {{
                font-size: 2.5rem;
            }}
            
            .hero-buttons {{
                flex-direction: column;
                align-items: center;
            }}
            
            .features-grid {{
                grid-template-columns: 1fr;
            }}
            
            .why-choose-content {{
                grid-template-columns: 1fr;
                gap: 40px;
            }}
            
            .stats {{
                justify-content: center;
            }}
            
            .footer-content {{
                grid-template-columns: 1fr;
                gap: 40px;
            }}
            
            .nav-links {{
                display: none;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div class="header">
        <div class="nav">
            <a href="#" class="logo">Smart Recruiter</a>
            <div class="nav-links">
                <a href="#features">Features</a>
                <a href="#about">About</a>
                <a href="#contact">Contact</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero fade-in">
        <div class="container">
            <div class="hero-content">
                <h1 class="hero-title">Revolutionize Your Hiring Process</h1>
                <p class="hero-subtitle">AI-powered recruitment platform that helps you find, evaluate, and hire the best talent faster than ever before.</p>
                <div class="hero-buttons">
                    <button class="btn-primary" onclick="window.location.href='mailto:menatarek04@gmail.com'">Start Free Trial</button>
                    <button class="btn-secondary" onclick="window.location.href='#features'">Learn More</button>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
    <div class="features-section" id="features">
        <div class="container">
            <h2 class="section-title">Powerful Features for Modern Recruiters</h2>
            <p class="section-subtitle">Everything you need to streamline your recruitment process and make data-driven hiring decisions.</p>
            
            <div class="features-grid">
                <div class="feature-card slide-up">
                    <div class="feature-icon">🎯</div>
                    <h3 class="feature-title">AI-Powered Matching</h3>
                    <p class="feature-description">Advanced algorithms analyze resumes and job requirements to find the perfect candidates automatically.</p>
                </div>
                
                <div class="feature-card slide-up">
                    <div class="feature-icon">🤖</div>
                    <h3 class="feature-title">Intelligent Assistant</h3>
                    <p class="feature-description">Your personal AI recruiter guides you through every step of the hiring process with smart recommendations.</p>
                </div>
                
                <div class="feature-card slide-up">
                    <div class="feature-icon">📊</div>
                    <h3 class="feature-title">Analytics & Insights</h3>
                    <p class="feature-description">Comprehensive reports and analytics help you optimize your recruitment strategy and track performance.</p>
                </div>
                
                <div class="feature-card slide-up">
                    <div class="feature-icon">⚡</div>
                    <h3 class="feature-title">Workflow Automation</h3>
                    <p class="feature-description">Automate repetitive tasks like screening, scheduling, and follow-ups to focus on what matters most.</p>
                </div>
                
                <div class="feature-card slide-up">
                    <div class="feature-icon">🔒</div>
                    <h3 class="feature-title">Enterprise Security</h3>
                    <p class="feature-description">Bank-level security ensures your sensitive recruitment data is always protected and compliant.</p>
                </div>
                
                <div class="feature-card slide-up">
                    <div class="feature-icon">🌐</div>
                    <h3 class="feature-title">Global Integration</h3>
                    <p class="feature-description">Seamlessly connects with popular job boards, ATS systems, and communication tools you already use.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why Choose Section
    st.markdown("""
    <div class="why-choose-section" id="about">
        <div class="container">
            <div class="why-choose-content">
                <div class="why-choose-text">
                    <h3>Why Choose Smart Recruiter Assistant?</h3>
                    <p>We're not just another recruitment tool. We're your strategic partner in building exceptional teams. Our AI-powered platform combines cutting-edge technology with human insight to deliver results that matter.</p>
                    
                    <ul class="benefits-list">
                        <li>Reduce time-to-hire by up to 70%</li>
                        <li>Improve candidate quality with AI screening</li>
                        <li>Scale your recruitment without growing your team</li>
                        <li>Make data-driven hiring decisions</li>
                        <li>Eliminate unconscious bias in screening</li>
                    </ul>
                    
                    <div class="stats">
                        <div class="stat">
                            <span class="stat-number">10K+</span>
                            <span class="stat-label">Candidates Screened</span>
                        </div>
                        <div class="stat">
                            <span class="stat-number">500+</span>
                            <span class="stat-label">Companies Trust Us</span>
                        </div>
                        <div class="stat">
                            <span class="stat-number">70%</span>
                            <span class="stat-label">Faster Hiring</span>
                        </div>
                    </div>
                </div>
                
                <div class="why-choose-visual">
                    <div class="feature-card">
                        <div class="feature-icon">🚀</div>
                        <h3 class="feature-title">Built for Scale</h3>
                        <p class="feature-description">Whether you're hiring 10 or 10,000 people, our platform grows with your needs.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Section
    st.markdown("""
    <div class="cta-section">
        <div class="container">
            <h2 class="cta-title">Ready to Transform Your Hiring?</h2>
            <p class="cta-subtitle">Join hundreds of companies already using Smart Recruiter Assistant to build better teams.</p>
            <button class="btn-primary" onclick="window.location.href='mailto:menatarek04@gmail.com'">Get Started Today</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer" id="contact">
        <div class="container">
            <div class="footer-content">
                <div>
                    <div class="footer-brand">Smart Recruiter Assistant</div>
                    <p class="footer-description">Empowering recruiters with AI-driven insights and automation to build exceptional teams faster and smarter.</p>
                </div>
                
                <div>
                    <h4 class="footer-title">Product</h4>
                    <ul class="footer-links">
                        <li><a href="#">Features</a></li>
                        <li><a href="#">Pricing</a></li>
                        <li><a href="#">Integrations</a></li>
                        <li><a href="#">API</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="footer-title">Contact</h4>
                    <ul class="footer-links">
                        <li><a href="mailto:menatarek04@gmail.com">menatarek04@gmail.com</a></li>
                        <li><a href="mailto:israaabdelghany9@gmail.com">israaabdelghany9@gmail.com</a></li>
                        <li><a href="mailto:nagwammatia919@gmail.com">nagwammatia919@gmail.com</a></li>
                        <li><a href="mailto:mohamedsalama152019@gmail.com">mohamedsalama152019@gmail.com</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="footer-bottom">
                <p>&copy; 2025 Smart Recruiter Assistant. All rights reserved.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
