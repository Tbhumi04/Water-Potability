# 💧 Water Potability Prediction App (Enhanced UI)
# =================================================
import streamlit as st
import numpy as np
import pickle
import time

# -----------------------------------------
# 1️⃣ Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Water Potability Predictor 💧",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# 2️⃣ Custom CSS for animations and styling
# -----------------------------------------
st.markdown("""
<style>
    /* Main title animation */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card hover effect */
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(59, 130, 246, 0.3);
        }
        50% {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
        }
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Scale in animation */
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Main container styling */
    .main-header {
        animation: fadeInDown 0.8s ease-out;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        animation: fadeIn 1s ease-out 0.3s both;
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result card styling */
    .result-card {
        animation: scaleIn 0.5s ease-out;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border: 2px solid #4ade80;
    }
    
    .unsafe-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 2px solid #f87171;
    }
    
    /* Confidence meter */
    .confidence-meter {
        margin: 1rem auto;
        max-width: 300px;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: fadeIn 0.8s ease-out;
        color: #1e40af;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
    }
    
    .info-box strong {
        color: #1e3a8a;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    /* Parameter cards */
    .param-info {
        font-size: 0.85rem;
        color: #6b7280;
        font-style: italic;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Loading animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f4f6;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #9ca3af;
        animation: fadeIn 1.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 3️⃣ Load the trained model and scaler
# -----------------------------------------
try:
    with open("rf_model.pkl", "rb") as file:
        model, scaler = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please ensure 'rf_model.pkl' is in the same directory.")
    st.stop()

# -----------------------------------------
# 4️⃣ App Header
# -----------------------------------------
st.markdown('<h1 class="main-header">💧 Water Potability Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze water quality parameters to determine if water is safe to drink</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>ℹ️ How it works:</strong> Enter the water quality parameters in the sidebar, 
    then click the predict button to get an instant assessment of water potability.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# 5️⃣ Sidebar - Collect input data
# -----------------------------------------
st.sidebar.markdown("### 🔬 Water Quality Parameters")
st.sidebar.markdown("---")

# Parameter descriptions
param_info = {
    "pH": "Acidity/alkalinity level (7 = neutral)",
    "Hardness": "Mineral content in water",
    "Solids": "Total dissolved solids",
    "Chloramines": "Disinfectant level",
    "Sulfate": "Sulfate concentration",
    "Conductivity": "Electrical conductivity",
    "Organic Carbon": "Organic matter content",
    "Trihalomethanes": "Disinfection byproducts",
    "Turbidity": "Water cloudiness"
}

ph = st.sidebar.number_input("🧪 pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, 
                              help=param_info["pH"])
st.sidebar.markdown('<p class="param-info">Normal range: 6.5 - 8.5</p>', unsafe_allow_html=True)

hardness = st.sidebar.number_input("💎 Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0, step=1.0,
                                   help=param_info["Hardness"])
st.sidebar.markdown('<p class="param-info">Soft: <75, Hard: >150</p>', unsafe_allow_html=True)

solids = st.sidebar.number_input("⚪ Solids (ppm)", min_value=0.0, max_value=60000.0, value=20000.0, step=100.0,
                                 help=param_info["Solids"])
st.sidebar.markdown('<p class="param-info">WHO limit: <1000 ppm</p>', unsafe_allow_html=True)

chloramines = st.sidebar.number_input("🧬 Chloramines (ppm)", min_value=0.0, max_value=10.0, value=7.0, step=0.1,
                                      help=param_info["Chloramines"])
st.sidebar.markdown('<p class="param-info">Safe level: <4 ppm</p>', unsafe_allow_html=True)

sulfate = st.sidebar.number_input("⚗️ Sulfate (mg/L)", min_value=0.0, max_value=600.0, value=330.0, step=1.0,
                                  help=param_info["Sulfate"])
st.sidebar.markdown('<p class="param-info">WHO limit: <500 mg/L</p>', unsafe_allow_html=True)

conductivity = st.sidebar.number_input("⚡ Conductivity (μS/cm)", min_value=0.0, max_value=800.0, value=400.0, step=1.0,
                                       help=param_info["Conductivity"])
st.sidebar.markdown('<p class="param-info">Pure water: 0.5-3 μS/cm</p>', unsafe_allow_html=True)

organic_carbon = st.sidebar.number_input("🌿 Organic Carbon (ppm)", min_value=0.0, max_value=30.0, value=10.0, step=0.1,
                                         help=param_info["Organic Carbon"])
st.sidebar.markdown('<p class="param-info">Lower is better</p>', unsafe_allow_html=True)

trihalomethanes = st.sidebar.number_input("☢️ Trihalomethanes (μg/L)", min_value=0.0, max_value=120.0, value=60.0, step=1.0,
                                          help=param_info["Trihalomethanes"])
st.sidebar.markdown('<p class="param-info">EPA limit: <80 μg/L</p>', unsafe_allow_html=True)

turbidity = st.sidebar.number_input("🌫️ Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.0, step=0.1,
                                    help=param_info["Turbidity"])
st.sidebar.markdown('<p class="param-info">WHO limit: <5 NTU</p>', unsafe_allow_html=True)

# Combine user inputs
input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

# -----------------------------------------
# 6️⃣ Scale the input
# -----------------------------------------
input_scaled = scaler.transform(input_data)

# -----------------------------------------
# 7️⃣ Prediction Section
# -----------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Analyze Water Quality", use_container_width=True)

if predict_button:
    # Add loading animation
    with st.spinner(''):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Make prediction
        pred = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0]
        prob_potable = prob[1]
        prob_not_potable = prob[0]
        
        progress_bar.empty()
    
    # Display results
    st.markdown("### 📊 Analysis Results")
    
    if pred[0] == 1:
        st.markdown(f"""
        <div class="result-card safe-card">
            <h2>✅ POTABLE WATER</h2>
            <p style="font-size: 1.2rem; color: #166534; margin-top: 1rem;">
                This water is <strong>safe to drink</strong> based on the provided parameters.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.success(f"**Confidence Level:** {prob_potable:.1%}")
    else:
        st.markdown(f"""
        <div class="result-card unsafe-card">
            <h2>❌ NOT POTABLE</h2>
            <p style="font-size: 1.2rem; color: #991b1b; margin-top: 1rem;">
                This water is <strong>unsafe to drink</strong>. Further treatment required.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"**Confidence Level:** {prob_not_potable:.1%}")
    
    # Confidence breakdown
    st.markdown("### 📈 Confidence Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Safe to Drink", value=f"{prob_potable:.1%}", 
                 delta=f"{prob_potable - 0.5:.1%}" if prob_potable > 0.5 else f"{prob_potable - 0.5:.1%}")
    
    with col2:
        st.metric(label="Unsafe to Drink", value=f"{prob_not_potable:.1%}",
                 delta=f"{prob_not_potable - 0.5:.1%}" if prob_not_potable > 0.5 else f"{prob_not_potable - 0.5:.1%}")
    
    # Progress bars for visual representation
    st.markdown("**Probability Distribution:**")
    st.progress(prob_potable, text=f"Potable: {prob_potable:.1%}")
    st.progress(prob_not_potable, text=f"Not Potable: {prob_not_potable:.1%}")

# -----------------------------------------
# 8️⃣ Information Section
# -----------------------------------------
with st.expander("📚 About Water Quality Parameters"):
    st.markdown("""
    **pH Level:** Measures acidity/alkalinity. Safe drinking water typically has pH between 6.5-8.5.
    
    **Hardness:** Indicates mineral content, primarily calcium and magnesium.
    
    **Solids:** Total dissolved solids (TDS) measure all organic and inorganic substances.
    
    **Chloramines:** Disinfectant used in water treatment, should be kept at safe levels.
    
    **Sulfate:** Naturally occurring mineral, excessive amounts can cause digestive issues.
    
    **Conductivity:** Measures water's ability to conduct electricity, indicating ion concentration.
    
    **Organic Carbon:** Indicates organic matter, which can affect taste and support bacterial growth.
    
    **Trihalomethanes:** Byproducts of chlorination, potential health concern at high levels.
    
    **Turbidity:** Measure of water clarity, high values indicate suspended particles.
    """)

# -----------------------------------------
# 9️⃣ Footer
# -----------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Made with ❤️ using Streamlit & Random Forest Classifier</p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem;">
        🔬 This tool provides predictions based on machine learning. 
        Always consult certified water quality experts for critical decisions.
    </p>
</div>
""", unsafe_allow_html=True)