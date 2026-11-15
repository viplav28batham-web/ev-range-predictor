import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep Purple Glassmorphic CSS Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0B0813;
        color: #F3F0FF;
    }

    /* Top Hero Card */
    .hero-container {
        background: linear-gradient(135deg, #16122B 0%, #1A1535 100%);
        border: 1px solid #2D2553;
        border-radius: 20px;
        padding: 40px;
        position: relative;
        overflow: hidden;
        margin-bottom: 32px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .pill-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #C4B5FD;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 12px 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #A79BCC;
        font-size: 15px;
        max-width: 580px;
        line-height: 1.6;
        margin: 0;
    }

    .icon-orb {
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, #7C3AED 0%, #4C1D95 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 44px;
        box-shadow: 0 0 35px rgba(124, 58, 237, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Form Section Containers */
    .section-card {
        background: #130F24;
        border: 1px solid #241D42;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 24px;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 1px solid #231B42;
    }

    .section-number-bar {
        width: 4px;
        height: 22px;
        background: linear-gradient(180deg, #A855F7 0%, #EC4899 100%);
        border-radius: 4px;
    }

    /* Metric Result Box */
    .result-card {
        background: linear-gradient(135deg, #1D173B 0%, #15102D 100%);
        border: 1px solid #372D6B;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .result-label { color: #9D8EC4; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .result-value { color: #F3E8FF; font-size: 34px; font-weight: 800; margin-top: 8px; text-shadow: 0 0 20px rgba(168, 85, 247, 0.4); }
    .result-sub { color: #C084FC; font-size: 13px; margin-top: 6px; }

    /* Override Streamlit Form Inputs Styling */
    div[data-baseweb="select"] > div {
        background-color: #1A1533 !important;
        border-color: #31275C !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    
    .stNumberInput input {
        background-color: #1A1533 !important;
        border: 1px solid #31275C !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7C3AED 0%, #C026D3 100%);
        color: #FFFFFF;
        border: none;
        padding: 16px 24px;
        font-size: 16px;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(192, 38, 211, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# 1. Load Trained Artifacts
@st.cache_resource
def load_artifacts():
    model_energy = joblib.load('model_energy.joblib')
    model_range = joblib.load('model_range.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return model_energy, model_range, feature_columns

try:
    model_energy, model_range, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header Card Component
hero_col1, hero_col2 = st.columns([3, 1])

with hero_col1:
    st.markdown("""
        <div class="pill-tag">✦ AI / ML PROJECT</div>
        <h1 class="hero-title">EV Performance Predictor.</h1>
        <p class="hero-subtitle">Discover insights into real-world electric vehicle performance through Machine Learning. Enter vehicle specs and trip parameters to compute exact range and energy consumption.</p>
    """, unsafe_allow_html=True)

with hero_col2:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div class="icon-orb">⚡</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Inputs Form
with st.container():
    # Section 01: Vehicle Specifications
    st.markdown("""
        <div class="section-header">
            <div class="section-number-bar"></div>
            <span>01 Vehicle Specifications</span>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=150.0, value=40.5)
        motor_power = st.number_input("Motor Power (kW)", min_value=20, max_value=600, value=105)
        vehicle_weight = st.number_input("Vehicle Weight (kg)", min_value=500, max_value=3500, value=1610)
        make_selected = st.selectbox("Manufacturer Make", options=["Tata", "MG", "Hyundai", "Mahindra", "BYD", "Smart"])

    with col2:
        certified_range = st.number_input("ARAI Certified Range (km)", min_value=100, max_value=900, value=465)
        battery_age = st.slider("Battery Age (Years)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        passengers = st.slider("Passengers", min_value=1, max_value=7, value=2)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    # Section 02: Trip & Environmental Parameters
    st.markdown("""
        <div class="section-header">
            <div class="section-number-bar"></div>
            <span>02 Trip & Environmental Conditions</span>
        </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        avg_speed = st.slider("Avg Driving Speed (km/h)", min_value=10, max_value=140, value=45)
        ambient_temp = st.slider("Ambient Temperature (°C)", min_value=-5, max_value=50, value=30)

    with col4:
        ac_usage = st.selectbox("AC Usage", options=[0, 1], format_func=lambda x: "On" if x == 1 else "Off")
        road_type_selected = st.selectbox("Road / City Type", options=["Delhi", "Mumbai", "Pune", "Highway_NH", "Expressway", "Ghat_Road"])

st.markdown("<br>", unsafe_allow_html=True)

# Submit Button & Prediction Computation
if st.button("Predict Real-World Performance ⚡"):
    # Create input dictionary with default 0s for all feature columns
    input_data = {col: 0 for col in feature_columns}

    # Fill in numerical features
    input_data['battery_capacity_kWh'] = battery_capacity
    input_data['motor_power_kW'] = motor_power
    input_data['vehicle_weight_kg'] = vehicle_weight
    input_data['ARAI_certified_range_km'] = certified_range
    input_data['avg_driving_speed_kmh'] = avg_speed
    input_data['ambient_temperature_C'] = ambient_temp
    input_data['ac_usage'] = ac_usage
    input_data['passengers'] = passengers
    input_data['battery_age_years'] = battery_age

    # One-hot encoded feature matching
    make_col = f"make_{make_selected}"
    if make_col in input_data:
        input_data[make_col] = 1

    road_col = f"city_road_type_{road_type_selected}"
    if road_col in input_data:
        input_data[road_col] = 1

    # Convert to DataFrame matching exact feature column order
    input_df = pd.DataFrame([input_data])[feature_columns]

    # Model Inference
    pred_energy = float(model_energy.predict(input_df)[0])
    pred_range = float(model_range.predict(input_df)[0])

    range_loss = certified_range - pred_range
    degradation_pct = (range_loss / certified_range) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 03: Output Results
    st.markdown("""
        <div class="section-header">
            <div class="section-number-bar"></div>
            <span>03 Predicted Performance Category</span>
        </div>
    """, unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimated Real Range</div>
                <div class="result-value">{pred_range:.1f} km</div>
                <div class="result-sub">Certified: {certified_range} km</div>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Energy Consumption</div>
                <div class="result-value">{pred_energy:.3f}</div>
                <div class="result-sub">kWh per kilometer</div>
            </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Range Deficit</div>
                <div class="result-value">-{range_loss:.1f} km</div>
                <div class="result-sub">-{degradation_pct:.1f}% vs ARAI rating</div>
            </div>
        """, unsafe_allow_html=True)
        
