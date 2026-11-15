import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Configuration
st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="wide")

# Inject Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .header-card {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
    }
    .header-title { color: #38BDF8; font-size: 26px; font-weight: 700; margin: 0; }
    .header-sub { color: #94A3B8; font-size: 14px; margin-top: 4px; }
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0EA5E9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title { color: #94A3B8; font-size: 13px; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #F8FAFC; font-size: 30px; font-weight: 700; margin-top: 4px; }
    .metric-sub { color: #38BDF8; font-size: 13px; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-card">
        <h1 class="header-title">⚡ Real-World EV Range & Energy Consumption Predictor</h1>
        <p class="header-sub">Input vehicle specifications and environmental parameters to estimate real-world range and energy consumption.</p>
    </div>
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
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. Sidebar Inputs for User Parameters
st.sidebar.header("Vehicle & Trip Parameters")

battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=150.0, value=40.5)
motor_power = st.sidebar.number_input("Motor Power (kW)", min_value=20, max_value=600, value=105)
vehicle_weight = st.sidebar.number_input("Vehicle Weight (kg)", min_value=500, max_value=3500, value=1610)
certified_range = st.sidebar.number_input("ARAI Certified Range (km)", min_value=100, max_value=900, value=465)

avg_speed = st.sidebar.slider("Avg Driving Speed (km/h)", min_value=10, max_value=140, value=45)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", min_value=-5, max_value=50, value=30)
ac_usage = st.sidebar.selectbox("AC Usage", options=[0, 1], format_func=lambda x: "On" if x == 1 else "Off")
passengers = st.sidebar.slider("Passengers", min_value=1, max_value=7, value=2)
battery_age = st.sidebar.slider("Battery Age (Years)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical Dropdowns
make_selected = st.sidebar.selectbox("Manufacturer Make", options=["Tata", "MG", "Hyundai", "Mahindra", "BYD", "Smart"])
road_type_selected = st.sidebar.selectbox("Road / City Type", options=["Delhi", "Mumbai", "Pune", "Highway_NH", "Expressway", "Ghat_Road"])

# 3. Predict Button & Logic
if st.sidebar.button("Predict Range", type="primary"):
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

    # Predict
    pred_energy = float(model_energy.predict(input_df)[0])
    pred_range = float(model_range.predict(input_df)[0])

    range_loss = certified_range - pred_range
    degradation_pct = (range_loss / certified_range) * 100

    # Display Metrics in Styled Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Real-World Range</div>
                <div class="metric-value">{pred_range:.1f} km</div>
                <div class="metric-sub">Certified: {certified_range} km</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #F59E0B;">
                <div class="metric-title">Energy Consumption</div>
                <div class="metric-value">{pred_energy:.3f}</div>
                <div class="metric-sub">kWh per kilometer</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #EF4444;">
                <div class="metric-title">Range Deficit</div>
                <div class="metric-value">-{range_loss:.1f} km</div>
                <div class="metric-sub">-{degradation_pct:.1f}% vs Certified Rating</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Retained Efficiency Bar
    retained_pct = max(0.0, min(1.0, pred_range / certified_range))
    st.write(f"**Retained Range Efficiency:** {retained_pct * 100:.1f}%")
    st.progress(retained_pct)
