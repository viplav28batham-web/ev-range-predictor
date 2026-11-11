import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Title & Configuration
st.set_page_config(page_title="EV Range Predictor", layout="wide")

st.title("⚡ Real-World EV Range & Energy Consumption Predictor")
st.write("Input vehicle parameters and environmental factors to estimate real-world range.")

# 1. Load Trained Artifacts
@st.cache_resource
def load_artifacts():
    model_energy = joblib.load('model_energy.joblib')
    model_range = joblib.load('model_range.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return model_energy, model_range, feature_columns

model_energy, model_range, feature_columns = load_artifacts()

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
if st.button("Predict Range"):
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
    pred_energy = model_energy.predict(input_df)[0]
    pred_range = model_range.predict(input_df)[0]

    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric(label="Estimated Real-World Range", value=f"{pred_range:.1f} km")
    col2.metric(label="Energy Consumption", value=f"{pred_energy:.3f} kWh/km")
