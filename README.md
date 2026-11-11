# ⚡ Real-World EV Range & Energy Consumption Predictor

An end-to-end Data Science and Machine Learning project designed to predict the **real-world driving range (km)** and **energy consumption (kWh/km)** of Electric Vehicles (EVs) under various environmental, vehicular, and driving conditions in India.

🚀 **[Live Web App](https://ev-range-predictor-h7n5gknpsbbjckciwijsnb.streamlit.app/)**

---

## 📌 Project Overview

Certified EV ranges (such as ARAI or WLTP) are often measured under idealized laboratory conditions and rarely reflect real-world performance. Factors such as aggressive driving, high ambient temperatures, heavy AC usage, traffic conditions, passenger loads, and battery degradation significantly impact range.

This project builds, benchmarks, and deploys machine learning regression models to provide accurate, real-world range estimates based on user-inputted driving parameters.

---

## 🛠️ Key Features

- **Dual-Target Prediction:** Simultaneously predicts real-world range ($y_1$) and energy consumption ($y_2$).
- **Multi-Factor Input Support:** Accounts for battery capacity, motor power, vehicle weight, ambient temperature, AC usage, driving speed, passenger count, battery age, and regional road conditions.
- **Model Benchmarking:** Evaluates multiple regression architectures (Linear Regression, Random Forest, Gradient Boosting, XGBoost).
- **Interactive Web App:** Deployed on Streamlit Cloud for instant model inference via a web browser.

---

## 📊 Dataset & Preprocessing

- **Dataset Size:** 7,291 entries and 25 feature columns (`real_india_ev_range_dataset.csv`).
- **Data Cleaning & Auditing:** Checked for missing values, duplicates, and physical sanity bounds.
- **Feature Engineering:** One-Hot Encoded categorical variables (`make`, `city_road_type`) and scaled features using `StandardScaler` to prevent data leakage.

---

## 📈 Model Performance & Results

Models were evaluated on a 20% holdout test set using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**:

| Target Variable | Best Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Energy Consumption (kWh/km)** | XGBoost Regressor | 0.0299 | 0.0472 | **0.9776** |
| **Real-World Range (km)** | XGBoost Regressor | 18.4106 | 26.3335 | **0.9662** |

---

## 💻 Tech Stack

- **Language:** Python 3.x
- **Data Manipulation & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Model Serialization:** `joblib`
- **Web Framework & Deployment:** Streamlit, Streamlit Community Cloud

---

## 🚀 Local Installation & Execution

To run this application locally on your machine:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/viplav28batham-web/ev-range-predictor.git](https://github.com/viplav28batham-web/ev-range-predictor.git)
   cd ev-range-predictor
