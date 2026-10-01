import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import zipfile

st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🍔", layout="wide")
st.title("🍔 Food Delivery Time Predictor")
st.write("Predict expected food delivery time using a trained Random Forest model.")

@st.cache_resource
def load_model():
    import_directory = sys._xoptions["snowflake_import_directory"]
    zip_path = os.path.join(import_directory, "delivery_model.zip")
    extract_dir = "/tmp/delivery_model"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return joblib.load(os.path.join(extract_dir, "delivery_time_random_forest.joblib"))

model = load_model()

st.header("Delivery Information")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Delivery Person Age", 18.0, 60.0, 30.0)
    rating = st.number_input("Delivery Person Rating", 1.0, 5.0, 4.5)
    vehicle_condition = st.number_input("Vehicle Condition", 0, 5, 2)

with c2:
    rest_lat = st.number_input("Restaurant Latitude", value=17.45)
    rest_long = st.number_input("Restaurant Longitude", value=78.38)
    delivery_lat = st.number_input("Delivery Latitude", value=17.44)
    delivery_long = st.number_input("Delivery Longitude", value=78.39)

with c3:
    multiple_deliveries = st.number_input("Multiple Deliveries", 0.0, 5.0, 1.0)
    weather = st.selectbox("Weather Conditions", ["Sunny","Stormy","Sandstorms","Cloudy","Fog","Windy"])
    traffic = st.selectbox("Road Traffic Density", ["Low","Medium","High","Jam"])

c4, c5, c6 = st.columns(3)
with c4:
    order_type = st.selectbox("Type of Order", ["Snack","Meal","Drinks","Buffet"])
with c5:
    vehicle_type = st.selectbox("Type of Vehicle", ["motorcycle","scooter","electric_scooter","bicycle"])
with c6:
    festival = st.selectbox("Festival", ["No","Yes"])

city = st.selectbox("City", ["Urban","Metropolitian","Semi-Urban"])

distance_km = 6371 * 2 * np.arcsin(np.sqrt(
    np.sin(np.radians(delivery_lat - rest_lat) / 2) ** 2
    + np.cos(np.radians(rest_lat)) * np.cos(np.radians(delivery_lat))
    * np.sin(np.radians(delivery_long - rest_long) / 2) ** 2
))

st.write(f"**Calculated delivery distance:** {distance_km:.2f} km")

if st.button("🚀 Predict Delivery Time"):
    input_data = pd.DataFrame({
        "AGE":[age], "RATING":[rating], "REST_LAT":[rest_lat], "REST_LONG":[rest_long],
        "DELIVERY_LAT":[delivery_lat], "DELIVERY_LONG":[delivery_long],
        "VEHICLE_CONDITION":[vehicle_condition], "MULTIPLE_DELIVERIES":[multiple_deliveries],
        "DISTANCE_KM":[distance_km], "WEATHER_CONDITIONS":[weather],
        "ROAD_TRAFFIC_DENSITY":[traffic], "TYPE_OF_ORDER":[order_type],
        "TYPE_OF_VEHICLE":[vehicle_type], "FESTIVAL":[festival], "CITY":[city]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"### Estimated Delivery Time: {prediction:.0f} minutes")

with st.expander("📊 Model Information"):
    st.write("**Problem:** Food Delivery Time Prediction")
    st.write("**Type:** Regression")
    st.write("**Model:** Random Forest Regressor")
    st.write("**Training Rows:** 36,467")
    st.write("**Test Rows:** 9,117")
    st.write("**MAE:** 3.193 minutes")
    st.write("**RMSE:** 4.065 minutes")
    st.write("**R²:** 0.813")
