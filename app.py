from pathlib import Path

import gdown
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.write("Enter the housing details to estimate the median house value.")

MODEL_ID = "https://drive.google.com/file/d/1bE_vT55cMRa6L6byzkhT8F_e_wE_BcZ1/view?usp=sharing"
SCALER_ID = "https://drive.google.com/file/d/1jzxXuMwemOl8PCCzjkFUR9luEH2bBOCN/view?usp=sharing"

@st.cache_resource
def load_assets():
    model_path = Path("house_price_model.pkl")
    scaler_path = Path("house_scaler.pkl")

    if not model_path.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_ID}",
            str(model_path),
            quiet=False,
        )

    if not scaler_path.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={SCALER_ID}",
            str(scaler_path),
            quiet=False,
        )

    return joblib.load(model_path), joblib.load(scaler_path)

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(
        "Could not download or load the model files. "
        "Check that the Google Drive links are shared with anyone who has the link."
    )
    st.exception(e)
    st.stop()

with st.form("house_details"):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.5f")
        latitude = st.number_input("Latitude", value=37.88, format="%.5f")
        housing_median_age = st.number_input("Housing median age", min_value=0, value=41)
        total_rooms = st.number_input("Total rooms", min_value=1, value=880)
        total_bedrooms = st.number_input("Total bedrooms", min_value=0, value=129)
        population = st.number_input("Population", min_value=0, value=322)

    with col2:
        households = st.number_input("Households", min_value=1, value=126)
        median_income = st.number_input("Median income (in tens of thousands)", min_value=0.0, value=8.3)
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        )

    submitted = st.form_submit_button("Estimate house value")

if submitted:
    if total_rooms <= 0 or households <= 0:
        st.error("Total rooms and households must be greater than zero.")
    else:
        # Match the notebook's feature engineering.
        rooms_per_household = total_rooms / households
        bedrooms_per_room = total_bedrooms / total_rooms
        population_per_household = population / households

        # Match LabelEncoder's alphabetical category order from the notebook.
        ocean_codes = {
            "<1H OCEAN": 0,
            "INLAND": 1,
            "ISLAND": 2,
            "NEAR BAY": 3,
            "NEAR OCEAN": 4,
        }

        features = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household": population_per_household,
            "ocean_proximity": ocean_codes[ocean_proximity],
        }])

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        st.success(f"Estimated median house value: ${prediction:,.0f}")
