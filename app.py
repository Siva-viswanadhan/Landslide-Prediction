import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import joblib
import os
import traceback
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Landslide Impact Prediction System", layout="wide")

# -------------------------------------------------
# LOAD MODELS (prefer joblib from models_joblib)
# -------------------------------------------------
BASE_DIR = Path(__file__).parent  # folder where app.py is
MODEL_DIR = BASE_DIR / "models_joblib"

def load_joblib_model(name: str):
    candidates = [
        MODEL_DIR / f"{name}.joblib",
        MODEL_DIR / f"{name}.pkl",
        BASE_DIR / "models" / f"{name}.pkl",
    ]
    last_exc = None
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                last_exc = e
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    files = os.listdir(MODEL_DIR)
    raise RuntimeError(f"Could not load model '{name}'. Tried: {candidates}. "
                       f"Files in {MODEL_DIR}: {files}. Last error: {last_exc}")

try:
    model1 = load_joblib_model("landslide_type_model")
    model2 = load_joblib_model("fatality_model")
    model3 = load_joblib_model("rescue_response_model")
except Exception as e:
    st.error("Failed to load ML models. Make sure 'models_joblib' contains .joblib (preferred) or .pkl files.")
    st.markdown(f"- Expected model directory: `{MODEL_DIR}`")
    if MODEL_DIR.exists():
        st.markdown(f"- Files found: {os.listdir(MODEL_DIR)}")
    st.markdown("Loader error:")
    st.code(traceback.format_exc())
    st.stop()

# -------------------------------------------------
# SESSION STATE INITIALIZATION FOR MANUAL INPUTS
# -------------------------------------------------
if "manual_inputs" not in st.session_state:
    st.session_state.manual_inputs = {
        "rainfall": 100.0,
        "soil_moisture": 0.5,
        "slope_angle": 20.0,
        "vegetation_density": 0.4,
        "altitude_m": 300.0,
        "distance_to_river": 1.0,
        "population_density": 300.0,
        "infrastructure_quality": "Low",
        "season": "Summer",
        "day_night": "Day"
    }

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Automatic Random Scenario",
        "Manual Prediction"
    ]
)

# =====================================================================
# PAGE 1 — HOME
# =====================================================================
if page == "Home":
    st.title("Landslide Impact Prediction System")

    st.markdown("""
    This system predicts:

    1. Landslide Type  
    2. Fatality Level  
    3. Rescue Response Level  

    It includes:
    - Automatic random scenario generation  
    - Manual prediction with adjustable inputs  
    """)

# =====================================================================
# PAGE 2 — AUTOMATIC RANDOM SCENARIO
# =====================================================================
elif page == "Automatic Random Scenario":

    st.title("Automatic Random Scenario and Prediction")

    st.markdown("""
    This feature generates random environmental conditions and produces:
    - Landslide Type  
    - Fatality Level  
    - Rescue Response Level  

    The generated values are also saved so you can adjust them in the Manual Prediction page.
    """)

    if st.button("Generate Random Scenario and Predict"):
        data = {
            "rainfall_mm": float(np.random.uniform(0, 500)),
            "soil_moisture": float(np.random.uniform(0.2, 1.0)),
            "slope_angle": float(np.random.uniform(5, 60)),
            "vegetation_density": float(np.random.uniform(0.1, 0.9)),
            "distance_to_river_km": float(np.random.uniform(0.1, 5)),
            "population_density": float(np.random.uniform(50, 2000)),
            "altitude_m": float(np.random.uniform(200, 3500)),

            # UPDATED HERE ↓↓↓
            "infrastructure_quality": np.random.choice(["Low", "Medium", "High"]),
            "season": np.random.choice(["Summer", "Winter", "Monsoon"]),
            # UPDATED ↑↑↑

            "day_night": np.random.choice(["Day", "Night"]),
        }

        st.success("Random scenario generated:")
        st.json(data)

        st.session_state.manual_inputs = {
            "rainfall": data["rainfall_mm"],
            "soil_moisture": data["soil_moisture"],
            "slope_angle": data["slope_angle"],
            "vegetation_density": data["vegetation_density"],
            "altitude_m": data["altitude_m"],
            "distance_to_river": data["distance_to_river_km"],
            "population_density": data["population_density"],
            "infrastructure_quality": data["infrastructure_quality"],
            "season": data["season"],
            "day_night": data["day_night"],
        }

        # MODEL 1
        m1 = {
            "rainfall_mm": data["rainfall_mm"],
            "soil_moisture": data["soil_moisture"],
            "slope_angle": data["slope_angle"],
            "vegetation_density": data["vegetation_density"],
            "distance_to_river_km": data["distance_to_river_km"],
            "altitude_m": data["altitude_m"],
            "infrastructure_quality_Low": 1 if data["infrastructure_quality"] == "Low" else 0,
            "infrastructure_quality_Medium": 1 if data["infrastructure_quality"] == "Medium" else 0,
            "season_Summer": 1 if data["season"] == "Summer" else 0,
            "season_Winter": 1 if data["season"] == "Winter" else 0,
        }

        df_m1 = pd.DataFrame([m1])[model1.feature_names_in_]
        landslide = model1.predict(df_m1)[0]
        st.subheader(f"Landslide Type: {landslide}")

        ls_no = 1 if landslide == "No Landslide" else 0
        ls_rock = 1 if landslide == "Rockfall" else 0

        # MODEL 2
        m2 = {
            **m1,
            "population_density": data["population_density"],
            "day_night_Night": 1 if data["day_night"] == "Night" else 0,
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
        }

        df_m2 = pd.DataFrame([m2])[model2.feature_names_in_]
        fatality = model2.predict(df_m2)[0]
        st.subheader(f"Fatality Level: {fatality}")

        death_estimate = {
            "Low": "0–2 possible deaths",
            "Moderate": "3–10 possible deaths",
            "High": "10–50 possible deaths",
            "Severe": "50+ possible deaths"
        }
        st.warning(f"If no rescue response is taken, estimated casualties: {death_estimate.get(fatality)}")

        fat_mod = 1 if fatality == "Moderate" else 0
        fat_sev = 1 if fatality == "Severe" else 0

        # MODEL 3
        m3 = {
            "slope_angle": data["slope_angle"],
            "vegetation_density": data["vegetation_density"],
            "distance_to_river_km": data["distance_to_river_km"],
            "population_density": data["population_density"],
            "altitude_m": data["altitude_m"],
            "infrastructure_quality_Low": m1["infrastructure_quality_Low"],
            "infrastructure_quality_Medium": m1["infrastructure_quality_Medium"],
            "day_night_Night": m2["day_night_Night"],
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
            "fatality_level_Moderate": fat_mod,
            "fatality_level_Severe": fat_sev,
        }

        df_m3 = pd.DataFrame([m3])[model3.feature_names_in_]
        rescue = model3.predict(df_m3)[0]
        st.subheader(f"Recommended Rescue Response: {rescue}")

# =====================================================================
# PAGE 3 — MANUAL PREDICTION
# =====================================================================
elif page == "Manual Prediction":

    st.title("Manual Prediction")

    st.markdown("The values below can be edited. Predictions will be updated based on your inputs.")

    mi = st.session_state.manual_inputs

    col1, col2 = st.columns(2)

    with col1:
        rainfall = st.number_input("Rainfall (mm)", value=float(mi["rainfall"]))
        soil_moisture = st.number_input("Soil Moisture (0–1)", value=float(mi["soil_moisture"]))
        slope_angle = st.number_input("Slope Angle (degrees)", value=float(mi["slope_angle"]))
        vegetation_density = st.number_input("Vegetation Density (0–1)", value=float(mi["vegetation_density"]))
        altitude_m = st.number_input("Altitude (m)", value=float(mi["altitude_m"]))

    with col2:
        distance_to_river = st.number_input("Distance to River (km)", value=float(mi["distance_to_river"]))
        population_density = st.number_input("Population Density", value=float(mi["population_density"]))

        # UPDATED HERE ↓↓↓
        infrastructure_quality = st.selectbox(
            "Infrastructure Quality",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(mi["infrastructure_quality"])
        )

        season = st.selectbox(
            "Season",
            ["Summer", "Winter", "Monsoon"],
            index=["Summer", "Winter", "Monsoon"].index(mi["season"])
        )
        # UPDATED ↑↑↑

        day_night = st.selectbox(
            "Day or Night", ["Day", "Night"],
            index=["Day", "Night"].index(mi["day_night"])
        )

    st.session_state.manual_inputs = {
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "slope_angle": slope_angle,
        "vegetation_density": vegetation_density,
        "altitude_m": altitude_m,
        "distance_to_river": distance_to_river,
        "population_density": population_density,
        "infrastructure_quality": infrastructure_quality,
        "season": season,
        "day_night": day_night,
    }

    if st.button("Predict"):
        # MODEL 1
        m1 = {
            "rainfall_mm": rainfall,
            "soil_moisture": soil_moisture,
            "slope_angle": slope_angle,
            "vegetation_density": vegetation_density,
            "distance_to_river_km": distance_to_river,
            "altitude_m": altitude_m,
            "infrastructure_quality_Low": 1 if infrastructure_quality == "Low" else 0,
            "infrastructure_quality_Medium": 1 if infrastructure_quality == "Medium" else 0,
            "season_Summer": 1 if season == "Summer" else 0,
            "season_Winter": 1 if season == "Winter" else 0,
        }

        df_m1 = pd.DataFrame([m1])[model1.feature_names_in_]
        landslide = model1.predict(df_m1)[0]
        st.success(f"Landslide Type: {landslide}")

        ls_no = 1 if landslide == "No Landslide" else 0
        ls_rock = 1 if landslide == "Rockfall" else 0

        # MODEL 2
        m2 = {
            **m1,
            "population_density": population_density,
            "day_night_Night": 1 if day_night == "Night" else 0,
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
        }

        df_m2 = pd.DataFrame([m2])[model2.feature_names_in_]
        fatality = model2.predict(df_m2)[0]
        st.warning(f"Fatality Level: {fatality}")

        death_estimate = {
            "Low": "0–2 possible deaths",
            "Moderate": "3–10 possible deaths",
            "High": "10–50 possible deaths",
            "Severe": "50+ possible deaths"
        }
        st.error(f"If no rescue response is taken, estimated casualties: {death_estimate.get(fatality)}")

        fat_mod = 1 if fatality == "Moderate" else 0
        fat_sev = 1 if fatality == "Severe" else 0

        # MODEL 3
        m3 = {
            "slope_angle": slope_angle,
            "vegetation_density": vegetation_density,
            "distance_to_river_km": distance_to_river,
            "population_density": population_density,
            "altitude_m": altitude_m,
            "infrastructure_quality_Low": m1["infrastructure_quality_Low"],
            "infrastructure_quality_Medium": m1["infrastructure_quality_Medium"],
            "day_night_Night": m2["day_night_Night"],
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
            "fatality_level_Moderate": fat_mod,
            "fatality_level_Severe": fat_sev,
        }

        df_m3 = pd.DataFrame([m3])[model3.feature_names_in_]
        rescue = model3.predict(df_m3)[0]
        st.info(f"Recommended Rescue Response Level: {rescue}")
