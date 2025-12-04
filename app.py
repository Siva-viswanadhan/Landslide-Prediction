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
st.set_page_config(page_title="Landslide Prediction System", layout="wide")

# -------------------------------------------------
# LOAD MODELS (prefer joblib from models_joblib)
# -------------------------------------------------
BASE_DIR = Path(__file__).parent  # folder where app.py is
MODEL_DIR = BASE_DIR / "models_joblib"

def load_joblib_model(name: str):
    """
    Try .joblib first, then .pkl in models_joblib, then fallback to models/*.pkl
    """
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
    # No candidate loaded
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
    st.markdown("**Loader error:**")
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
# OPTIONAL: STUBS FOR ALERT SENDING (EMAIL/SMS)
# (Replace with real APIs like SendGrid/Twilio later)
# -------------------------------------------------
def send_email_alert(to_email: str, subject: str, message: str):
    # TODO: integrate real email API (e.g., SendGrid/SMTP)
    print(f"[SIMULATED EMAIL] To: {to_email}, Subject: {subject}, Message: {message}")


def send_sms_alert(to_phone: str, message: str):
    # TODO: integrate real SMS API (e.g., Twilio)
    print(f"[SIMULATED SMS] To: {to_phone}, Message: {message}")


# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Automatic Random Scenario",
        "Manual Prediction",
        "Real-time Rainfall & Alerts"
    ]
)

# =====================================================================
# PAGE 1 — HOME
# =====================================================================
if page == "Home":
    st.title("🏔️ AI-Based Landslide Risk & Response Prediction System")

    st.markdown("""
    This project demonstrates a **3-stage machine learning pipeline**:

    1. **Landslide Type Prediction**  
    2. **Fatality Level Prediction**  
    3. **Rescue Response Level Recommendation**  

    It includes:
    - Automatic random scenario generation (like simulated sensor/API input)  
    - Manual prediction with auto-filled values from the auto page  
    - Real-time rainfall monitoring using an external API (via FastAPI)  
    - Alert simulation via Email/SMS and safer-location suggestion.
    """)

# =====================================================================
# PAGE 2 — AUTOMATIC RANDOM SCENARIO
# =====================================================================
elif page == "Automatic Random Scenario":

    st.title("🤖 Automatic Random Scenario & Prediction")

    st.markdown("""
    This page simulates **automatic sensor/API input** by generating random but realistic values.
    It predicts:
    - 🌋 Landslide Type  
    - 💀 Fatality Level  
    - 🚑 Rescue Response Level  

    The same random values are also saved and will appear on the **Manual Prediction** page
    so you can adjust them further.
    """)

    if st.button("🎲 Generate Random Scenario & Predict"):
        # Generate synthetic data
        data = {
            "rainfall_mm": float(np.random.uniform(0, 500)),
            "soil_moisture": float(np.random.uniform(0.2, 1.0)),
            "slope_angle": float(np.random.uniform(5, 60)),
            "vegetation_density": float(np.random.uniform(0.1, 0.9)),
            "distance_to_river_km": float(np.random.uniform(0.1, 5)),
            "population_density": float(np.random.uniform(50, 2000)),
            "altitude_m": float(np.random.uniform(200, 3500)),
            "infrastructure_quality": np.random.choice(["Low", "Medium"]),
            "season": np.random.choice(["Summer", "Winter"]),
            "day_night": np.random.choice(["Day", "Night"]),
        }

        # Show scenario
        st.success("Random scenario generated:")
        st.json(data)

        # ALSO update manual_inputs so page 3 uses same values
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

        # ---------------------------
        # MODEL 1 — LANDSLIDE TYPE
        # ---------------------------
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
        st.subheader(f"🌋 Predicted Landslide Type: **{landslide}**")

        ls_no = 1 if landslide == "No Landslide" else 0
        ls_rock = 1 if landslide == "Rockfall" else 0

        # ---------------------------
        # MODEL 2 — FATALITY LEVEL
        # ---------------------------
        m2 = {
            **m1,
            "population_density": data["population_density"],
            "day_night_Night": 1 if data["day_night"] == "Night" else 0,
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
        }

        df_m2 = pd.DataFrame([m2])[model2.feature_names_in_]
        fatality = model2.predict(df_m2)[0]
        st.subheader(f"💀 Predicted Fatality Level: **{fatality}**")

        # Casualty estimate (no response)
        death_estimate = {
            "Low": "0–2 possible deaths",
            "Moderate": "3–10 possible deaths",
            "High": "10–50 possible deaths",
            "Severe": "50+ possible deaths"
        }
        st.error(
            f"🧍 If **no rescue response** is taken, estimated casualties: "
            f"**{death_estimate.get(fatality, 'N/A')}**"
        )

        fat_mod = 1 if fatality == "Moderate" else 0
        fat_sev = 1 if fatality == "Severe" else 0

        # ---------------------------
        # MODEL 3 — RESCUE RESPONSE
        # ---------------------------
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
        st.subheader(f"🚑 Recommended Rescue Response: **{rescue}**")

# =====================================================================
# PAGE 3 — MANUAL PREDICTION (AUTO-FILLED)
# =====================================================================
elif page == "Manual Prediction":

    st.title("🧪 Manual Landslide, Fatality & Rescue Prediction")

    st.markdown("""
    You can **manually edit** the values here.  
    When you generate a random scenario on the **Automatic Random Scenario** page,
    the same values will appear here automatically.
    """)

    mi = st.session_state.manual_inputs  # shortcut

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

        infrastructure_quality = st.selectbox(
            "Infrastructure Quality",
            ["Low", "Medium"],
            index=["Low", "Medium"].index(mi["infrastructure_quality"])
        )

        season = st.selectbox(
            "Season", ["Summer", "Winter"],
            index=["Summer", "Winter"].index(mi["season"])
        )

        day_night = st.selectbox(
            "Day or Night", ["Day", "Night"],
            index=["Day", "Night"].index(mi["day_night"])
        )

    # Update session state with current manual edits
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

    if st.button("🔍 Predict (Manual Inputs)"):

        # MODEL 1 — LANDSLIDE TYPE
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
        st.success(f"🌋 Landslide Type: {landslide}")

        ls_no = 1 if landslide == "No Landslide" else 0
        ls_rock = 1 if landslide == "Rockfall" else 0

        # MODEL 2 — FATALITY LEVEL
        m2 = {
            **m1,
            "population_density": population_density,
            "day_night_Night": 1 if day_night == "Night" else 0,
            "landslide_type_No Landslide": ls_no,
            "landslide_type_Rockfall": ls_rock,
        }

        df_m2 = pd.DataFrame([m2])[model2.feature_names_in_]
        fatality = model2.predict(df_m2)[0]
        st.warning(f"💀 Fatality Level: {fatality}")

        death_estimate = {
            "Low": "0–2 possible deaths",
            "Moderate": "3–10 possible deaths",
            "High": "10–50 possible deaths",
            "Severe": "50+ possible deaths"
        }
        st.error(
            f"🧍 If **no rescue response** is taken, estimated casualties: "
            f"**{death_estimate.get(fatality, 'N/A')}**"
        )

        fat_mod = 1 if fatality == "Moderate" else 0
        fat_sev = 1 if fatality == "Severe" else 0

        # MODEL 3 — RESCUE RESPONSE
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
        st.info(f"🚑 Recommended Rescue Response Level: {rescue}")

# =====================================================================
# PAGE 4 — REAL-TIME RAINFALL & ALERTS (API)
# =====================================================================
elif page == "Real-time Rainfall & Alerts":

    st.title("🌧 Real-time Rainfall Monitoring, Alerts & Safe Place Suggestion")

    st.markdown("""
    This page connects to a **FastAPI backend** (running separately) that:
    - Gets live **rainfall**, **temperature**, **humidity**, **altitude**, and **slope angle**  
      for a given location using external APIs  
    - Checks if rainfall exceeds a threshold and simulates an **Email/SMS alert**  
    - Suggests another location with **lower rainfall** as a safer place.
    """)

    location = st.text_input("Enter location (e.g., Munnar, Kerala)", "Munnar, Kerala")

    if st.button("📡 Get Live Data & Check Risk"):
        try:
            api_url = "http://127.0.0.1:8000/get_features"  # Your FastAPI endpoint
            res = requests.get(api_url, params={"location": location})
            res.raise_for_status()
            data = res.json()

            st.success("✅ Live data fetched successfully!")
            st.write("### 🌍 Environmental Features")
            st.json(data)

            rainfall = float(data.get("rainfall_mm", 0.0))
            slope_angle = float(data.get("slope_angle", 0.0))

            st.write(f"**Rainfall (last 1h):** {rainfall:.2f} mm")
            st.write(f"**Slope Angle (approx):** {slope_angle:.2f}°")

            # ----- Simple alert logic -----
            if rainfall > 20 and slope_angle > 15:
                st.error("⚠ High landslide risk based on heavy rainfall and steep slope.")
                high_risk = True
            elif rainfall > 10:
                st.warning("⚠ Moderate risk: rainfall is significant.")
                high_risk = False
            else:
                st.success("✅ Low immediate landslide risk based on current rainfall.")
                high_risk = False

            # ----- Simulated alert sending via Email/SMS -----
            if high_risk:
                st.markdown("---")
                st.subheader("🚨 Send Alert (Simulated)")

                contact_method = st.selectbox("Choose alert method", ["None", "Email", "SMS"])

                if contact_method == "Email":
                    to_email = st.text_input("Recipient email", "example@example.com")
                    if st.button("📧 Send Email Alert"):
                        send_email_alert(
                            to_email,
                            subject="High Landslide Risk Alert",
                            message=f"High landslide risk detected in {location}. Rainfall={rainfall:.2f} mm, slope={slope_angle:.2f}°"
                        )
                        st.success("Simulated email alert sent.")

                elif contact_method == "SMS":
                    to_phone = st.text_input("Recipient phone number", "+910000000000")
                    if st.button("📱 Send SMS Alert"):
                        send_sms_alert(
                            to_phone,
                            message=f"High landslide risk in {location}. Rainfall={rainfall:.2f} mm, slope={slope_angle:.2f}°"
                        )
                        st.success("Simulated SMS alert sent.")

            # ----- Suggest safer place based on lower rainfall -----
            st.markdown("---")
            st.subheader("🧭 Suggesting a Safer Nearby Location (Lower Rainfall)")

            candidate_locations = [
                "Kochi, Kerala",
                "Coimbatore, Tamil Nadu",
                "Bangalore, Karnataka"
            ]

            safer_place = None
            min_rain = None

            for loc in candidate_locations:
                try:
                    r2 = requests.get(api_url, params={"location": loc})
                    r2.raise_for_status()
                    d2 = r2.json()
                    rfall2 = float(d2.get("rainfall_mm", 0.0))

                    if (min_rain is None) or (rfall2 < min_rain):
                        min_rain = rfall2
                        safer_place = (loc, d2)
                except Exception:
                    continue

            if safer_place:
                name, sp_data = safer_place
                st.success(f"✅ Safer suggestion based on lower current rainfall: **{name}**")
                st.write(f"Rainfall there: **{min_rain:.2f} mm**, which is lower than in **{location}**.")
                st.write("You can integrate maps later to visualize this.")
            else:
                st.info("Could not find a safer alternative (API error or all places have similar rainfall).")

        except Exception as e:
            st.error("❌ Could not fetch live data. Make sure the FastAPI backend is running.")
            st.text(str(e))
