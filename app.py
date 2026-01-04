import streamlit as st
import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# App configuration
st.set_page_config(page_title="Landslide Prediction System", layout="wide")
load_dotenv()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Home", "Landslide Prediction", "Landslide Knowledge Chatbot"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":
    st.title("Landslide Prediction System")
    st.markdown("""
### Modules
- Landslide Prediction (Type, Fatality Level, Rescue Response Level)
- Landslide Knowledge Chatbot

### Technologies
- Streamlit
- Scikit-learn (Random Forest)
- LangChain
- Groq LLM
- FAISS
- Ollama / HuggingFace embeddings
""")

# =====================================================
# LANDSLIDE PREDICTION
# =====================================================
elif page == "Landslide Prediction":
    st.title("Landslide Impact Prediction")

    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models_joblib"

    def load_model(name):
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            st.error(f"Model file {path} not found. Upload it to the project folder.")
            st.stop()
        return joblib.load(path)

    model1 = load_model("landslide_type_model")
    model2 = load_model("fatality_model")
    model3 = load_model("rescue_response_model")

    mode = st.radio("Choose Prediction Mode", ["Manual Input", "Random Scenario"])

    if mode == "Random Scenario":
        if st.button("Generate & Predict"):
            # Generate random input
            data = {
                "rainfall_mm": np.round(np.random.uniform(0, 500), 2),
                "soil_moisture": np.round(np.random.uniform(0.2, 1.0), 2),
                "slope_angle": np.round(np.random.uniform(5, 60), 2),
                "vegetation_density": np.round(np.random.uniform(0.1, 0.9), 2),
                "distance_to_river_km": np.round(np.random.uniform(0.1, 5), 2),
                "population_density": np.round(np.random.uniform(50, 2000), 2),
                "altitude_m": np.round(np.random.uniform(200, 3500), 2),
                "infrastructure_quality": np.random.choice(["Low", "Medium", "High"]),
                "season": np.random.choice(["Summer", "Winter", "Monsoon"]),
                "day_night": np.random.choice(["Day", "Night"]),
            }

            st.write("Generated Input Values:", data)

            # Model 1: Landslide Type
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
            df1 = pd.DataFrame([m1])[model1.feature_names_in_]
            landslide = model1.predict(df1)[0]
            st.write("Predicted Landslide Type:", landslide)

            # Model 2: Fatality Level
            ls_no = 1 if landslide == "No Landslide" else 0
            ls_rock = 1 if landslide == "Rockfall" else 0
            m2 = {**m1,
                  "population_density": data["population_density"],
                  "day_night_Night": 1 if data["day_night"] == "Night" else 0,
                  "landslide_type_No Landslide": ls_no,
                  "landslide_type_Rockfall": ls_rock
                  }
            df2 = pd.DataFrame([m2])[model2.feature_names_in_]
            fatality = model2.predict(df2)[0]
            st.write("Predicted Fatality Level:", fatality)

            # Model 3: Rescue Response Level
            fat_mod = 1 if fatality == "Moderate" else 0
            fat_sev = 1 if fatality == "Severe" else 0
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
            df3 = pd.DataFrame([m3])[model3.feature_names_in_]
            rescue = model3.predict(df3)[0]
            st.write("Predicted Rescue Response Level:", rescue)

    else:  # Manual input
        # Input fields
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
        soil = st.slider("Soil Moisture", 0.0, 1.0, 0.5)
        slope = st.slider("Slope Angle", 0.0, 90.0, 25.0)
        vegetation = st.slider("Vegetation Density", 0.0, 1.0, 0.4)
        distance = st.number_input("Distance to River (km)", 0.0, 10.0, 1.0)
        population = st.number_input("Population Density", 50.0, 2000.0, 300.0)
        altitude = st.number_input("Altitude (m)", 0.0, 4000.0, 500.0)
        infra = st.selectbox("Infrastructure Quality", ["Low", "Medium", "High"])
        season = st.selectbox("Season", ["Summer", "Winter", "Monsoon"])
        day_night = st.selectbox("Day / Night", ["Day", "Night"])

        if st.button("Predict"):
            m1 = {
                "rainfall_mm": rainfall,
                "soil_moisture": soil,
                "slope_angle": slope,
                "vegetation_density": vegetation,
                "distance_to_river_km": distance,
                "altitude_m": altitude,
                "infrastructure_quality_Low": 1 if infra == "Low" else 0,
                "infrastructure_quality_Medium": 1 if infra == "Medium" else 0,
                "season_Summer": 1 if season == "Summer" else 0,
                "season_Winter": 1 if season == "Winter" else 0,
            }
            df1 = pd.DataFrame([m1])[model1.feature_names_in_]
            landslide = model1.predict(df1)[0]
            st.write("Predicted Landslide Type:", landslide)

            ls_no = 1 if landslide == "No Landslide" else 0
            ls_rock = 1 if landslide == "Rockfall" else 0
            m2 = {**m1,
                  "population_density": population,
                  "day_night_Night": 1 if day_night == "Night" else 0,
                  "landslide_type_No Landslide": ls_no,
                  "landslide_type_Rockfall": ls_rock
                  }
            df2 = pd.DataFrame([m2])[model2.feature_names_in_]
            fatality = model2.predict(df2)[0]
            st.write("Predicted Fatality Level:", fatality)

            fat_mod = 1 if fatality == "Moderate" else 0
            fat_sev = 1 if fatality == "Severe" else 0
            m3 = {
                "slope_angle": slope,
                "vegetation_density": vegetation,
                "distance_to_river_km": distance,
                "population_density": population,
                "altitude_m": altitude,
                "infrastructure_quality_Low": m1["infrastructure_quality_Low"],
                "infrastructure_quality_Medium": m1["infrastructure_quality_Medium"],
                "day_night_Night": m2["day_night_Night"],
                "landslide_type_No Landslide": ls_no,
                "landslide_type_Rockfall": ls_rock,
                "fatality_level_Moderate": fat_mod,
                "fatality_level_Severe": fat_sev,
            }
            df3 = pd.DataFrame([m3])[model3.feature_names_in_]
            rescue = model3.predict(df3)[0]
            st.write("Predicted Rescue Response Level:", rescue)

# =====================================================
# LANDSLIDE KNOWLEDGE CHATBOT
# =====================================================
elif page == "Landslide Knowledge Chatbot":
    st.title("Landslide Knowledge Chatbot")

    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in Streamlit secrets.")
        st.stop()

    if "vectors" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            loader = WebBaseLoader(
                "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/landslide.html"
            )
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)

            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.5)

    prompt = ChatPromptTemplate.from_template("""
Answer the user's question ONLY about landslides or related topics.

Rules:
- Use context first. If context does not fully answer, use general landslide knowledge.
- Do NOT answer unrelated questions. If question is unrelated, give a short note.
- Keep explanation simple and concise.

<context>
{context}
</context>

Question: {input}
""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_input = st.text_input("Ask anything about landslides")
    if user_input:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        st.write(response["answer"])
        st.caption(f"Response Time: {time.process_time() - start:.2f} seconds")
