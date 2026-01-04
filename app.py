import streamlit as st
import os
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Streamlit config
st.set_page_config(page_title="Landslide Prediction System", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Home", "Landslide Prediction", "Landslide Knowledge Chatbot"]
)

# Home
if page == "Home":
    st.title("Landslide Intelligence System")
    st.markdown("""
### Landslide Prediction System

This project is a simple and practical landslide prediction system built using machine learning and AI to support disaster analysis and awareness.

The system predicts three important outcomes:
Landslide type  
Fatality level  
Rescue response level  

A Random Forest model is used to make predictions based on factors such as rainfall, soil moisture, slope angle, vegetation density, population density, altitude, and infrastructure quality.

Along with prediction, the project includes a Landslide Knowledge Chatbot. This chatbot allows users to ask questions related to landslides and provides clear, relevant answers using trusted information sources and AI-based retrieval.

### Modules
Landslide Prediction  
Landslide Knowledge Chatbot  

### Technologies Used
Streamlit  
Scikit-learn  
Pandas and NumPy  
LangChain  
FAISS  
Groq LLM  

This project focuses on applying machine learning to real-world disaster management problems in a clear and easy-to-use way.
""")


# Landslide Prediction
elif page == "Landslide Prediction":
    st.title("Landslide Impact Prediction")

    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models_joblib"

    def load_model(name):
        return joblib.load(MODEL_DIR / f"{name}.joblib")

    try:
        model1 = load_model("landslide_type_model")
        model2 = load_model("fatality_model")
        model3 = load_model("rescue_response_model")
    except Exception as e:
        st.error("Model files not found")
        st.exception(e)
        st.stop()

    mode = st.radio("Choose Prediction Mode", ["Manual Input", "Random Scenario"])

    if mode == "Random Scenario":
        if st.button("Generate & Predict"):
            data = {
                "rainfall_mm": np.random.uniform(0, 500),
                "soil_moisture": np.random.uniform(0.2, 1.0),
                "slope_angle": np.random.uniform(5, 60),
                "vegetation_density": np.random.uniform(0.1, 0.9),
                "distance_to_river_km": np.random.uniform(0.1, 5),
                "population_density": np.random.uniform(50, 2000),
                "altitude_m": np.random.uniform(200, 3500),
                "infrastructure_quality": np.random.choice(["Low", "Medium", "High"]),
                "season": np.random.choice(["Summer", "Winter", "Monsoon"]),
                "day_night": np.random.choice(["Day", "Night"]),
            }

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
            st.success(f"Landslide Type: {landslide}")

            ls_no = 1 if landslide == "No Landslide" else 0
            ls_rock = 1 if landslide == "Rockfall" else 0

            m2 = {
                **m1,
                "population_density": data["population_density"],
                "day_night_Night": 1 if data["day_night"] == "Night" else 0,
                "landslide_type_No Landslide": ls_no,
                "landslide_type_Rockfall": ls_rock,
            }

            df2 = pd.DataFrame([m2])[model2.feature_names_in_]
            fatality = model2.predict(df2)[0]
            st.warning(f"Fatality Level: {fatality}")

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
            st.info(f"Rescue Response Level: {rescue}")

# Chatbot
elif page == "Landslide Knowledge Chatbot":
    st.title("Landslide Knowledge Chatbot")

    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found")
        st.stop()

    if "vectors" not in st.session_state:
        with st.spinner("Loading knowledge base"):
            loader = WebBaseLoader(
                "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/landslide.html"
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the user's question using the context below.

Rules:
- Answer landslide-related questions and its related.
- Use context first.
- If context is insufficient, use general landslide knowledge only.
- Clearly mention when using general knowledge.
- Do not answer unrelated topics.

<context>
{context}
</context>

Question: {input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_input = st.text_input("Ask a landslide-related question")

    if user_input:
        response = retrieval_chain.invoke({"input": user_input})
        st.success(response.get("answer", "No relevant information found"))
