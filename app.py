import streamlit as st
import numpy as np
import joblib
import random
import os

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Landslide Prediction System", layout="wide")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("landslide_model.pkl")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Knowledge Chatbot"])

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.title("Landslide Prediction System")

    st.markdown("""
    This project is a web-based system designed to predict landslide risks and provide landslide-related knowledge.

    Key features
    - Predicts landslide type using a Random Forest model
    - Estimates fatality level based on environmental and population factors
    - Suggests rescue response level for emergency planning
    - Supports both manual input and random scenario generation
    - Displays generated values used during random predictions

    Technologies used
    - Streamlit for web application
    - Scikit-learn for machine learning
    - Pandas and NumPy for data handling
    - LangChain for chatbot logic
    - FAISS for document retrieval
    - Groq LLM using LLaMA Instant model

    Landslide Knowledge Chatbot
    - Answers only landslide and landslide-related questions
    - Uses context when available
    - If context is missing, responds using general landslide knowledge
    - Politely informs users when questions are outside the landslide domain
    """)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
elif page == "Prediction":
    st.title("Landslide Prediction")

    mode = st.radio("Select Input Mode", ["Manual Input", "Random Scenario"])

    if mode == "Manual Input":
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 120.0)
        soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 45.0)
        slope = st.number_input("Slope Angle (degrees)", 0.0, 90.0, 30.0)
        vegetation = st.number_input("Vegetation Density (%)", 0.0, 100.0, 60.0)
        population = st.number_input("Population Density", 0.0, 10000.0, 800.0)
        altitude = st.number_input("Altitude (m)", 0.0, 6000.0, 900.0)
        infrastructure = st.number_input("Infrastructure Quality (1–10)", 1.0, 10.0, 6.0)

        if st.button("Predict"):
            features = np.array([[rainfall, soil_moisture, slope, vegetation,
                                  population, altitude, infrastructure]])

            prediction = model.predict(features)[0]

            st.subheader("Prediction Result")
            st.write("Landslide Type:", prediction[0])
            st.write("Fatality Level:", prediction[1])
            st.write("Rescue Response Level:", prediction[2])

    else:
        if st.button("Generate Random Scenario"):
            rainfall = random.uniform(50, 300)
            soil_moisture = random.uniform(20, 90)
            slope = random.uniform(10, 60)
            vegetation = random.uniform(20, 90)
            population = random.uniform(100, 5000)
            altitude = random.uniform(100, 3000)
            infrastructure = random.uniform(1, 10)

            features = np.array([[rainfall, soil_moisture, slope, vegetation,
                                  population, altitude, infrastructure]])

            prediction = model.predict(features)[0]

            st.subheader("Generated Values")
            st.write("Rainfall:", rainfall)
            st.write("Soil Moisture:", soil_moisture)
            st.write("Slope Angle:", slope)
            st.write("Vegetation Density:", vegetation)
            st.write("Population Density:", population)
            st.write("Altitude:", altitude)
            st.write("Infrastructure Quality:", infrastructure)

            st.subheader("Prediction Result")
            st.write("Landslide Type:", prediction[0])
            st.write("Fatality Level:", prediction[1])
            st.write("Rescue Response Level:", prediction[2])

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------
elif page == "Knowledge Chatbot":
    st.title("Landslide Knowledge Chatbot")

    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
You are a landslide knowledge assistant.

Rules:
- Answer only questions related to landslides, landslide causes, impacts, safety, prediction, fatalities, or rescue.
- Do not mention context or documents.
- If the question is outside landslides, say:
  "This question is outside the landslide domain. Here is a short answer based on general knowledge."
- Keep answers simple and clear.

Question: {question}
Answer:
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    user_question = st.text_input("Ask a landslide-related question")

    if user_question:
        response = chain.run(user_question)
        st.write(response)
