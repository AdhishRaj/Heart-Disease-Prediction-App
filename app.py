import streamlit as st
from joblib import load
import pandas as pd

model = load("random_forest_model.joblib")

st.title("Heart Disease Prediction App")

st.header("Enter the patient details below")
cp = st.selectbox("Type of Chest Pain(0 - Typical Angina, 1 - Atypical Angina, 2 - Non Anginal Pain, 3 - Asymptomatic)", (0, 1, 2, 3))
sex = st.selectbox("Gender(0 - Female, 1 - Male)", (0, 1))
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=210)
exang = st.selectbox("Exercise induced angina (0 - No, 1 - Yes)", (0, 1))
oldpeak = st.number_input("ST depression", min_value=0, max_value=7)

prediction = model.predict([[cp, sex, thalach, exang, oldpeak]])

st.header("Prediction Result")
if prediction[0] == 0:
    st.success("The patient is free from any Heart Disease")
else:
    st.error("The patient is suffering from a Heart Disease")