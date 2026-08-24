import streamlit as st
import joblib

# 1. Load the saved model and features
model = joblib.load('logistic_regression_model.pkl')
features = joblib.load('model_features.pkl')

st.title("My Logistic Regression Model App")

# 2. Collect input values from user
inputs = []
st.write("### Enter Input Values:")
for feature in features:
    val = st.number_input(f"Enter {feature}:", value=0.0)
    inputs.append(val)

# 3. Prediction button
if st.button("Predict"):
    prediction = model.predict([inputs])
    st.success(f"Prediction result: {prediction[0]}")