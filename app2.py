import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
with open('titanic_file', 'rb') as model_file:
    model = pickle.load(model_file)

# Custom CSS styling
st.markdown(
    """
    <style>
    .title {
        font-family: Arial, sans-serif;
        font-size: 36px;
        text-align: center;
        color: #004a8e;
    }
    .header {
        font-family: Arial, sans-serif;
        font-size: 24px;
        color: #004a8e;
        margin-top: 20px;
    }
    .subheader {
        font-family: Arial, sans-serif;
        font-size: 20px;
        color: #004a8e;
    }
    .prediction-info {
        font-family: Arial, sans-serif;
        font-size: 18px;
        color: #004a8e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.markdown("<h1 class='title'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)

# Sidebar for value limits
st.sidebar.markdown("<h2 class='header'>Value Limits</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Passenger Class: 1, 2, 3")
st.sidebar.markdown("Sex: Male, Female")
st.sidebar.markdown("Age: 0 - 100")
st.sidebar.markdown("Fare: 0.0 - 512.33")

# User input for feature values in the main area
st.markdown("<h2 class='header'>Enter Passenger Information</h2>", unsafe_allow_html=True)

# Passenger Class
pclass = st.radio("Passenger Class", [1, 2, 3], index=2)

# Sex
sex = st.radio("Sex", ["Male", "Female"], index=1)

# Age
age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)

# Fare
fare = st.number_input("Fare", min_value=0.0, max_value=512.33, value=50.0, step=0.01)

# Convert 'sex' feature to numeric representation
sex = 0 if sex == "Male" else 1

# Submit button
if st.button("Predict"):
    # Make prediction
    features = {
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'fare': fare
    }

    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)
    prediction_str = "Survived" if prediction[0] == 1 else "Not Survived"

    # Display prediction
    st.markdown("<h2 class='header'>Predicted Passenger Information</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-info'>Passenger Class: <b>{pclass}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-info'>Sex: <b>{'Male' if sex == 0 else 'Female'}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-info'>Age: <b>{age}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-info'>Fare: <b>{fare}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-info'>Prediction: <b>{prediction_str}</b></p>", unsafe_allow_html=True)
