import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import numpy as np

st.markdown(
 """
    <style>
    /* --- Page background --- */
    /* Page background to beige */
    .stApp {
    background-color: #E8DCB8 !important;

    /* --- Headers --- */
    h1, h2, h3, h4, h5, h6 {
        color: #FF8C42 !important;
    }

    /* --- Buttons --- */
    .stButton>button {
        background-color: #4a3b41;
        color: #FFAA5C;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7a656b;
        color: #FFD28C;
        cursor: pointer;
    }

    /* --- Radio buttons --- */
    div[role="radiogroup"] label div {
        color: #FF8C42 !important;  /* warm orange text */
        transition: color 0.3s ease;
    }
    div[role="radiogroup"] label:hover div {
        color: #FFA75C !important;  /* lighter orange on hover */
        cursor: pointer;
    }

    /* Slider track */
.stSlider > div > div > div > div {
    background-color: #4a3b41 !important;  /* dark track */
}

/* Slider value tooltip */
.stSlider > div > div > div > div > div > div > div {
    background-color: #0d0d0d !important;  /* match app background */
    color: #FF8C42 !important;  /* warm orange text */
    font-weight: bold;
    border-radius: 6px;
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="GoodreadsRec",page_icon="📖", layout="wide")

st.title('Goodreads recommandation App')
st.write("This App will Recommend you Books you might Like!")

@st.cache_resource
def load_models():
    try:
        # Classification
        rf_model = joblib.load("models/book_rf_model.joblib")
        rf_features = joblib.load("models/book_rf_features.joblib")

        # Regression
        lin_model = joblib.load("models/book_lin_model.joblib")
        lin_features = joblib.load("models/book_lin_features.joblib")

        return rf_model, rf_features, lin_model, lin_features
    
    except FileNotFoundError:
        st.error("Model files not found. Please ensure that the files are in the same directory")
        st.stop()   # Stop the app if models are not found

rf_model, rf_features, lin_model, lin_features = load_models()
# --- Input Widgets for Customer Features ---

st.sidebar.header("User Details")

average_rating = st.slider("Preferred average rating", 0.0, 5.0, 2.5, 0.5)
num_pages = st.radio("Do you like short, medium, or long books?", ["Short","Medium","Long"])
ratings_count = st.radio("Do you like popular or niche books?", ["Niche", "Popular", "Blockbusters"])

if st.button("Submit"):
    st.write("Processing your desired books now...")

input_data = {
    'average_rating' : average_rating,
    'num_pages' : num_pages,
    'ratings_count' : ratings_count
}

input_rf_df = pd.DataFrame([input_data])
input_rf_df = input_rf_df[rf_features] # Reorder columns to match trained model's features

input_lin_df = pd.DataFrame([input_data])
input_lin_df = pd.DataFrame([lin_features])

st.subheader("Customer Input Summary:")
st.dataframe(input_rf_df)
st.dataframe(input_lin_df)