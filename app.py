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
        st.error("Model files not found.")
        st.stop()

rf_model, rf_features, lin_model, lin_features = load_models()

# Load real dataset

@st.cache_data
def load_book_data():
    try:
        return pd.read_csv('data/books_copy.csv')
    except FileNotFoundError:
        st.error("Book data not found!")
        st.stop()

df_real = load_book_data()

# --- Input Widgets for Customer Features ---

st.sidebar.header("📋 Your Preferences")

num_pages = st.radio("Preffered Book Length:",
["Short (< 250 pages)", "Medium (250-450 pages)",
  "Long (> 450 pages)"])

if "Short" in num_pages:
    num_pages = 200
elif "Medium" in num_pages:
    num_pages = 350
else:
    num_pages = 600


ratings_count = st.radio("Preffered Popularity",
["Niche", "Popular", "Blockbusters"])

if "Niche" in ratings_count:
    ratings_count = 500
elif "Popular" in ratings_count:
    ratings_count = 5000
else:
    ratings_count = 50000


average_rating = st.slider("Preferred average rating:", 0.0, 5.0, 2.5, 0.5)

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