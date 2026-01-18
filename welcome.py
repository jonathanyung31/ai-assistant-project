import streamlit as st

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

st.title('Welcome to the Goodreads recommandation App!')
st.write("This App will Recommend you Books you might Like!")

st.markdown("""
    ## Discover your next favorite read here!
    This app uses machine learning models to help you find books based on your preferences.
    
    **What can you do here?**
    - You can view the dataset that is being used to train the ML models.
    - Using these ML models you can examine how different features effect the book results.
            
    **How to get Started?**
    Select the page you want to get results from.
    
    **Models**
    Can be either one of the 2 types of models: 
    **Regression** (**Linear Regression**) or **Classification** (**Random Forest Classifier**).
    
    After deciding upon your model, choose if you want to get your results 
    from **Real Data** or **Combined Data (70% real + 30% fake)**
    
    Now you can choose your inputs that will get you your book results
    
    **Chatbot**
    
""")