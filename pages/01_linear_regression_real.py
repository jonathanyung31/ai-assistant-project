import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="GoodreadsRec",page_icon="📖", layout="wide")

st.title('Goodreads recommandation App')
st.write("## This App will Recommend you Books you might Like!")
st.markdown('### **Linear Regression - Real Data**')

@st.cache_resource
def load_models():
    try:
        lin_model = joblib.load("models/book_lin_model.joblib")
        lin_features = joblib.load("models/book_lin_features.joblib")
        scaler = joblib.load("models/scaler_lin_real.joblib")
        return lin_model, lin_features, scaler
    
    except FileNotFoundError:
        st.error("Model files not found.")
        st.stop()

lin_model, lin_features, scaler = load_models()

# Load real dataset

@st.cache_data
def load_book_data():
    try:
        return pd.read_csv('data/books_copy.csv')
    except FileNotFoundError:
        st.error("Book data not found!")
        st.stop()

df_real = load_book_data()

# Input Widgets

st.header("📋 Your Preferences")

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


book_age_days = st.slider("Book Age (in years):", 
    min_value=1, max_value=50, value=10,
    help="How old should the book be?") * 365


if st.button("Submit"):

    st.subheader("Linear Regression Prediction - Real Data")

    input_data = pd.DataFrame([{
        'num_pages': num_pages,
        'ratings_count': ratings_count,
        'book_age_days': book_age_days
    }])

    input_data = input_data[lin_features]
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=lin_features)
    
    predicted_rating = lin_model.predict(input_data_scaled)[0]
    st.success(f"Predicted Average Rating: **{predicted_rating:.2f} / 5.0**")

    if predicted_rating >= 4.5:
        st.info("Books with these inputs usually receive high ratings!")
    elif predicted_rating >= 4.0:
        st.info("Books with these inputs have better than usual ratings!")
    elif predicted_rating >= 3.5:
        st.info("This is a nice rating range")
    else:
        st.info("Low ratings for this combination of inputs")
    
    st.subheader("Here are the Books that Match Your Preference")

    filtered_books = df_real[
        (df_real['num_pages'].between(num_pages * 0.7, num_pages * 1.3)) &
        (df_real['ratings_count'].between(ratings_count * 0.5, ratings_count * 2)) &
        (df_real['book_age_days'].between(book_age_days * 0.7, book_age_days * 1.3))].copy()

    if len(filtered_books) > 0:
        top_books = filtered_books.nlargest(10, 'average_rating')
        st.write(f"These top 10 average ratings books were Found for You:")

        for id, book in top_books.iterrows():
            with st.expander(f"{book['title']} - Rating: {book['average_rating']:.2f}"):
                column1, column2 = st.columns([2, 1])
                with column1:
                    st.write(f"**Author:** {book['authors']}")
                    st.write(f"**Number of Pages:** {int(book['num_pages'])}")
                    st.write(f"**Publishers:** {book['publisher']}")
                with column2:
                    st.write(f"**Rating:** {book['average_rating']:.2f} / 5.0")
                    st.write(f"**Ratings Count** {int(book['ratings_count']):,}")
                    st.write(f"**Language** {book['language_code']}")
    else:
        st.warning("No Books matched your preferences. Try different inputs!")
else:
    st.info("Select your preferences and click 'Submit' to get your predictions!")
