import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="GoodreadsRec", page_icon= "📖", layout="wide")
st.title('Goodreads recommandation App')
st.write("This App will Recommend you Books you might Like!")
st.markdown('**Random Forest Classification - Combined Data (70% real + 30% fake)**')


@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("models/book_rf_model_fake.joblib")
        rf_features = joblib.load("models/book_rf_features_fake.joblib")
        scaler = joblib.load("models/scaler_rand_fake.joblib")
        return rf_model, rf_features, scaler
    
    except FileNotFoundError:
        st.error("Model files not found.")
        st.stop()

rf_model, rf_features, scaler = load_models()

@st.cache_data
def load_book_data():
    try:
        return pd.read_csv('data/books_combined_fake.csv')
    except FileNotFoundError:
        st.error("Book data not found!")
        st.stop()

df_fake = load_book_data()

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
    st.subheader("""Random Forest Classifier Prediction - Combined Data (70% Real + 30% Fake)""")

    input_data = pd.DataFrame([{
        'num_pages': num_pages,
        'ratings_count': ratings_count,
        'book_age_days': book_age_days,
        'language_code_encoded': int(df_fake['language_code_encoded'].median()),
        'author_encoded': int(df_fake['author_encoded'].median())
    }])

    input_data = input_data[rf_features]
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=rf_features)

    predict_cat = rf_model.predict(input_data_scaled)[0]
    st.success(f"Predicted Rating Category: **{predict_cat}**")

    if predict_cat == 'High':
        st.info("Books with these inputs usually receive high ratings!")
    elif predict_cat == 'Medium':
        st.info("Books with these inputs have better than usual ratings!")
    else:
        st.info("Low ratings for this combination of inputs")

    st.subheader("Here are the Books that Match Your Preference")

    filter_books = df_fake[
        (df_fake['num_pages'].between(num_pages * 0.7, num_pages * 1.3)) &
        (df_fake['ratings_count'].between(ratings_count * 0.5, ratings_count * 2)) &
        (df_fake['book_age_days'].between(book_age_days * 0.7, book_age_days * 1.3)) &
        (df_fake['rating_category'] == predict_cat)
    ].copy()

    if len(filter_books) > 0:
        top = filter_books.nlargest(10, 'average_rating')
        st.write(f"""These top 10 average ratings books 
                were found for you in the **{predict_cat}** category""")
        
        for id, book in top.iterrows():
            with st.expander(f"{book['title']} - Rating: {book['average_rating']:.2f}"):
                general, specific = st.columns([2, 1])
                with general:
                    st.write(f"**Author:** {book['authors']}")
                    st.write(f"**Number of Pages:** {int(book['num_pages'])}")
                    st.write(f"**Publisher:** {book['publisher']}")
                with specific:
                    st.write(f"**Rating:** {book['average_rating']:.2f} / 5.0")
                    st.write(f"**Category:** {book['rating_category']}")
                    st.write(f"**Ratings Count:** {int(book['ratings_count']):,}")
                    st.write(f"**Language:** {book['language_code']}")
    else:
        st.warning("No Books matched your preferences. Try different inputs!")
else:
    st.info("Select your preferences and click 'Submit' to get your predictions!")