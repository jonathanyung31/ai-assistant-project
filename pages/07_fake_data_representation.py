import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fake Dataset", page_icon="🎭", layout="wide")
st.title("Fake Dataset (70% Real + 30% Fake)")
st.markdown("This page shows the fake dataset that includes" \
"both real and fake data used for model training.")

# Structure of Dataset
st.header("Dataset Overview")

try:
    df_combined = pd.read_csv(r"data\books_combined_fake.csv")
    df_real = pd.read_csv(r"data\books_copy.csv")
    df_fake = pd.read_csv(r"data\books_fake.csv")

    num_real = len(df_real)
    num_fake = len(df_fake)
    num_combined = len(df_combined)

    Total_books, Real_books, Fake_books, Columns = st.columns(4)
    with Total_books:
        st.metric("Total Books", f"{num_combined:,}")
    with Real_books:
        st.metric("Real Books", f"{num_real:,}", f"{num_real/num_combined*100:.1f}%")
    with Fake_books:
        st.metric("Fake Books", f"{num_fake:,}", f"{num_fake/num_combined*100:.1f}%")
    with Columns:
        st.metric("Columns", len(df_combined.columns))

    st.info("""
        Using Combined real and fake data helps us see how well do our models
        work with fake data compared to real data, as well as how well our models
        deal with fake data
            """)
    
except FileNotFoundError:
    st.error("One of the files were not Found. Run main.py first!")
    st.stop()

st.header("fake Data Generation")
st.markdown("""
        ### How Fake Data was Created
            
        The fake data was created using parameters that are based on the real
        data to make it look realistic:
            
        1. **Authors and Publishers**: Randomly selected from top 100 real authors and publishers
        2. **Ratings**: Generated between 2.5 and 5.0 (ratings of most real books)
        3. **Page Numbers**: Random values between 100-1000 pages (somewhat realistic)
        4. **Ratings Count**: Random values between 50-300,000
        5. **Book Age**: Random age between 1-55 years
        6. **Languages**: Randomly selected from real language codes in real dataset
            
        All fake books hav IDs starting from 100,000 so they are clearly seperated from real books ID's
            """)

# Comparison between Real Fake
st.header("Comparison Between Real and Fake Books")

real_examp, fake_examp = st.columns(2)

with real_examp:
    st.subheader("Real Book Example")
    try:
        real_sample = df_real[df_real['bookID'] < 100000].sample(5)
        st.dataframe(real_sample[['bookID', 'title', 'authors', 'average_rating', 'num_pages']],
                     use_container_width=True)
    except:
        st.write("No real books to display")
with fake_examp:
    st.subheader("Fake Book Example")
    try:
        fake_sample = df_combined[df_combined['bookID'] >= 100000].sample(5)
        st.dataframe(fake_sample[['bookID', 'title', 'authors', 'average_rating', 'num_pages']],
                     use_container_width=True)
        st.caption("As indicated before: Book IDs >= 100,000 indicate fake data")
    except:
        st.write("No fake books to display")

# Statistical Comparison
st.header("Statistical Comparison")

average, pages, count = st.columns(3)
with average:
    st.subheader("Average Rating")
    real_avg = df_real['average_rating'].mean()
    fake_avg = df_fake['average_rating'].mean()
    combined_avg = df_combined['average_rating'].mean()

    st.metric("Real Data", f"{real_avg:.2f}")
    st.metric("Fake Data", f"{fake_avg:.2f}")
    st.metric("Combined Data", f"{combined_avg:.2f}")
with pages:
    st.subheader("Average Pages")
    real_pages = df_real['num_pages'].mean()
    fake_pages = df_fake['num_pages'].mean()
    combined_pages = df_combined['num_pages'].mean()

    st.metric("Real Data", f"{real_pages:.0f}")
    st.metric("Fake Data", f"{fake_pages:.0f}")
    st.metric("Combined Data", f"{combined_pages:.0f}")
with count:
    st.subheader("Average Ratings Count")
    real_ratings = df_real['ratings_count'].mean()
    fake_ratings = df_fake['ratings_count'].mean()
    combined_ratings = df_combined['ratings_count'].mean()

    st.metric("Real Data", f"{real_ratings:,.0f}")
    st.metric("Fake Data", f"{fake_ratings:,.0f}")
    st.metric("Combined Data", f"{combined_ratings:,.0f}")

# Looking into Combined Dataset
st.header("Looking Through Combined Data")

st.dataframe(df_combined, use_container_width=True, height=500)

with st.expander("Statistical Summary"):
    st.dataframe(df_combined.describe(), use_container_width=True)

with st.expander("Column Info"):
    column_info = pd.DataFrame({
        'Column': df_combined.columns,
        'Data Type': df_combined.dtypes.values,
        'Non-Null Count': df_combined.count().values,
        'Null Count': df_combined.isnull().sum().values
    })
    st.dataframe(column_info, use_container_width=True)

# Rating Distribution
st.header("Rating Category Distribution")

rating_counts = df_combined['rating_category'].value_counts()
low, medium, high = st.columns(3)
with low:
    st.metric("Low Rated", rating_counts.get("Low", 0))
with medium:
    st.metric("Medium Rated", rating_counts.get("Medium", 0))
with high:
    st.metric("High Rated", rating_counts.get("High", 0))


