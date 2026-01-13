import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Real Dataset", page_icon="📊", layout="wide")
st.title("📊 Real Dataset Walkthrough")
st.markdown("This page shows the data preprocessing processes from raw" \
"data to the final cleaned dataset ready to be used.")

# --- Original Raw Data ---
st.header("Original Raw Data")

with st.expander("📂 View Original Dataset Before Cleaning", expanded=False):
    try:
        df_real = pd.read_csv("data/books.csv", on_bad_lines="skip")
        df_real.columns = df_real.columns.str.strip()
        column1,column2, column3 = st.columns(3)
        with column1:
            st.metric("Total Books", f"{len(df_real):,}")
        with column2:
            st.metric("Columns", len(df_real.columns))
        with column3:
            st.metric("Duplicates", df_real.duplicated().sum())
        
        st.dataframe(df_real.head(20), use_container_width=True)
        st.subheader("Problems that were Found:")
        st.warning(f""" 
        - {df_real.duplicated().sum()} duplicate entries
        - 76 books with 0 pages
        - 25 books with 0 average rating
        - 80 books with 0 ratings count
        - Extra whitespace in column names
        """)

    except FileNotFoundError:
        st.error("Original Dataset not found")


# --- Statistical Analysis ---
st.header("Statistical Analysis Before Cleaning")

st.subheader("📊 Distributions")
try:
    pic1 = Image.open("visuals/before_cleaning_distributions.png")
    st.image(pic1, use_container_width=True)
except FileNotFoundError:
    st.warning("May need to run main.py file to generate visuals")

st.subheader("⭕ Zero Values")  # ← No indentation here
try:
    pic2 = Image.open("visuals/zero_values_heatmap.png")
    st.image(pic2, use_container_width=True)
except FileNotFoundError:
    st.warning("May need to run main.py file to generate visuals")

# --- Cleaning Steps ---
st.header("Data Cleaning Steps")
st.markdown("""
    The transformations indicated below were applied to clean the data:
""")

cleaning_steps = [

]


# --- Cleaned Data ---
st.header("Cleaned Data")

try:
    df_real_after = pd.read_csv('data/books_copy.csv')

    column6, column7, column8 = st.columns(3)
    with column6:
        st.metric("Total Books", f"{len(df_real_after):,}")
    with column7:
        st.metric("Columns", len(df_real_after.columns))
    with column8:
        avg_rating = df_real_after["average_rating"].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")

    st.subheader("Look Into Cleaned Data")
    st.dataframe(df_real_after, use_container_width=True, height=400)

    with st.expander("Statistical Summary"):
        st.dataframe(df_real_after.describe(), use_container_width=True)
    
    with st.expander("Column Information"):
        column_info = pd.DataFrame({
            "Column": df_real_after.columns,
            "Data Type": df_real_after.dtypes.values,
            "Non-Null Count": df_real_after.count().values,
            "Null Count": df_real_after.isnull().sum().values
        })
        st.dataframe(column_info, use_container_width=True)

    with st.expander("Random Books"):
        st.dataframe(df_real_after.sample(10), use_container_width=True)
    

    st.subheader("Rating Distribution")
    rating_counts = df_real_after["rating_category"].value_counts()
    column9, column10, column11 = st.columns(3)
    with column9:
        st.metric("Low Rated", rating_counts.get("Low", 0))
    with column10:
        st.metric("Mdeium Rated", rating_counts.get("Medium", 0))
    with column11:
        st.metric("High Rated", rating_counts.get("High", 0))

except FileNotFoundError:
    st.error("Cleaned dataset no found. Run main.py for that")

st.subheader("Steps that were taken to Clean Data:")
st.markdown("""
- **Remove Duplicates**: Eliminated duplicate book entries
- **Strip Column Names**: Removed whitespace from column headers  
- **Drop Unused Columns**: Removed 'text_reviews_count' (too many zeros)
- **Fix num_pages**: Replaced 0 values with median (299 pages)
- **Remove Invalid Ratings**: Dropped books with 0 average rating
- **Remove Zero Rating Counts**: Dropped books with 0 ratings count
- **Encode language_code**: Converted text to numbers using Label Encoding
- **Calculate book_age_days**: Computed age from publication date
- **Encode authors**: Converted author names to numbers using Label Encoding
- **Create rating_category**: Low (0-1.7), Medium (1.7-3.4), High (3.4-5.0)
""")

st.header("Key Takeaways")
avg_pages = df_real_after["num_pages"].mean()
avg_rating = df_real_after["average_rating"].mean()
num_languages = df_real_after["language_code"].nunique()
num_publishers = df_real_after["publisher"].nunique()
most_common = df_real_after["rating_category"].value_counts().idxmax()
total_count = len(df_real_after)

st.success(f"""
           **Data is ready for Models!**

           **Important Statistics**
           - Average Book Length: ~{avg_pages:.0f} pages
           - Average Rating: {avg_rating:.2f} / 5.0
           - Dataset Spans {num_languages} languages and {num_publishers} publishers
           - Total Books after Cleaning: {len(df_real_after):,}
           """)

st.info("""
**How to use this Data:**
1. Browse the dataset
2. Look into Statistical Summaries
3. Understanding the Cleaning Process
4. Walkthrough of how Raw Data becomes ML-ready
""")