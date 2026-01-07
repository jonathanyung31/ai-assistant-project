import streamlit as st
import pandas as pd

st.title("📊 Real Dataset")
df_real = pd.read_csv('data/books_copy.csv')
st.dataframe(df_real)
st.write(f"Total books: {len(df_real)}")