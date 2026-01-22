import streamlit as st

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
    
    ### **Models**
    Can be either one of the 2 types of models: 
    **Regression** (**Linear Regression**) or **Classification** (**Random Forest Classifier**).
    
    After deciding upon your model, choose if you want to get your results 
    from **Real Data** or **Combined Data (70% real + 30% fake)**
    
    Now you can choose your inputs that will get you your book results
    
    ### **Chatbot**
    Interact with our AI-powered Book Recommendation Assistant!
    You can text and say what you want and the Chatbot will get it for you.
    
    This intelligent chatbot combines voice recognition and text processing to help
    you discover great books through a conversation.
    
    ##### **What can the chatbot do?**

    - Find top-rated books from our dataset
    - Search for books by specific authors
    - Look up detailed information about any book title
    - Respond to your queries through both text and voice
    - Provide audio responses that read answers aloud
""")