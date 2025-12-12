import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import json
import random

'''
The Use Case the chatbot will be focused on is for the user
to see top-rated books
'''
import os
print(os.getcwd())

nltk.download('punkt')

with open("intents_data.json", "r") as f:
    training_data = json.load(f)


def extract_features(text):

    features = {}
    words = word_tokenize(text.lower())
    
    for word in words:
        features[f'has({word})'] = True
    
    return features

feature_sets = [(extract_features(item["text"]), item["intent"]) 
                for item in training_data] # Extracting from JSON

random.shuffle(feature_sets)

split_point = int(len(feature_sets) * 0.8)
train_set = feature_sets[:split_point]
test_set = feature_sets[split_point:]

classifier = nltk.NaiveBayesClassifier.train(feature_sets)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f"Overall Accuracy: {accuracy:.2%}")

st.title("Book Chatbot")
user_input = st.text_input("Do You Have any Questions about Books?")

if user_input:
    features = extract_features(user_input)
    intent = classifier.classify(features)

# Example response logic
    if intent == "top_rated_books":
        st.write("Here are the top-rated books...")
    elif intent == "top_rated_author_books":
        st.write("Which author's top books would you like?")
    elif intent == "find_book_by_title":
        st.write("Searching for that book...")
    elif intent == "get_book_details":
        st.write("Fetching book details...")
    else:
        st.write("Sorry, I didn't catch that, could you say that more clearly?")


'''
df = pd.read_csv("books_copy.csv")

def top_rated_books(df, n=5):
    # Sort by average_rating, then pick top n books
    top_books = df.sort_values('average_rating', ascending=False).head(n)
    return top_books[['title', 'authors', 'average_rating']]
'''


def speech_to_text():
    rec = sr.Recognizer()
    print("I'm Listening, You can Speak Now!")
    with sr.Microphone() as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = rec.listen(source, timeout=3, phrase_time_limit=12)
        
        except sr.WaitTimeoutError:
            print("No speech detected... Please try speaking again clearly🔊")
            return None
    print("Processing what you just said... Please wait a few seconds")
    
    try:
        text = rec.recognize_google(audio).lower()

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio, could you repeat that please?")
    except sr.RequestError:
        print(f"Could not find results, Sorry...")




'''
test_sentences = [
    "show me the top rated books",
    "tell me about Dune",
    "show me the best books by Stephen King"
]

for sentence in test_sentences:
    features = extract_features(sentence)
    predicted_intent = classifier.classify(features)
    print(f"'{sentence}' -> {predicted_intent}")
    
print("\nMost Informative Features (Mensa Data):")
classifier.show_most_informative_features(5)
'''
