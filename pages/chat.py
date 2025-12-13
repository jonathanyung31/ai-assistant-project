import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import json
import random
from collections import defaultdict
import pandas as pd

'''
The Use Case the chatbot will be focused on is for the user
to see top-rated books
'''

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

print("\nMost Informative Features (Mensa Data):")
classifier.show_most_informative_features(5)

# 1. Get predictions for the test set
refsets = defaultdict(set) # Reference (Actual) Intents
testsets = defaultdict(set) # Predicted Intents

for i, (features, intent_actual) in enumerate(test_set):
    intent_predicted = classifier.classify(features)
    refsets[intent_actual].add(i)
    testsets[intent_predicted].add(i)

st.title("Book Chatbot")
user_input = st.text_input("Do You Have any Questions about Books?")

if user_input:
    features = extract_features(user_input)
    intent = classifier.classify(features)

intents = classifier.labels()

for intent in intents:
    if intent in refsets:

        precision = nltk.scores.precision(refsets[intent], testsets[intent])

        recall = nltk.scores.recall(refsets[intent], testsets[intent])

        f1_score = nltk.scores.f_measure(refsets[intent], testsets[intent])

        # Using 0.0 if one of matrices is None
        print(f"Intent: {intent}")
        print(f"Precision: {(precision if precision else 0.0):.2f}")
        print(f"Recall: {(recall if recall else 0.0):.2f}")
        print(f"F1-Score: {(f1_score if f1_score else 0.0):.2f}")

test_sentences = [
    "show me the top rated books",
    "tell me about Dune",
    "show me the best books by Stephen King"
]

for sentence in test_sentences:
    features1 = extract_features(sentence)
    predicted_intent = classifier.classify(features1)  # ← Use features1, not features
    print(f"'{sentence}' -> {predicted_intent}")
    # Remove the second print line entirely - it's causing the weird output

if intent == "top_rated_books":
    st.write("Here are the top-rated books...")
elif intent == "get_books_by_author":
    st.write("Which author's top books would you like?")
elif intent == "search_by_title":
    st.write("Searching for that book...")
elif intent == "get_details":
    st.write("Fetching book details...")
else:
    st.write("Sorry, I didn't catch that, could you say that more clearly?")

@st.cache_data
def load_data():
    df = pd.read_csv("data/books_copy.csv")
    return df

df = load_data()

def top_rated_books(limit=10):
    #Get the highest rated books
    top_books = df.nlargest(limit, 'average_rating')
    #Each row becomes one dictionary
    return top_books[['title', 'authors', 'average_rating']].to_dict('records')

def get_books_by_author(author_name, limit=10):
    #Get books by a specific author
    author_books = df[df['authors'].str.contains(author_name, case=False)]
    author_books = author_books.nlargest(limit, 'average_rating')
    return author_books[['title', 'authors', 'average_rating']].to_dict('records')

def search_by_title(title):
    #Find a specific book by title
    result = df[df['title'].str.contains(title, case=False)]
    if len(result) > 0:
        return result.iloc[0].to_dict()
    return None

def get_details(title):
    #Get full details about a book
    book = search_by_title(title)
    return book


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
        print(f"Could not find results. If it doesn't work again, could you try later?")

