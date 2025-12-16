import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import json
import random
from collections import defaultdict
import pandas as pd
import pyttsx3
import datetime
import time

#The Use Case the chatbot will be focused on is for the user to see top-rated books


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


# if intent == "top_rated_books":
  #  st.write("Here are the top-rated books...")
#elif intent == "get_books_by_author":
 #   st.write("Which author's top books would you like?")
#elif intent == "search_by_title":
  #  st.write("Searching for that book...")
#elif intent == "get_details":
 #   st.write("Fetching book details...")
#else:
 #   st.write("Sorry, I didn't catch that, could you say that more clearly?")


@st.cache_data
def load_data():
    #Using cache to save time
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

st.set_page_config(layout="centered")
st.markdown("If You Want to Speak Click on the Microphone")

def change_voice(engine, language, gender='VoiceGenderFemale'):
    #Checks if the voice exists on the current Hardware
    for voice in engine.getProperty('voices'):
        if language in voice.languages and gender == voice.gender:
            engine.setProperty('voice', voice.id)
            return True

    raise RuntimeError("Language '{}' for gender '{}' not found".format(language, gender))

def speak(text):
    #Translates Text to Speech by the Engine
    st.session_state.tts_engine.say(text)
    try:
        st.session_state.tts_engine.runAndWait()
    except RuntimeError:
        pass

def speech_to_text():
    #Listens and Transcribe whatever the user have said
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait.")
        rec.adjust_for_ambient_noise(source, duration=1)
        st.success("Listening now! Say your command...")
        try:
            audio = rec.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Recognizing...")
            command = rec.recognize_google(audio).lower() # Convert to lowercase for easier matching
            st.write(f"You said: **{command}**")
            return command
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            speak("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Sorry, I could not understand that command.")
            speak("Sorry, I could not understand that command.")
            return None
        except sr.RequestError as e:
            st.error(f"Speech service error: {e}. Check internet connection.")
            speak("I am having trouble connecting to the speech service.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            speak("An unexpected error occurred.")
            return None

def process_command(command):
    if command is None:
        return

    response = ""
    if "hello" in command or "hi" in command:
        response = "Hello there! How can I help you?"
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%H:%M")
        response = f"The current time is {current_time}."
    elif "joke" in command:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Did you hear about the mathematician who was afraid of negative numbers? He would stop at nothing to avoid them!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        response = random.choice(jokes)
    elif "how are you" in command:
        response = "I'm doing great, thank you for asking!"
    elif "goodbye" in command or "bye" in command:
        response = "Goodbye! Have a great day."
    else:
        response = "I'm sorry, I don't recognize that command. Can you please try again?"

    st.write(f"Assistant says: **{response}**")
    speak(response)