import streamlit as st
import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import datetime
import base64
import io
from gtts import gTTS

st.set_page_config(page_title="Book Chatbot", page_icon="💬", layout="wide")
st.title("💬 BooCompass - Voice & Chat Enabled AI Book Recommendation Chatbot")
st.write("""Ask me about your preffered books!
         I will help you find top-rated books,
         you can search by author, get book details and more!""")
st.markdown("Type your question below or click 'Speak' to use your voice!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

def extract_features(text):
    features = {}
    words = word_tokenize(text.lower())
    for word in words:
        features[f'has({word})'] = True
    return features


def text_to_audio_for_web1(text):
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text, lang='en', tld='com.au')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data = mp3_fp.read()
        b64 = base64.b64encode(data).decode()
        return f"""
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    except Exception as e:
        st.error(f"TTS Error: Could not generate audio. Details: {e}")
        return None
    

def assistant_action(intent, query_text=""):
    if "joke" in query_text:
        return "What do you call a fake noodle? An impasta!"
    if "time" in query_text:
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."
    if "hello" in query_text or "hi" in query_text:
        return "Hello! I'm your book assistant. How can I help you find a great read today?"
    if "bye" in query_text or "goodbye" in query_text:
        return "Goodbye! Happy reading!"
    
    actions = {
        "top_rated_books": "Here are the top rated books you asked for!",
        "get_books_by_author": "I found some great books by that author.",
        "search_by_title": "I found the book title you were looking for.",
        "get_details": "I've pulled up the full details for that book."
    }
    
    return actions.get(intent, "I'm sorry, I don't recognize that command. Try asking for a book title or an author!")
    

@st.cache_resource
def load_intent_model():
    try:
        classifier = joblib.load('models/book_intent_model.joblib')
        return classifier
    except FileNotFoundError:
        st.error("book_intent_model.joblib not found. Please run intent_classifier.py first.")
        st.stop()

classifier = load_intent_model()

@st.cache_data
def load_book():
    try:
        return pd.read_csv("data/books_copy.csv")
    except FileNotFoundError:
        st.error("Data file not found.")

df = load_book()


def top_rated_books(limit=10):
    #Get the highest rated books
    top_books = df.nlargest(limit, 'average_rating')
    return top_books[['title', 'authors', 'average_rating']].to_dict('records')

def get_books_by_author(author_name, limit=10):
    #Get books by a specific author
    author_books = df[df['authors'].str.contains(author_name, case=False, na=False)]
    author_books = author_books.nlargest(limit, 'average_rating')
    return author_books[['title', 'authors', 'average_rating']].to_dict('records')

def search_by_title(title):
    #Find a specific book by title
    result = df[df['title'].str.contains(title, case=False, na=False)]
    if len(result) > 0:
        return result.iloc[0].to_dict()
    return None

def get_details(title):
    #Get full details about a book
    book = search_by_title(title)
    return book

# -------------------------------------------------------------
# Voice Assistant

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
            response = "No speech detected. Please try again."
            audio_html = text_to_audio_for_web1(response)
            if audio_html:
                st.components.v1.html(audio_html, height=100)
            return None
        
        except sr.UnknownValueError:
            st.warning("Sorry, I could not understand that command.")
            response = "Sorry, I could not understand that command."
            audio_html = text_to_audio_for_web1(response)
            if audio_html:
                st.components.v1.html(audio_html, height=100)

            return None
        except sr.RequestError as e:
            st.error(f"Speech service error: {e}. Check internet connection.")
            response = "I am having trouble connecting to the speech service."
            audio_html = text_to_audio_for_web1(response)
            if audio_html:
                st.components.v1.html(audio_html, height=100)
            
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            response = "An unexpected error occurred."
            audio_html = text_to_audio_for_web1(response)
            if audio_html:
                st.components.v1.html(audio_html, height=100)
            
            return None
        
    
for id, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "data" in message:
            if isinstance(message["data"], list):
                st.table(pd.DataFrame(message["data"]))
            else:
                st.json(message["data"])
        if "audio" in message:
            is_last_assistant = (id == len(st.session_state.messages) - 1 and message["role"] == "assistant")
            if is_last_assistant:
                st.components.v1.html(message["audio"], height=80)
            else:
                audio_no_autoplay = message["audio"].replace("autoplay", "")
                st.components.v1.html(audio_no_autoplay, height=80)

user_input = st.text_input("Do You Have any Questions about Books?", key=f"user_input_{st.session_state.input_key}")

send, talk, garbage, push = st.columns([1, 1, 1, 6])
with send:
    submit = st.button("📬 Send")
with talk:
    speak = st.button("🎤 Speak")
with garbage: 
    clear = st.button("🗑️ Clear")

st.write("💡 Tips Below:")
st.write("Please write clearly and use proper spelling before clicking on 'Send'")
st.write("Please Remember to speak clearly after clicking on 'Speak'.")
st.write("If you want to get rid of chat history, click on 'clear chat'")

st.markdown("---")

# Handeling Voice Input
if speak:
    voice_cmd = speech_to_text()
    if voice_cmd:
        # Saving to history
        st.session_state.messages.append({"role": "user", "content": voice_cmd})
        
        voice_features = extract_features(voice_cmd)
        voice_intent = classifier.classify(voice_features)

        res_table = None
        res_json = None

        if voice_intent == "top_rated_books":
            results = top_rated_books(5)
            res_table = results
            response_msg = assistant_action(voice_intent, voice_cmd)
        
        elif voice_intent == "get_books_by_author":
            que = voice_cmd.lower()
            for word in ["show", "me", "books", "by", "author",
                        "find", "get", "top", "rated"]:
                que = que.replace(word, "")
            author_name = que.strip()
            if author_name:
                results = get_books_by_author(author_name, limit=3)
                if results:
                    res_table = results
                    response_msg = f"I found top rated books by {author_name}. Here they are:"
                else:
                    response_msg = f"Sorry, I couldn't find any books by {author_name}."
            else:
                response_msg = "Which author would you like top rated books from?"
        
        elif voice_intent == "search_by_title" or voice_intent == "get_details":
            que = voice_cmd.lower()
            for word in ["find", "search", "for",
                         "details", "about", "book", "the", "title"]:
                que = que.replace(word, "")
            book_title = que.strip()
            result = search_by_title(book_title)
            if result:
                res_json = result
                response_msg = f"I found the details for {result['title']}"
            else:
                response_msg = "I couldn't find that book in my records."
        
        else:
            response_msg = assistant_action(voice_intent, voice_cmd)
        
        history_msg = {"role": "assistant", "content": response_msg}
        if res_table is not None: 
            history_msg["data"] = res_table
        if res_json is not None: 
            history_msg["data"] = res_json

        audio_html = text_to_audio_for_web1(response_msg)
        if audio_html:
            history_msg["audio"] = audio_html

        st.session_state.messages.append(history_msg)
        
        st.rerun()

if clear:
    st.session_state.messages = []
    st.session_state.input_key += 1
    st.rerun()

# ------------------------------------------------------------    

# Handeling user text input
if user_input or submit:
    if user_input:
        
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
        else:
            last_msg = None

        if last_msg and last_msg.get("role") == "user" and last_msg.get("content") == user_input:
            duplicate = True
        else:
            duplicate = False

        if not duplicate:
            st.session_state.messages.append({"role": "user", "content": user_input})

        features = extract_features(user_input)
        intent = classifier.classify(features)

        if "by" in user_input.lower() or "from" in user_input.lower():
            intent = "get_books_by_author"

        res_table = None
        res_json = None
        response = ""

        if intent == "top_rated_books":
            results = top_rated_books(5)
            res_table = results
            response = "Here are the top rated books you asked for"

        elif intent == "get_books_by_author":
            que = user_input.lower()
            for word in ["show", "me", "books", "by", "author",
                        "find", "get", "top", "rated"]:
                que = que.replace(word, "")
            author_name = que.strip()
            if author_name:
                results = get_books_by_author(author_name, limit=5)
                if results:
                    res_table = results
                    response = f"I found top rated books by {author_name}. Here they are:"
                
                else:
                    response = f"Sorry, I couldn't find any books by {author_name}."
            else:
                response = "Which author would you like top rated books from?"
        
        elif intent == "search_by_title" or intent == "get_details":
            que = user_input.lower()
            for word in ["find", "search", "for", "the", "book",
                         "title" , "details", "about"]:
                que = que.replace(word, "")
            book_title = que.strip()
            result = search_by_title(book_title)
            if result:
                res_json = result
                response = assistant_action(intent, user_input)
            else:
                response = "I'm sorry, I couldn't find that book in my dataset."

        else:
            response = assistant_action(intent, user_input)
        
        history_msg = {"role": "assistant", "content": response}
        if res_table is not None: 
            history_msg["data"] = res_table
        if res_json is not None: 
            history_msg["data"] = res_json

        audio_html = text_to_audio_for_web1(response)
        if audio_html:
            history_msg["audio"] = audio_html

        st.session_state.messages.append(history_msg)
        
        st.session_state.input_key += 1
        st.rerun()