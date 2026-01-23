Yung, Jonathan, 22402613

Book Recommander

https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan.git

https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan/-/wikis/home

## Project description

In this project my goal was to create a Book Assistant System that can predict top rated books that the user might like based on their inputs in different ML models operating with different data and a chatbot that interacts with text and voice with the user.

## Installation

To set up the Book Assistant System locally, make sure that your environment meets the following specs the steps indicated below.

#### Requirements: 

- Python version: 3.13.3
- Pandas version: 2.3.3
- NumPy version: 2.3.4
- Scikit-learn version: 1.7.2
- Streamlit version: 1.51.0
- SpeechRecognition version: 3.14.3
- joblib: 1.5.2
- nltk: 3.9.2
- gTTS: 2.5.4

#### Setup Instructions

Clone the Repository: Download the project files to locally by cloning the repository.

Install Necessary Libraries: Install all required libraries using the requirements.txt file generated from your environment.

run this command in a terminal you opened: 
`pip install -r requirements.txt`

Download NLTK Corpora: The chatbot relies on the 'punkt' tokenizer to process the user input.

run this command in a terminal you opened: 
`import nltk`
`nltk.download('punkt')`

Verify Model Files: In case the models/ folder doesn't contain the trained .joblib files run the training scripts first(main.py and chatbot_logic\intent_classifier.py) to generate them. The models nor the chatbot pages will work before these joblib files exist.

## Data

The dataset the project uses is taken from kaggle and is a GoodReads dataset (A well known book site), link can be found here:

https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks

Data description can be found at: https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan/-/wikis/home/Data

The dataset contains a single csv file including 12 column attributes for each book.

#### Data Cleaning & Preprocessing

To make sure there aren't any errors or noise that hinder correct results of the ML models and chatbot, the dataset contant was searched through to find missing values of any kind and deal with them, wether by deleting the entire row containing them or altering their values into a non missing value.

- Handling Missing Values: `text_reviews_count`: dropped, `num_pages`: zeros replaced with median, `average_rating`: zeros removed, `ratings_count`: zeros removed

- Label Encoding: `author_encoded` and `language_code_encoded` are attributes that were created by implementing Label Encoding on existing attribute `authors` and `language_code` to convert them from categorical text attributes into numeric values.

- Transformation was implemented using StandardScaler to ensure attributes have zero mean and unit variance.

- Duplicates rows were removed.

#### Fake Data Generation

combined (fake + real) data was generated using 70% real data and 30% synthetic fake data that resembles the real data.

Same processes that were implemented on the real data indicated before, was as well implemented on the combined (fake + real) dataset.

#### Training & Testing

After cleaning, label encoding and transforming, the complete dataset was split into 70% training data and 30% testing data in main.py to estimate the performance of the predictions from the different models.

Same exact steps were taken for combined (fake + real) dataset in main.py.

The detailed description about the data and the processes performed can be found in the 'Data' wiki page on gitlab.

## Basic Usage

Link on Youtube for Screencast of the App:
https://youtu.be/Jr07qMFelVs

### Models Information

The detailed description about the models and how they were used can be found in the 'Models' wiki page on gitlab.

The system uses three different machine learning models to provide useful results and support a conversational interface.

#### Regression Model

This model is used to predict a user's preferred book details.
Linear Regression is the algorithm that was used here.

The features that are used by the model are number of pages, ratings count and the age of the book.

Pipeline: Data preprocessing steps, including feature scaling, are applied using Scikit-Learn tools before training the model.

#### Classification Model

This model is used to predict the rating category of a book.
Random Forest Classification is the algorithm that was used here.

The features that are used by the model are number of pages, ratings count, age of the book, language code and author after being encoded as input features.

For Preprocessing, Categorical variables like authors and language_code are converted into numerical values using label encoding, and all features are scaled to improve model performance.

The target of the model is to predict the rating category. Can either be Low, Medium, or High.

#### NLP Intent Classifier (Chatbot Logic)

This model helps the chatbot understand what the user wants when they type or speak a message.

The algorithm that was used here is NLTK-based Naive Bayes Classifier.

The functionality of It identifies the intent of the user’s input, such as asking for top-rated books, searching by author, getting book details, or small talk (hello or joke intent), so the chatbot can respond appropriately.

The user input is processed by being tokenized into words, and features are extracted from these tokens to classify the message into one of the predefined intents.

### Streamlit Pages

The application is divided into multiple pages to provide a complete experience for exploring and interacting with book data.

- Welcome Page: Serves as the introduction to the project, providing an overview, instructions, and guidance on how to navigate the other pages.

- Real Data Representation Page: Displays the original cleaned dataset, allowing users to explore statistics and summaries of real book data.

- Combined Fake Data Page: Shows the dataset that combines real and fake books (30% fake + 70% real), giving an overview of how the augmented data looks and its statistical properties and summaries.

- Linear Regression – Real Data: Using the predictions of linear regression model the user can receive top rated book results predicted upon his inputs for specific features.

- Linear Regression – Combined Fake Data: Similar to the above page, but uses the Linear Regression model trained on the combined real and fake dataset. It allows comparison of model results between real dataset and augmented one.

- Classification – Real Data: Displays results of top rated books by their rating categories (Low, Medium, High) using the Random Forest classifier trained on real data.

- Classification – Combined Fake Data: Same as the classification page but here with augmented data. Just like with linear regression, here as well the user can compare between the use of real and augmented datasets for their results.

- Chatbot Page: A conversational assistant that helps users search for books by titles, get top rated books, find books by authors, and view book details:

The chat is interactive in a way that Users can type queries or speak to the assistant.

On top of answering by text, the chatbot can Convert speech to text and respond with text to speech.

The chatbot recognizes intents by using a NLTK-based Naive Bayes classifier to understand user requests and respond accurately.

The cahtbot is able to store past interactions so the user can go back to previous requests and answers.

### Usage of the App

0. First of all, if you do not have any joblib files of this project on your local machine, please run main.py and intent_classifier.py first of all. These 2 will evaluate the models and save the updated .joblib files into the models/.

1. Launching the Application

To start the BooCompass Book Recommendation system, navigate to the project’s root directory in your terminal and run:

streamlit run welcome.py

This command opens the main interface. Once the local server is running, the app will automatically launch in your default web browser. Please wait a few seconds until the app loads.

You will find yourself in the welcome page.

2. Navigating in the Application

If you would look at the Streamlit sidebar you will see multiple choices for different pages you can access and switch between. 

- Welcome: Introduction to the project, instructions, and navigation guide.

- Real Data: Explore the cleaned book dataset with statistics and summaries.

- Combined Fake Data: View the augmented dataset with real and synthetic books, statistics and summaries.

- Linear Regression – Real Data: Select your value choices between the different widgets and get top rated books details that match them by the model.

- Linear Regression – Combined Fake Data: Do the same thing here as in the linear regression with real data, but expect to receive some fake books as well. You can compare the results between here and the real data operated one. 

- Classification – Real Data: Put your desired inputs and get top rated books details depending on the classification of your predictive result. 

- Classification – Do the same here as you did with classification with real data, but expect to get fake generated books.

- Chatbot: Here you can search by title or author, get top-rated books, or view book details by using text input or pressing by speaking after pressin the 'speak' button. Supports text and voice input and response.

3. Enjoy your time

Try it out and see what you get, maybe you'll find your next GREAT read!