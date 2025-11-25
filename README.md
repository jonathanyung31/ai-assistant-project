# ai-assistant-project-jonathan

Yung, Jonathan, 22402613


https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan.git

https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan/-/wikis/home

## Project description

A book recommendation system that will give the user books that he might like.

## Installation
- Python version: 3.13.3
- Pandas version: 2.3.3
- NumPy version: 2.3.4
- Scikit-learn version: 1.7.2
- Streamlit version: 1.50.0
- SpeechRecognition version: 3.14.3
- joblib: 1.5.2

## Data
https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks

Data description can be found at: https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan/-/wikis/home/Data

Cleaning Data: After checking for all missing values datatypes (Null/NaN, empty values and 0 values), I discovered my data set's missing values are only shown as 0's. and so I decided to drop one column that wasn't relevant, another changed all missing values to median, and other 2 columns I dropped all rows with 0 values.

Linear Regression Model: I added a linear regression model to my dataset, split my data into training data (70%) and testing data (30%), and also showed the mse and r**2 score results.

Classifiaction Model: I created a new column (rating_catagory) with bins (Low, Medium, High) on the back off the already existing target variable (average_rating). Split the data into training data (70%) and testing data (30%), I used RandomForestClassifier Model and also showed accuracy results.

## Basic Usage
Explained later on in the project.