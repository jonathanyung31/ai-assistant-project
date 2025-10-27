# ai-assistant-project-jonathan

Yung, Jonathan, 22402613


https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan.git

https://mygit.th-deg.de/jy31613/ai-assistant-project-jonathan/-/wikis/home

## Project description
Basics of the Project

## Installation
rerequisites and installation of the project on another
computer. Python, scikit-learn, streamlit are used here.

## Data
Project will use 3 files from the same dataset chosen:
Books.csv, Ratings.csv, Users.csv.

Attributes/Columns from each file and their respective data type:

Books.csv: 
ISBN (int): Book's ISBN (spcial serial number that belongs to specific book)
title (str): Book's Title
author (str): Book's Author
year (int): Book's year of Publication
publisher (str): Publisher of the book
images: won't be used in this Project

Ratings.csv: 
ISBN (int): Book's ISBN
rating (int): rating of a book by a user
user_id (int): ID of users

Users.csv: 
location (str): Location of where the user is from
age (int): User's age
user_id (int): ID of users

Feature Variables:
author: some users prefer certain authors
year: some users like older/newer books
publisher: another feature some users may prefer
age: preference may change by age
location: regional preferences
title: popular/attracting titles

Identifiers for linking features/data:
user_id: linking rating to users
ISBN: linking ratings to books

Task of the System:
Using features and ratings of books to predict and provide personlized recommendation. 

Status/Target Variable: Rating

Analyzing each variable:

author and publication is a feature because users often prefer books by certain authors/publication company, which helps the system predict ratings more accurately.

year is a feature because some users prefer older/newer books (if it is because of the language style or that era's way of thinking/writing).

age: different ages/generations are in most times attracted to different styles of books.

location: regional preferences can impact which books are more popular. Some places in the world are more attracted to certain types of books which others aren't.

title: some books are famous, others attract people, and so might be preferable by some.

## Basic Usage
how to start the project, for example first logins with
passwords (if it exists) and key use cases.
