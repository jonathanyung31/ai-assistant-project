import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import random

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path('visuals').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# Load data
try:
    df = pd.read_csv(r'data/books.csv', on_bad_lines='skip')
except FileNotFoundError:
    print('Error: File not Found!')
    exit()

df.columns = df.columns.str.strip()

print("\n--- Statistical Analysis Before Cleaning ---")

# 5-number summary for numerical columns
print("\n5-Number Summary Before Cleaning:")
print(df[['num_pages', 'average_rating', 'ratings_count']].describe())

# Check for 0 values
print("\nZero values count:")
print(df.select_dtypes(include='number').eq(0).sum())

plt.figure(figsize=(12, 5))

# Histogram for average_rating
plt.subplot(1, 3, 1)
plt.hist(df['average_rating'], bins=20, edgecolor='black')
plt.title('Average Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

# Histogram for num_pages
plt.subplot(1, 3, 2)
plt.hist(df['num_pages'], bins=30, edgecolor='black')
plt.title('Number of Pages Distribution')
plt.xlabel('Pages')
plt.ylabel('Frequency')

# Histogram for ratings_count
plt.subplot(1, 3, 3)
plt.hist(df['ratings_count'], bins=30, edgecolor='black')
plt.title('Ratings Count Distribution')
plt.xlabel('Count')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('visuals/before_cleaning_distributions.png')
print("\n Saved: visuals/before_cleaning_distributions.png")
plt.close()

# Heatmap of zero values
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').eq(0).sum().to_frame(), 
            annot=True, fmt='d', cbar=False, cmap='coolwarm')
plt.title('Zero Values Heatmap Before Cleaning')
plt.tight_layout()
plt.savefig('visuals/zero_values_heatmap.png')
print("Saved: visuals/zero_values_heatmap.png")
plt.close()

# Data cleaning
df = df.drop_duplicates()
df = df.drop(columns=['text_reviews_count'])

# Replacing all 0 values of 'num_pages' column with the median
df['num_pages'] = df['num_pages'].replace(0, df['num_pages'].median())

# Dropping rows with no attribute value (0) by using a boolean mask
df = df[df['average_rating'] != 0]
df = df[df['ratings_count'] != 0]

# Transforming attribute values to fit models
label_encoder_language = LabelEncoder()
df['language_code_encoded'] = label_encoder_language.fit_transform(df['language_code'])

df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df['book_age_days'] = (datetime.now() - df['publication_date']).dt.days
df.drop(columns=['publication_date'], inplace=True)

label_encoder_author = LabelEncoder()
df['author_encoded'] = label_encoder_author.fit_transform(df['authors'])

df = df.dropna(subset=['book_age_days'])

''' print(df.select_dtypes(include='number').eq(0).sum()) '''

# Linear Regression Model

y_reg = df['average_rating']
X_reg = df[['num_pages', 'ratings_count', 'book_age_days']]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_reg, y_train_reg)
predicted_rating = lin_reg_model.predict(X_test_reg)

# Calculating Regression Matrices

mse = mean_squared_error(y_test_reg, predicted_rating)
r2 = r2_score(y_test_reg, predicted_rating)

print("\n--- Linear Regression Model ---")
print(f"Coefficients: {lin_reg_model.coef_}")
print(f"Intercept: {lin_reg_model.intercept_}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Random Forest Classification

df['rating_category'] = pd.cut(
    df['average_rating'],
    bins=[0, 1.7, 3.4, 5],
    labels=['Low', 'Medium', 'High']
)

df = df.dropna(subset=['rating_category'])

# Now we can save it with all attributes
df.to_csv('data/books_copy.csv', index=False)

y_class = df['rating_category']
X_class = df[['num_pages', 'ratings_count', 'book_age_days',
               'language_code_encoded', 'author_encoded']]

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_class, y_train_class)

predicted_y_class = rf_model.predict(X_test_class)

accuracy_rf = accuracy_score(y_test_class, predicted_y_class)

print(f"\n--- Random Forest Classifier ---")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")


# Dumping to Joblib for use in Streamlit app

joblib.dump(rf_model, 'models/book_rf_model.joblib')
joblib.dump(list(X_train_class.columns), "models/book_rf_features.joblib")
joblib.dump(lin_reg_model, 'models/book_lin_model.joblib')
joblib.dump(list(X_train_reg.columns), "models/book_lin_features.joblib")

# Save label encoders if the user will ask for a new book in chatbot
joblib.dump(label_encoder_language, 'models/language_encoder.joblib')
joblib.dump(label_encoder_author, 'models/author_encoder.joblib')

# Generating Fake Data

num_real_books = len(df)
num_fake_books = int(num_real_books * 0.3 / 0.7)

# Random values will be consistent
np.random.seed(42)
random.seed(42)

authors = df['authors'].unique()[:100]
real_languages = df['language_code'].unique()
real_publishers = df['publisher'].unique()[:100]

fake_data = {
    'bookID': range(100000, 100000 + num_fake_books),
    'title': [f'Fake Book Title {i}' for i in range(num_fake_books)],
    'authors': np.random.choice(authors, num_fake_books),
    'average_rating': np.round(np.random.uniform(2.5, 5.0, num_fake_books), 2),
    'isbn': [str(random.randint(1000000000, 9999999999)) for i in range(num_fake_books)],
    'isbn13': [str(random.randint(1000000000000, 9999999999999)) for i in range(num_fake_books)],
    'language_code': np.random.choice(real_languages, num_fake_books),
    'num_pages': np.random.randint(100, 1000, num_fake_books),
    'ratings_count': np.random.randint(50, 300000, num_fake_books),
    'publisher': np.random.choice(real_publishers, num_fake_books),
    'book_age_days': np.round(np.random.uniform(365, 20000, num_fake_books), 1)
}

fake_df = pd.DataFrame(fake_data)

# Adding rest of the attributes that were created from previous ones

fake_label_encoder_language = LabelEncoder()
fake_df['language_code_encoded'] = fake_label_encoder_language.fit_transform(fake_df['language_code'])

fake_label_encoder_author = LabelEncoder()
fake_df['author_encoded'] = fake_label_encoder_author.fit_transform(fake_df['authors'])

fake_df['rating_category'] = pd.cut(
    fake_df['average_rating'],
    bins=[0, 1.7, 3.4, 5],
    labels=['Low', 'Medium', 'High']
)

fake_df = fake_df.dropna(subset=['rating_category'])

# Reordering columns for easier comparison with real data columns
fake_df = fake_df[df.columns]

fake_df.to_csv('data/books_fake.csv', index=False)

# --- Creating 30% fake data with 70% real data Dataset ---
combined_df = pd.concat([df, fake_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv('data/books_combined_fake.csv', index=False)
# --- Training Models with Fake Data ---

# Fake Data Linear Regression Model

y_lin_fake = combined_df['average_rating']
X_lin_fake = combined_df[['num_pages', 'ratings_count', 'book_age_days']]

X_train_lin_fake, X_test_lin_fake, y_train_lin_fake, y_test_lin_fake = train_test_split(
    X_lin_fake, y_lin_fake, test_size=0.3, random_state=42
)

lin_model_fake = LinearRegression()
lin_model_fake.fit(X_train_lin_fake, y_train_lin_fake)
predicted_rating_fake = lin_model_fake.predict(X_test_lin_fake)

mse_fake = mean_squared_error(y_test_lin_fake, predicted_rating_fake)
r2_fake = r2_score(y_test_lin_fake, predicted_rating_fake)

print("\n--- Linear Regression Model with Fake Data ---")
print(f"Coefficients: {lin_model_fake.coef_}")
print(f"Intercept: {lin_model_fake.intercept_}")
print(f"Mean Squared Error (MSE): {mse_fake:.2f}")
print(f"R-Squared (R²): {r2_fake:.2f}")

# Fake Data Random Forest Classification
y_class_fake = combined_df['rating_category']
X_class_fake = combined_df[['num_pages', 'ratings_count', 'book_age_days',
                         'language_code_encoded', 'author_encoded']]

X_train_class_fake, X_test_class_fake, y_train_class_fake, y_test_class_fake = train_test_split(
    X_class_fake, y_class_fake, test_size=0.3, random_state=42
)

rf_model_fake = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_fake.fit(X_train_class_fake, y_train_class_fake)
predicted_y_class_fake = rf_model_fake.predict(X_test_class_fake)

accuracy_rf_fake = accuracy_score(y_test_class_fake, predicted_y_class_fake)

print(f"\n--- Random Forest Classifier using Fake Data ---")
print(f"Random Forest Accuracy: {accuracy_rf_fake:.2f}")

# Dumping Fake Data Models to Joblib files

joblib.dump(rf_model_fake, 'models/book_rf_model_fake.joblib')
joblib.dump(list(X_train_class_fake.columns), 'models/book_rf_features_fake.joblib')
joblib.dump(lin_model_fake, 'models/book_lin_model_fake.joblib')
joblib.dump(list(X_train_lin_fake.columns), 'models/book_lin_features_fake.joblib')

joblib.dump(fake_label_encoder_language, 'models/language_encoder_fake.joblib')
joblib.dump(fake_label_encoder_author, 'models/author_encoder_fake.joblib')
