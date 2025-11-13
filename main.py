import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns


try:
    df = pd.read_csv(r"C:\Users\jonat\Projects\Assistance Systems\ai-assistant-project-jonathan\books.csv", on_bad_lines='skip')
except FileNotFoundError:
    print('Error: File not Found!')

df.columns = df.columns.str.strip()
df = df.drop(columns=['text_reviews_count'])

'''
## Seeing how many 0 values I have in the dataset
print(df.select_dtypes(include='number').eq(0).sum())
## Representing in a visual diagram
plt.figure(figsize=(5, 4))
sns.heatmap(df.select_dtypes(include='number').eq(0).sum().to_frame(), annot=True,
fmt='d', cbar=False, cmap='coolwarm')
plt.title('0 Values Heatmap Before Cleaning')
plt.tight_layout()
plt.show()
'''

# Replacing all 0 values of 'num_pages' column with the median
df['num_pages'] = df['num_pages'].replace(0, df['num_pages'].median())

# Dropping rows with no average_rating (0) by using a boolean mask
df = df[df['average_rating'] != 0]

# Dropping rows with no rating_count (0) by using boolean mask
df = df[df['ratings_count'] != 0]

df.to_csv('books_copy.csv', index=False)

# print(df.select_dtypes(include='number').eq(0).sum())

#Linear Regression
y = df['average_rating']
X = df[['num_pages', 'ratings_count']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
predicted_y = lin_reg_model.predict(X_test)

print("Coefficients:", lin_reg_model.coef_)
print("Intercept:", lin_reg_model.intercept_)


mse = mean_squared_error(y_test, predicted_y)
r2 = r2_score(y_test, predicted_y)

print("Mean Squared Error:", mse)
print("R² score:", r2)

# Classification 

df['rating_catagory'] = pd.cut(df['average_rating'], 
bins = [0, 1.7, 3.4, 5], labels = ['Low', 'Medium', 'High'])

y_class = df['rating_catagory']
X_class = df[['num_pages', 'ratings_count']]

X_train_class, X_test_class, y_train_class,y_test_calss = train_test_split(X_class,
y_class, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_class, y_train_class)

predicted_y_class = rf_model.predict(X_test_class)

accuracy_log_reg = accuracy_score(y_test_calss, predicted_y_class)

print(f"Random Forest Accuracy: {accuracy_log_reg:.2f}")
'''
st.title('Goodreds Recommansation App')
st.write("This App will Recommand you Books you might Like!")
'''
# X = pd.get_dummies(df[['num_pages', 'ratings_count', 'language_code', 'publisher']], drop_first=True)
