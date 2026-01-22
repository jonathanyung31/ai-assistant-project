import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

Path('visuals').mkdir(exist_ok=True)

try:
    df = pd.read_csv(r'data/books.csv', on_bad_lines='skip')
except FileNotFoundError:
    print('Error: File not Found!')
    exit()

df.columns = df.columns.str.strip()

print("Explanatory Data Analysis - Goodreads Books Dataset")

print("\n Missing Values Analysis")

# Check for 0 values
print("\nZero values count:")
print(df.select_dtypes(include='number').eq(0).sum())

print("\nNaN/Null values count:")
print(df.isnull().sum())


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

# Heatmap of zero values Before Cleaning
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include='number').eq(0).sum().to_frame(), 
            annot=True, fmt='d', cbar=False, cmap='coolwarm')
plt.title('Zero Values Heatmap Before Cleaning')
plt.tight_layout()
plt.savefig('visuals/zero_values_heatmap.png')
print("Saved: visuals/zero_values_heatmap.png")
plt.close()

print("\n Statistical Analysis Before Cleaning")

# 5-number summary for numerical columns
print("\n5-Number Summary Before Cleaning:")
print(df[['num_pages', 'average_rating', 'ratings_count']].describe())

num_cols = ['num_pages', 'average_rating', 'ratings_count']
for column in num_cols:
    print(f"\n{column}:")
    print(f"  Mean: {df[column].mean():.2f}")
    print(f"  Median: {df[column].median():.2f}")
    print(f"  Std Dev: {df[column].std():.2f}")
    print(f"  Skewness: {df[column].skew():.2f}")
    print(f"  Kurtosis: {df[column].kurtosis():.2f}")

print("\n Outlier Detection")

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect Outlier for each column
for column in ['num_pages', 'ratings_count', 'average_rating']:
    outliers, lower, upper = detect_outliers_iqr(df[df[column] != 0], column)
    print(f"\n{column}:")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")
    print(f"Number of outliers: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}%)")
    if len(outliers) > 0:
        print(f"Min outlier value: {outliers[column].min():.2f}")
        print(f"Max outlier value: {outliers[column].max():.2f}")

# Boxplot and zero's removal for Outlier Visualization
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
df_no_zeros = df[df['num_pages'] != 0]
df_no_zeros = df_no_zeros[df_no_zeros['ratings_count'] != 0]
df_no_zeros = df_no_zeros[df_no_zeros['average_rating'] != 0]

axes[0].boxplot(df_no_zeros['num_pages'])
axes[0].set_title('num_pages Boxplot')
axes[0].set_ylabel('Pages')

axes[1].boxplot(df_no_zeros['ratings_count'])
axes[1].set_title('ratings_count Boxplot')
axes[1].set_ylabel('Count')

axes[2].boxplot(df_no_zeros['average_rating'])
axes[2].set_title('average_rating Boxplot')
axes[2].set_ylabel('Rating')

plt.tight_layout()
plt.savefig('visuals/outlier_boxplots.png')
plt.close()

print("\n Distribution Analysis")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['average_rating'], bins=20, edgecolor='black', color='skyblue')
plt.title('Average Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.axvline(df['average_rating'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['average_rating'].median(), color='green', linestyle='--', label='Median')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(df[df['num_pages'] != 0]['num_pages'], bins=30, edgecolor='black', color='lightcoral')
plt.title('Number of Pages Distribution')
plt.xlabel('Pages')
plt.ylabel('Frequency')
plt.axvline(df[df['num_pages'] != 0]['num_pages'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df[df['num_pages'] != 0]['num_pages'].median(), color='green', linestyle='--', label='Median')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(df[df['ratings_count'] != 0]['ratings_count'], bins=30, edgecolor='black', color='lightgreen')
plt.title('Ratings Count Distribution')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.axvline(df[df['ratings_count'] != 0]['ratings_count'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df[df['ratings_count'] != 0]['ratings_count'].median(), color='green', linestyle='--', label='Median')
plt.legend()

plt.tight_layout()
plt.savefig('visuals/before_cleaning_distributions.png')
plt.close()

print("\n Correlation Analysis")

correlation_matrix = df[['num_pages', 'average_rating', 'ratings_count', 'text_reviews_count']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap - Numerical Features')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png')
plt.close()

print("\n Categorial Features Analysis")

category_coloumns = ['language_code', 'publisher', 'authors']

for coloum in category_coloumns:
    print(f"\n{coloum}:")
    print(f"Unique values: {df[column].nunique()}")
    print(f"Top 5 most common:")
    print(df[column].value_counts().head())

# Language attribute heatmap
plt.figure(figsize=(12, 6))
language_counts = df['language_code'].value_counts().head(10)
plt.bar(language_counts.index, language_counts.values, color='steelblue')
plt.title('Top 10 Languages in Dataset')
plt.xlabel('Language Code')
plt.ylabel('Number of Books')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visuals/language_distribution.png')
plt.close()


print("\n Relationship Analysis")

figure, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(df['num_pages'], df['average_rating'], alpha=0.3, s=10)
axes[0].set_xlabel('Number of Pages')
axes[0].set_ylabel('Average Rating')
axes[0].set_title('Pages vs Rating')
axes[0].set_xlim(0, 1000)

axes[1].scatter(df['ratings_count'], df['average_rating'], alpha=0.3, s=10)
axes[1].set_xlabel('Ratings Count')
axes[1].set_ylabel('Average Rating')
axes[1].set_title('Popularity vs Rating')
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('visuals/feature_relationships.png')
plt.close()


print("\n After Cleaning Comparison")

try:
    df_cleaned = pd.read_csv('data/books_copy.csv')
    
    print("\nDataset size comparison:")
    print(f"  Before: {len(df)} rows")
    print(f"  After: {len(df_cleaned)} rows")
    print(f"  Rows removed: {len(df) - len(df_cleaned)} ({((len(df) - len(df_cleaned))/len(df)*100):.2f}%)")
    
    print("\n5-Number Summary AFTER Cleaning:")
    print(df_cleaned[['num_pages', 'average_rating', 'ratings_count']].describe())
    
    # Heatmap After cleaning
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_cleaned.select_dtypes(include='number').eq(0).sum().to_frame(), 
                annot=True, fmt='d', cbar=False, cmap='coolwarm')
    plt.title('Zero Values Heatmap After Cleaning')
    plt.ylabel('Columns')
    plt.xlabel('Count of Zero Values')
    plt.tight_layout()
    plt.savefig('visuals/zero_values_heatmap_after.png')
    print("\nSaved: visuals/zero_values_heatmap_after.png")
    plt.close()

except FileNotFoundError:
    print("\nCleaned data not found. Run main.py first to generate books_copy.csv")
