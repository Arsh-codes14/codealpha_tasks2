# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load dataset
# Replace 'C:\titanic dataset.csv' with your dataset's file path or URL
df = pd.read_csv(r'C:\titanic dataset.csv')

# Display the first few rows of the dataset
print("Initial DataFrame:\n", df.head())

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# 1. Handling Missing Values

# Impute numeric columns with the mean
imputer_numeric = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])

# Impute categorical columns with the most frequent value
imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

print("DataFrame after handling missing values:\n", df.head())

# 2. Handling Outliers
# Example: Removing outliers using the Z-score method
from scipy.stats import zscore

z_scores = np.abs(zscore(df[numeric_cols]))  # Calculate Z-scores for numeric columns
df_no_outliers = df[(z_scores < 3).all(axis=1)]  # Keep rows where all Z-scores are less than 3

print("DataFrame after removing outliers:\n", df_no_outliers.head())

# 3. Normalization / Scaling
# Example: Standard scaling (z-score scaling)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_no_outliers[numeric_cols]), columns=numeric_cols)

# If you want to include categorical columns back into the dataframe, do so here
df_scaled = pd.concat([df_scaled, df_no_outliers[categorical_cols].reset_index(drop=True)], axis=1)

print("DataFrame after scaling:\n", df_scaled.head())

# 4. Splitting the Data into Training and Testing Sets
# Define your features (X) and target variable (y)
X = df_scaled.drop('Survived', axis=1)  # Replace 'Survived' with the actual target column name if different
y = df_scaled['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(f"Training feature set shape: {X_train.shape}")
print(f"Testing feature set shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")
