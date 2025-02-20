# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Step 2: Load and Explore the Dataset
# Load the preprocessed data (from the merged dataset)
df = pd.read_csv("data/expanded_data.csv")

# Check the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())

# Check for missing values and data types to ensure data integrity
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nData types of each column:")
print(df.dtypes)

# Step 3: Split the Data into Training and Testing Sets
# Features (processed text messages) and target variable (labels)
X = df["processed_message"]  # Features: The preprocessed messages
y = df["label"]  # Target: The label (e.g., spam or not)

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the sizes of the training and testing sets
print(f"\nTraining data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")

# Step 4: Text Vectorization (TF-IDF)
# TF-IDF Vectorization to convert text into numerical format
vectorizer = TfidfVectorizer(max_features=5000)  # Limit the features to 5000 most important words

# Fit the vectorizer on the training data and transform both train and test data
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit on training data and transform
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data based on the fitted model

# Check the shape of the transformed data
print(f"\nShape of training data (TF-IDF): {X_train_tfidf.shape}")
print(f"Shape of test data (TF-IDF): {X_test_tfidf.shape}")

# Step 5: Build Machine Learning Models
# Initialize models
logreg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = MultinomialNB()

# Train the models on the training data
logreg.fit(X_train_tfidf, y_train)
rf.fit(X_train_tfidf, y_train)
nb.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the Models
# Make predictions on the test data
y_pred_logreg = logreg.predict(X_test_tfidf)
y_pred_rf = rf.predict(X_test_tfidf)
y_pred_nb = nb.predict(X_test_tfidf)

# Print classification reports for each model
print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_logreg))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))

print("\nNaive Bayes:")
print(classification_report(y_test, y_pred_nb))
