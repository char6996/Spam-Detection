import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load SpaCy model (ensure you have downloaded the model using 'python -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

# Load the combined dataset (replace with your correct path)
df = pd.read_csv("data/expanded_data.csv")

# Preprocessing function using SpaCy
def preprocess_text(text):
    # Apply SpaCy NLP pipeline to the text
    doc = nlp(text.lower())  # Lowercase all text
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  # Remove stopwords and keep only alphabetic tokens
    return " ".join(tokens)

# Apply preprocessing to the 'message' column
df["processed_message"] = df["message"].apply(preprocess_text)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["processed_message"])

# Add the processed TF-IDF features to the DataFrame
df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Save the processed data to the 'data' folder (ensure the folder exists)
df.to_csv("data/processed_data.csv", index=False)
df_tfidf.to_csv("data/tfidf_features.csv", index=False)

# Print the processed data and TF-IDF features
print("Processed Data:")
print(df[['message', 'processed_message']])

print("\nTF-IDF Features:")
print(df_tfidf)
