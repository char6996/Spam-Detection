import pandas as pd
import re
import spacy
from nltk.corpus import stopwords

# Load the expanded dataset
df = pd.read_csv("data/expanded_data.csv")

# Load SpaCy English model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Download stopwords if needed (this is for NLTK)
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, and non-alphanumeric characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters

    # Tokenize and remove stopwords
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_ for token in doc if token.text not in stop_words])
    
    return cleaned_text

# Apply the preprocessing function to the 'message' column
df['processed_message'] = df['message'].apply(preprocess_text)

# Save the preprocessed dataset
df.to_csv("data/processed_data.csv", index=False)

print("Text preprocessing completed! The processed data is saved.")
