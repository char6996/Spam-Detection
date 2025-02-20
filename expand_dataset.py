import pandas as pd

# Load existing dataset
df_existing = pd.read_csv("data/processed_data.csv")

# Load new dataset (SMS Spam Collection) and handle the error
# Option 1: If the file is tab-separated
df_new = pd.read_csv("SMSSpamCollection.csv", delimiter="\t", encoding="latin-1", names=["label", "message"])

# Option 2: If you want to skip malformed lines
# df_new = pd.read_csv("SMSSpamCollection.csv", encoding="latin-1", names=["label", "message"], on_bad_lines='skip')

# Option 3: If you only need the first two columns
# df_new = pd.read_csv("SMSSpamCollection.csv", encoding="latin-1", usecols=[0, 1], names=["label", "message"])

# Preprocess labels
df_new['label'] = df_new['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 (ham) and 1 (spam)

# Merge the datasets
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Save the merged dataset as 'expanded_data.csv'
df_combined.to_csv("data/expanded_data.csv", index=False)

print("Dataset expanded successfully! Total messages:", len(df_combined))
