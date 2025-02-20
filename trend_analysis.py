import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Load the preprocessed dataset
df = pd.read_csv("data/processed_data.csv")

# Separate spam and non-spam messages
spam_messages = df[df['label'] == 1]['processed_message']
non_spam_messages = df[df['label'] == 0]['processed_message']

# Function to count word frequency
def get_word_frequencies(messages):
    word_counts = Counter()
    for msg in messages.dropna():  # Drop NaN values if any
        word_counts.update(msg.split())
    return word_counts

# Get word frequencies for spam and non-spam messages
spam_word_freq = get_word_frequencies(spam_messages)
non_spam_word_freq = get_word_frequencies(non_spam_messages)

# Convert to DataFrame for visualization
spam_df = pd.DataFrame(spam_word_freq.most_common(20), columns=['Word', 'Frequency'])
non_spam_df = pd.DataFrame(non_spam_word_freq.most_common(20), columns=['Word', 'Frequency'])

# Plot bar charts
plt.figure(figsize=(12, 5))
sns.barplot(x='Frequency', y='Word', data=spam_df, palette='Reds')
plt.title('Top 20 Words in Spam Messages')
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x='Frequency', y='Word', data=non_spam_df, palette='Blues')
plt.title('Top 20 Words in Non-Spam Messages')
plt.show()

# Generate word clouds
spam_wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(spam_word_freq)
non_spam_wordcloud = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(non_spam_word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Spam Messages')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(non_spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Non-Spam Messages')
plt.show()
