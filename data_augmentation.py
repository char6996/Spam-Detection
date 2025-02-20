# data_augmentation.py
import random
import spacy
from nltk.corpus import wordnet
import openai

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

openai.api_key = "your_openai_api_key"  # Your GPT-3 API key

def synonym_augmentation(text):
    """
    Augment text by replacing words with their synonyms.
    """
    doc = nlp(text)
    augmented_text = []
    
    for token in doc:
        # Replace word with a synonym if it is a noun, verb, or adjective
        if token.pos_ in ["ADJ", "VERB", "NOUN"]:
            synonyms = set()
            for syn in wordnet.synsets(token.text):
                for lemma in syn.lemmas():
                    if lemma.name() != token.text:
                        synonyms.add(lemma.name())
            if synonyms:
                # Randomly select a synonym from the set of synonyms
                augmented_text.append(random.choice(list(synonyms)))
            else:
                augmented_text.append(token.text)
        else:
            augmented_text.append(token.text)
    
    return " ".join(augmented_text)

def generate_text_example(prompt):
    """
    Generate new text using GPT-3 (OpenAI).
    """
    response = openai.Completion.create(
        model="text-davinci-003",  # Or another GPT model
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def augment_dataset(texts):
    """
    Augment the dataset by applying synonym replacement and GPT-3 text generation.
    """
    augmented_texts = [synonym_augmentation(text) for text in texts]
    
    # Optionally: Generate additional examples using GPT-3
    for text in texts:
        new_example = generate_text_example(f"Generate a sentence similar to: '{text}'")
        augmented_texts.append(new_example)
    
    return augmented_texts
