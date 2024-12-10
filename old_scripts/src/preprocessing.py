# src/preprocessing.py
import re

def preprocess_text(text):
    """
    Preprocess the text by normalizing case, removing punctuation, and extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text