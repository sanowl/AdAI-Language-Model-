import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("Spacy model loaded successfully!")
except IOError as e:
    print(f"Error loading Spacy model: {e}")
