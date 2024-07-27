import spacy
from collections import Counter
from torchtext.vocab import Vocab
import torch

# Initialize Spacy tokenizer
nlp = spacy.load("en_core_web_sm")

# Function to clean and tokenize text
def clean_and_tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Tokenize and build vocabulary
counter = Counter()
for line in synthetic_ads:
    tokens = clean_and_tokenize(line)
    counter.update(tokens)

vocab_size = 5000  # Define the maximum size of the vocabulary
vocab = Vocab(counter, max_size=vocab_size)

# Save the vocabulary
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
