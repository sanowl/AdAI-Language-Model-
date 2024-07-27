import random
import spacy
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
import torch
import pickle

# Define components of ads
products = ["shoes", "electronics", "fashion products", "items", "books", "gadgets", 
    "toys", "furniture", "clothes", "accessories", "kitchen appliances", 
    "smartphones", "laptops", "sporting goods", "jewelry", "household goods", 
    "beauty products", "health supplements", "fitness equipment", "pet supplies"
]
offers = ["Buy one get one free", "Limited time offer", "Hurry up, sale ends soon", 
    "Amazing discounts available now", "Flash sale: Up to 50% off", 
    "Exclusive offer for new customers", "Summer clearance sale", 
    "Free shipping on orders over $50", "Special discount for members", 
    "Limited stock available", "Weekend special", "Holiday sale", 
    "Black Friday deals", "Cyber Monday discounts", "End of season sale"
]
actions = ["Shop now", "Don't miss out", "Grab yours today", "Check out our deals", 
    "Visit our website", "Order now", "Add to cart", "Buy before it's too late", 
    "Take advantage of this offer", "Get it now", "Hurry, offer ends soon", 
    "Limited time only", "Save big", "Act fast", "Redeem your discount"
]
additional_phrases = ["Best prices guaranteed", "Quality you can trust", "Exclusive deals", 
    "Customer favorite", "Top-rated products", "Shop with confidence", 
    "Satisfaction guaranteed", "Limited edition", "New arrivals", 
    "Bestsellers", "Highly recommended", "Editor's pick", "Special promotion", 
    "Award-winning products", "Eco-friendly options"
]

# Additional structures
structures = [
    "{offer} on all {product}! {action}.",
    "Get {product} with {offer}. {action}!",
    "{action} and enjoy {offer} on {product}!",
    "Don't miss {offer} on {product}. {action} now!",
    "{offer}! {action} to get {product}.",
    "{action}! {offer} on our best {product}.",
    "{additional_phrase}! {offer} on {product}. {action}.",
    "Check out {product} with {offer}. {action}!",
    "Limited offer: {product} with {offer}. {action}.",
    "{additional_phrase} on {product}. {offer}! {action}.",
    "Special deal: {offer} on {product}. {action}."
]

# Generate synthetic ads with diverse structures and additional phrases
def generate_synthetic_ads(num_ads=10000):
    synthetic_ads = []
    for _ in range(num_ads):
        product = random.choice(products)
        offer = random.choice(offers)
        action = random.choice(actions)
        additional_phrase = random.choice(additional_phrases)
        structure = random.choice(structures)
        ad = structure.format(
            offer=offer, product=product, action=action, additional_phrase=additional_phrase
        )
        synthetic_ads.append(ad)
    return synthetic_ads

# Generate 10,000 synthetic ads
synthetic_ads = generate_synthetic_ads(10000)

# Save the generated ads to a file
def save_ads_to_file(ads, filename="ads_data.txt"):
    with open(filename, "w") as file:
        for ad in ads:
            file.write(ad + "\n")

# Save the ads
save_ads_to_file(synthetic_ads, "synthetic_ads.txt")

# Print a few examples
for ad in synthetic_ads[:10]:
    print(ad)

# Initialize Spacy tokenizer
nlp = spacy.load("en_core_web_sm")

# Function to clean and tokenize text
def clean_and_tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Tokenize and build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield clean_and_tokenize(text)

vocab = build_vocab_from_iterator(yield_tokens(synthetic_ads), specials=["<unk>", "<pad>"], max_tokens=5000)
vocab.set_default_index(vocab["<unk>"])

# Save the vocabulary
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Convert text to tensor format
def text_pipeline(text):
    tokens = clean_and_tokenize(text)
    return [vocab[token] for token in tokens]

# Convert all ads to tensors
ad_tensors = [torch.tensor(text_pipeline(ad), dtype=torch.long) for ad in synthetic_ads]

# Save the tensors
torch.save(ad_tensors, 'ad_tensors.pt')
