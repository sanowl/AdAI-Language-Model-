import streamlit as st
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
from model import AdAI

# Load the vocabulary and model
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Ensure vocab has `itos` and `stoi` methods or create them
if not hasattr(vocab, 'itos'):
    vocab.itos = {i: token for token, i in vocab.__dict__.get('stoi', {}).items()}

if not hasattr(vocab, 'stoi'):
    vocab.stoi = {token: i for i, token in vocab.__dict__.get('itos', {}).items()}

# Ensure special tokens exist
special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
for token in special_tokens:
    if token not in vocab.stoi:
        index = len(vocab.stoi)
        vocab.stoi[token] = index
        vocab.itos[index] = token

vocab_size = len(vocab)
d_model = 512
d_state = 256
d_conv = 3
expand = 4
num_layers = 8

# Load the model
model = AdAI(vocab_size, d_model, d_state, d_conv, expand, num_layers)
model.load_state_dict(torch.load('ad_language_model_best.pth', map_location=torch.device('cpu')))
model.eval()

# Utility functions
def text_pipeline(text):
    tokens = text.split()
    indices = [vocab.stoi.get(token, vocab.stoi['<unk>']) for token in tokens]
    return torch.tensor(indices, dtype=torch.long)

def generate_response(input_text):
    input_tensor = text_pipeline(input_text).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    response_indices = output.argmax(dim=-1).squeeze().tolist()
    if isinstance(response_indices, int):
        response_indices = [response_indices]
    response_text = ' '.join([vocab.itos.get(idx, '<unk>') for idx in response_indices if idx != vocab.stoi['<pad>']])
    return response_text

# Streamlit app
st.title("AdAI Chat Interface")
st.write("Welcome to the AdAI chat interface. Ask your questions and get responses!")

if 'history' not in st.session_state:
    st.session_state.history = []

# Input and response
user_input = st.text_input("You:", "")
if user_input:
    response_text = generate_response(user_input)
    st.session_state.history.append((user_input, response_text))

# Display chat history
for user_input, response in st.session_state.history:
    st.write(f"**You:** {user_input}")
    st.write(f"**AdAI:** {response}")
