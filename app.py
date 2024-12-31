import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import os
import pickle
import nltk
from pycocotools.coco import COCO
from collections import Counter
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import io
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .caption-box {
        padding: 1rem;
        background-color: #e6f3ff;
        border-radius: 5px;
        margin: 1rem 0;
        color: #ff4b4b; /* Change this to your desired color */
    }
    </style>
    """, unsafe_allow_html=True)

# Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vocabulary Class
class Vocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0
 
    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
 
    def __len__(self):
        return len(self.w2i)
    
    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1

# Load vocabulary
with open('models/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# Model Definitions
class CNNModel1(nn.Module):
    def __init__(self, embedding_size):
        super(CNNModel1, self).__init__()
        resnet = models.resnet34(weights=True)
        module_list = list(resnet.children())[:-1]
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)
        
    def forward(self, input_images):
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features

class LSTMModel1(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        super(LSTMModel1, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len
        
    def forward(self, input_features, capts, lens):
        embeddings = self.embedding_layer(capts)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True) 
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs
    
    def sample(self, input_features, lstm_states=None):
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs, lstm_states)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))
            _, predicted_outputs = model_outputs.max(1)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)
            lstm_inputs = lstm_inputs.unsqueeze(1)
        sampled_indices = torch.stack(sampled_indices, 1)
        return sampled_indices

# Load models
@st.cache_resource
def load_models():
    encoder_model = CNNModel1(256).eval()
    decoder_model = LSTMModel1(256, 512, len(vocabulary), 1)
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    
    encoder_model.load_state_dict(torch.load('models/encoder-5-3200.ckpt', map_location=device))
    decoder_model.load_state_dict(torch.load('models/decoder-5-3200.ckpt', map_location=device))
    return encoder_model, decoder_model

encoder_model, decoder_model = load_models()

def load_image(image_file, transform=None):
    if isinstance(image_file, str):
        img = Image.open(image_file).convert('RGB')
    else:
        img = Image.open(image_file).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        img = transform(img).unsqueeze(0)
    return img

def generate_caption(image_file):
    img = load_image(image_file, transform)
    img_tensor = img.to(device)
    
    start_time = time.time()
    
    with torch.no_grad():
        feat = encoder_model(img_tensor)
        sampled_indices = decoder_model.sample(feat)
    
    sampled_indices = sampled_indices[0].cpu().numpy()
    
    predicted_caption = []
    for token_index in sampled_indices:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
    
    predicted_sentence = ' '.join([word for word in predicted_caption if word not in {'<start>', '<end>'}])
    processing_time = time.time() - start_time
    
    return predicted_sentence, processing_time

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Model Settings")
    st.write(f"Running on: {device}")
    
    # Theme selector
    theme = st.select_slider(
        "Choose Theme",
        options=["Light", "Dark", "Custom"],
        value="Light"
    )
    
    if theme == "Custom":
        primary_color = st.color_picker("Choose primary color", "#ff4b4b")

# Main content
st.title("🤖 AI Image Captioning")
st.markdown("Transform your images into descriptive captions using AI")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Generate Caption", "History", "About"])

with tab1:
    # Input method selection
    st.write("Choose your input method:")
    input_method = st.radio("", ["Upload Image", "Capture from Camera"])
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = uploaded_file
    else:
        input_method == "Capture from Camera"
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image = camera_input

    if image is not None:
        # Display image and generate caption
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(Image.open(image), caption="Input Image", use_column_width=True)
        
        with col2:
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    caption, processing_time = generate_caption(image)
                    
                    # Display metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with metrics_col2:
                        st.metric("Caption Length", len(caption.split()))
                    
                    # Display caption
                    st.markdown(f"""
                        ### Generated Caption
                        <div class='caption-box'>
                            {caption}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.history.append({
                        'caption': caption,
                        'timestamp': datetime.now(),
                        'processing_time': processing_time
                    })
                    


with tab2:
    # Display history
    st.title("📜 Caption History")
    if len(st.session_state.history) > 0:
        for item in reversed(st.session_state.history):
            with st.expander(f"Caption from {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write(f"Caption: {item['caption']}")
                st.write(f"Processing Time: {item['processing_time']:.2f}s")
    else:
        st.info("No captions generated yet!")

with tab3:
    # About section
    st.title("ℹ️ About")
    st.write("This is an AI Image Captioning application built using Streamlit.")