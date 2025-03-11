import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import io
import numpy as np
import requests
from model import SlimCAE  # Assuming SlimCAE is in a separate file model.py

# Load Pre-trained Model (Modify path if needed)
def load_model():
    model = SlimCAE()
    model_path = "slimcae_pretrained.pth"  # Update path if needed
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load pre-trained model: {e}")
        return None

# Function to compress and decompress an image
def compress_decompress_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to model input size
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        compressed = model.encoder(image_tensor)  # Get compressed representation
        decompressed = model.decoder(compressed)  # Reconstruct image
    
    decompressed_image = decompressed.squeeze(0).permute(1, 2, 0).numpy()  # Convert back to image format
    decompressed_image = (decompressed_image * 255).astype(np.uint8)
    return Image.fromarray(decompressed_image)

# Streamlit App UI
st.title("AI-Based Image Compression with SlimCAE")
model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    
    compressed_image = compress_decompress_image(image, model)
    st.image(compressed_image, caption="Compressed & Reconstructed Image", use_column_width=True)
    
    img_byte_arr = io.BytesIO()
    compressed_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button("Download Compressed Image", img_byte_arr, file_name="compressed.png", mime="image/png")
