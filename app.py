import streamlit as st
import torch
from transformers import AutoModel
from PIL import Image
import numpy as np
import io

# Load the pretrained AI model
@st.cache_resource
def load_model():
    return AutoModel.from_pretrained("CAILLE/ImageCompression")

model = load_model()

# Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert image to tensor
    image_np = np.array(image) / 255.0  # Normalize
    image_tensor = torch.tensor(image_np).unsqueeze(0).float()

    # Perform AI-based compression
    with torch.no_grad():
        compressed_image = model(image_tensor)

    # Simulate compression by saving the tensor and reloading it
    compressed_bytes = io.BytesIO()
    torch.save(compressed_image, compressed_bytes)
    compressed_bytes.seek(0)

    # Show compressed tensor size
    st.write(f"✅ Compressed Tensor Size: {len(compressed_bytes.getvalue())} bytes")

    # Download compressed tensor
    st.download_button("💾 Download Compressed File", compressed_bytes, "compressed.pt", "application/octet-stream")

