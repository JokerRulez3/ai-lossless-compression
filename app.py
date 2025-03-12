import streamlit as st
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
import os
import requests
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Model Path
MODEL_PATH = "models/autoencoder-highres.pth"
MODEL_URL = "https://raw.githubusercontent.com/JokerRulez3/ai-lossless-compression/main/models/autoencoder-highres.pth"

# ✅ Residual Block with BatchNorm + LeakyReLU
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, negative_slope=0.01)

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # Increased downsampling
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Use Tanh instead of Sigmoid for [-1, 1] range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def download_model():
    """Downloads the model if missing or corrupted."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    """Load and initialize the trained model."""
    download_model()  # Ensure the model is downloaded
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model.to("cpu").float()  # Ensure FP32 on CPU

# Load the model
model = load_model()

# ✅ Image Preprocessing
def preprocess_image(image):
    """
    Convert an image to a tensor, ensuring it matches the model's expected input.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match Tanh output range [-1,1]
    ])
    return transform(image).unsqueeze(0).float()  # Ensure batch dimension & FP32

# ✅ AI Compression-Decompression
def ai_compress_decompress(image, model):
    """
    Pass an image through the autoencoder for compression & decompression.
    """
    image_tensor = preprocess_image(image).to("cpu")  # Ensure it's on CPU

    with torch.no_grad():
        # Forward pass through Autoencoder
        decompressed = model(image_tensor)

    # ✅ Ensure output is scaled back to [0,255]
    decompressed_np = decompressed.squeeze(0).permute(1, 2, 0).numpy()
    decompressed_np = np.clip((decompressed_np * 0.5) + 0.5, 0, 1)  # Convert back from [-1,1] to [0,1]
    decompressed_np = (decompressed_np * 255).astype(np.uint8)  # Convert to uint8 for image display

    return Image.fromarray(decompressed_np)


# Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression & Decompression")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image, dtype=np.uint8)
    end_upload = time.time()

    # WebP Compression
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    start_compression = time.time()
    _, compressed_image = cv2.imencode(".webp", image_cv, [cv2.IMWRITE_WEBP_QUALITY, 80])
    end_compression = time.time()
    compressed_size = len(compressed_image) / 1024  # KB

    # WebP Decompression
    start_decompression = time.time()
    decompressed_np = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
    decompressed_np = cv2.cvtColor(decompressed_np, cv2.COLOR_BGR2RGB)
    end_decompression = time.time()

    # AI Decompression
    ai_decompressed = ai_compress_decompress(image, model)

    # Resize images to match dimensions
    original_resized = cv2.resize(image_np, (128, 128))
    webp_resized = cv2.resize(decompressed_np, (128, 128))
    ai_resized = np.array(ai_decompressed)

    # Convert to grayscale for PSNR/SSIM calculations
    gray_original = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(webp_resized, cv2.COLOR_RGB2GRAY)
    gray_ai_decompressed = cv2.cvtColor(ai_resized, cv2.COLOR_RGB2GRAY)

    # Compute Metrics
    psnr_value_webp = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value_webp = ssim(gray_original, gray_compressed, data_range=255)

    psnr_value_ai = psnr(gray_original, gray_ai_decompressed, data_range=255)
    ssim_value_ai = ssim(gray_original, gray_ai_decompressed, data_range=255)

    # Display Results
    st.image([image, Image.fromarray(decompressed_np), ai_decompressed],
             caption=["Original", "Decompressed (WebP)", "AI Decompressed"])
    
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ WebP Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")
    
    st.write(f"🎯 WebP PSNR: {psnr_value_webp:.2f} dB | AI PSNR: {psnr_value_ai:.2f} dB")
    st.write(f"🔍 WebP SSIM: {ssim_value_webp:.4f} | AI SSIM: {ssim_value_ai:.4f}")

    st.write(f"⏳ Upload Time: {end_upload - start_upload:.4f} sec")
    st.write(f"⚡ Compression Time: {end_compression - start_compression:.4f} sec")
    st.write(f"♻️ Decompression Time: {end_decompression - start_decompression:.4f} sec")

    # Download Button
    img_byte_arr = io.BytesIO()
    ai_decompressed.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button("Download AI Decompressed Image", img_byte_arr, file_name="ai_compressed.png", mime="image/png")
