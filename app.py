import streamlit as st
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
import os
import requests
import time
import brotli
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Model Path
MODEL_PATH = "models/autoencoder-highres.pth"
MODEL_URL = "https://raw.githubusercontent.com/JokerRulez3/ai-lossless-compression/main/models/autoencoder-highres.pth"

# ✅ Define Residual Block
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

# ✅ Define Autoencoder Model
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
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
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
            nn.Tanh()  # Output in [-1,1] range
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

@st.cache_resource
def load_model():
    """Load the trained model into CPU"""
    download_model()
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model.to("cpu").float()

model = load_model()

# ✅ Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for Tanh
    ])
    return transform(image).unsqueeze(0).float()  # Add batch dimension

# ✅ AI Compression-Decompression
def ai_compress_decompress(image, model):
    """Compress and decompress an image using the AI model."""
    image_tensor = preprocess_image(image).to("cpu")

    with torch.no_grad():
        decompressed = model(image_tensor)

    # ✅ Resize output to match original image size
    decompressed = torch.nn.functional.interpolate(
        decompressed, size=image.size[::-1], mode='bilinear', align_corners=False
    )

    # ✅ Convert from Tanh [-1,1] to [0,255]
    decompressed_np = decompressed.squeeze(0).permute(1, 2, 0).numpy()
    decompressed_np = np.clip((decompressed_np + 1) / 2, 0, 1)  # Convert to [0,1]
    decompressed_np = (decompressed_np * 255).astype(np.uint8)

    return Image.fromarray(decompressed_np)

# ✅ Brotli Compression
def brotli_compress(image):
    """Compress an image using Brotli"""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    compressed = brotli.compress(image_bytes.getvalue(), quality=11)
    return compressed

# ✅ Brotli Decompression
def brotli_decompress(compressed_data):
    """Decompress Brotli-compressed image"""
    decompressed_data = brotli.decompress(compressed_data)
    return Image.open(io.BytesIO(decompressed_data))

# ✅ Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression & Decompression (Brotli)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image, dtype=np.uint8)
    end_upload = time.time()

    # Brotli Compression
    start_compression = time.time()
    compressed_brotli = brotli_compress(image)
    end_compression = time.time()
    compressed_size = len(compressed_brotli) / 1024  # KB

    # Brotli Decompression
    start_decompression = time.time()
    decompressed_image = brotli_decompress(compressed_brotli)
    end_decompression = time.time()

    # AI Decompression
    ai_decompressed = ai_compress_decompress(image, model)

    # ✅ Resize images to match dimensions
    original_resized = cv2.resize(image_np, (128, 128))
    brotli_resized = cv2.resize(np.array(decompressed_image), (128, 128))
    ai_resized = np.array(ai_decompressed)

    # ✅ Convert to grayscale for PSNR/SSIM calculations
    gray_original = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(brotli_resized, cv2.COLOR_RGB2GRAY)
    gray_ai_decompressed = cv2.cvtColor(ai_resized, cv2.COLOR_RGB2GRAY)

    # ✅ Ensure all images have the same dimensions
    gray_ai_decompressed = cv2.resize(gray_ai_decompressed, (gray_original.shape[1], gray_original.shape[0]))

    # ✅ Compute Metrics
    psnr_value_brotli = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value_brotli = ssim(gray_original, gray_compressed, data_range=255)

    psnr_value_ai = psnr(gray_original, gray_ai_decompressed, data_range=255)
    ssim_value_ai = ssim(gray_original, gray_ai_decompressed, data_range=255)

    # ✅ Display Results
    st.image([image, decompressed_image, ai_decompressed],
             caption=["Original", "Decompressed (Brotli)", "AI Decompressed"])
    
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ Brotli Compressed Size: {compressed_size:.2f} KB")

    st.write(f"🎯 Brotli PSNR: {psnr_value_brotli:.2f} dB | AI PSNR: {psnr_value_ai:.2f} dB")
    st.write(f"🔍 Brotli SSIM: {ssim_value_brotli:.4f} | AI SSIM: {ssim_value_ai:.4f}")

    st.write(f"⚡ Compression Time: {end_compression - start_compression:.4f} sec")
    st.write(f"♻️ Decompression Time: {end_decompression - start_decompression:.4f} sec")

    st.download_button("Download AI Decompressed Image", ai_decompressed, file_name="ai_compressed.png", mime="image/png")
