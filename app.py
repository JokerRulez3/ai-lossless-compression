import streamlit as st
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import brotli  # Brotli compression
from PIL import Image
import io
import os
import requests
import time
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
            nn.Tanh()  # Tanh output ([-1,1])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Download Model if missing
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

@st.cache_resource
def load_model():
    download_model()
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model.to("cpu").float()

model = load_model()

# ✅ Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    return transform(image).unsqueeze(0).float()

# ✅ AI Compression-Decompression
def ai_compress_decompress(image, model):
    image_tensor = preprocess_image(image).to("cpu")

    with torch.no_grad():
        decompressed = model(image_tensor)

    # ✅ Resize output to original image size
    decompressed = torch.nn.functional.interpolate(
        decompressed, size=image.size[::-1], mode='bilinear', align_corners=False
    )

    # ✅ Convert [-1,1] to [0,255]
    decompressed_np = decompressed.squeeze(0).permute(1, 2, 0).numpy()
    decompressed_np = np.clip((decompressed_np + 1) / 2, 0, 1)  
    decompressed_np = (decompressed_np * 255).astype(np.uint8)

    return Image.fromarray(decompressed_np)

# ✅ Brotli Compression
def brotli_compress(image):
    """Compress an image using Brotli."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # Convert to bytes
    compressed_data = brotli.compress(img_byte_arr.getvalue())  # Brotli compress
    return compressed_data

# ✅ Brotli Decompression
def brotli_decompress(compressed_data):
    """Decompress Brotli compressed image."""
    decompressed_bytes = brotli.decompress(compressed_data)  # Brotli decompress
    return Image.open(io.BytesIO(decompressed_bytes))  # Convert to image

# ✅ Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression with Brotli & AI")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image, dtype=np.uint8)
    end_upload = time.time()

    # ✅ Brotli Compression
    start_compression = time.time()
    compressed_brotli = brotli_compress(image)
    end_compression = time.time()
    compressed_size = len(compressed_brotli) / 1024  # KB

    # ✅ Brotli Decompression
    start_decompression = time.time()
    decompressed_brotli = brotli_decompress(compressed_brotli)
    decompressed_brotli_np = np.array(decompressed_brotli)
    end_decompression = time.time()

    # ✅ AI Decompression
    ai_decompressed = ai_compress_decompress(image, model)

    # ✅ Resize images for PSNR/SSIM calculations
    original_resized = cv2.resize(image_np, (128, 128))
    brotli_resized = cv2.resize(decompressed_brotli_np, (128, 128))
    ai_resized = np.array(ai_decompressed)

    # ✅ Convert to grayscale
    gray_original = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    gray_brotli = cv2.cvtColor(brotli_resized, cv2.COLOR_RGB2GRAY)
    gray_ai = cv2.cvtColor(ai_resized, cv2.COLOR_RGB2GRAY)

    # ✅ Compute PSNR & SSIM
    psnr_brotli = psnr(gray_original, gray_brotli, data_range=255)
    ssim_brotli = ssim(gray_original, gray_brotli, data_range=255)
    psnr_ai = psnr(gray_original, gray_ai, data_range=255)
    ssim_ai = ssim(gray_original, gray_ai, data_range=255)

    # ✅ Display Results
    st.image([image, decompressed_brotli, ai_decompressed],
             caption=["Original", "Decompressed (Brotli)", "AI Decompressed"])
    
    st.write(f"📏 Brotli Compressed Size: {compressed_size:.2f} KB")
    st.write(f"🎯 Brotli PSNR: {psnr_brotli:.2f} dB | AI PSNR: {psnr_ai:.2f} dB")
    st.write(f"🔍 Brotli SSIM: {ssim_brotli:.4f} | AI SSIM: {ssim_ai:.4f}")
