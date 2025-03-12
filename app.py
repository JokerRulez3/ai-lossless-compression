import streamlit as st
import asyncio
import torch
import torch.nn as nn
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
MODEL_PATH = "models/autoencoder.pth"
MODEL_URL = "https://raw.githubusercontent.com/JokerRulez3/ai-lossless-compression/main/models/autoencoder.pth"

# Ensure async event loop works
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Download Model if Missing
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

# Load AI Model
@st.cache_resource
def load_model():
    download_model()
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# 🛠 Fix 1: Optimize Image Preprocessing (Reduce Memory Usage)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float16)  # ✅ Convert to float16 to reduce RAM usage
    ])
    return transform(image).unsqueeze(0)

# 🛠 Fix 2: AI Compression-Decompression with Efficient Tensor Processing
def ai_compress_decompress(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        compressed = model.encoder(image_tensor)
        decompressed = model.decoder(compressed)
    decompressed_np = decompressed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decompressed_np = (decompressed_np * 255).astype(np.uint8)
    return Image.fromarray(decompressed_np)

# Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression & Decompression")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB to avoid grayscale issues
    image_np = np.array(image, dtype=np.uint8)
    end_upload = time.time()

    # 🛠 Fix 3: Efficient WebP Compression
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    start_compression = time.time()
    _, compressed_image = cv2.imencode(".webp", image_cv, [cv2.IMWRITE_WEBP_QUALITY, 80])
    end_compression = time.time()
    compressed_size = len(compressed_image) / 1024  # KB

    # 🛠 Fix 4: WebP Decompression with Optimized Read
    start_decompression = time.time()
    decompressed_np = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_UNCHANGED)
    decompressed_np = cv2.cvtColor(decompressed_np, cv2.COLOR_BGR2RGB)
    end_decompression = time.time()

    # AI Decompression
    ai_decompressed = ai_compress_decompress(image, model)

    # Compute Metrics
    min_dim = min(image_np.shape[0], image_np.shape[1])
    win_size = min(11, min_dim) if min_dim >= 7 else 3  # 🛠 Auto-select `win_size` to prevent crashes

    # 🛠 Fix 5: Resize Images to Match Before PSNR/SSIM Calculation
    gray_original = cv2.cvtColor(cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(cv2.resize(decompressed_np, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray_ai_decompressed = cv2.cvtColor(np.array(ai_decompressed), cv2.COLOR_RGB2GRAY)

    psnr_value_webp = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value_webp = ssim(gray_original, gray_compressed, data_range=255, win_size=win_size)
    
    psnr_value_ai = psnr(gray_original, gray_ai_decompressed, data_range=255)
    ssim_value_ai = ssim(gray_original, gray_ai_decompressed, data_range=255, win_size=win_size)

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
