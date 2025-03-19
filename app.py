import streamlit as st
import asyncio
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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ✅ Model Paths
MODEL_PATH = "models/srgan_generator.pth"
MODEL_URL = "https://raw.githubusercontent.com/JokerRulez3/ai-lossless-compression/main/models/srgan_generator.pth.pth"

# ✅ Define SRGAN Generator Model
class SRGenerator(nn.Module):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[self._residual_block(64) for _ in range(5)]
        )

        # Upsampling blocks
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.tanh(self.conv2(x))
        return x

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

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
    model = SRGenerator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model.to("cpu").float()

model = load_model()

# ✅ Image Preprocessing for SRGAN
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Downsample input for SRGAN
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    return transform(image).unsqueeze(0).float()

# ✅ AI Compression-Decompression using SRGAN
def ai_compress_decompress(image, model):
    image_tensor = preprocess_image(image).to("cpu")

    with torch.no_grad():
        sr_tensor = model(image_tensor)  # Super-Resolve image

    # ✅ Convert [-1,1] to [0,255]
    sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    sr_image = np.clip((sr_image + 1) / 2, 0, 1)  
    sr_image = (sr_image * 255).astype(np.uint8)

    return Image.fromarray(sr_image)

# ✅ Streamlit UI
st.title("🔗 AI-Based Lossless Image Compression & Decompression (SRGAN)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image, dtype=np.uint8)
    end_upload = time.time()

    # ✅ WebP Compression
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    start_compression = time.time()
    _, compressed_image = cv2.imencode(".webp", image_cv, [cv2.IMWRITE_WEBP_QUALITY, 80])
    end_compression = time.time()
    compressed_size = len(compressed_image) / 1024  

    # ✅ WebP Decompression
    start_decompression = time.time()
    decompressed_np = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
    decompressed_np = cv2.cvtColor(decompressed_np, cv2.COLOR_BGR2RGB)
    end_decompression = time.time()

    # ✅ AI Decompression (Super-Resolution)
    ai_decompressed = ai_compress_decompress(image, model)

    # ✅ Resize images for consistent dimensions
    original_resized = cv2.resize(image_np, (128, 128))
    webp_resized = cv2.resize(decompressed_np, (128, 128))
    ai_resized = np.array(ai_decompressed)

    # ✅ Convert to grayscale for PSNR/SSIM
    gray_original = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(webp_resized, cv2.COLOR_RGB2GRAY)
    gray_ai_decompressed = cv2.cvtColor(ai_resized, cv2.COLOR_RGB2GRAY)

    # ✅ Compute Metrics (Ensure shape match)
    psnr_value_webp = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value_webp = ssim(gray_original, gray_compressed, data_range=255)

    if gray_ai_decompressed.shape != gray_original.shape:
        gray_ai_decompressed = cv2.resize(gray_ai_decompressed, (gray_original.shape[1], gray_original.shape[0]))

    psnr_value_ai = psnr(gray_original, gray_ai_decompressed, data_range=255)
    ssim_value_ai = ssim(gray_original, gray_ai_decompressed, data_range=255)

    # ✅ Display Results
    st.image([image, Image.fromarray(decompressed_np), ai_decompressed],
             caption=["Original", "Decompressed (WebP)", "AI Super-Resolved"])

    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ WebP Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")

    st.write(f"🎯 WebP PSNR: {psnr_value_webp:.2f} dB | AI PSNR: {psnr_value_ai:.2f} dB")
    st.write(f"🔍 WebP SSIM: {ssim_value_webp:.4f} | AI SSIM: {ssim_value_ai:.4f}")

    st.write(f"⏳ Upload Time: {end_upload - start_upload:.4f} sec")
    st.write(f"⚡ Compression Time: {end_compression - start_compression:.4f} sec")
    st.write(f"♻️ Decompression Time: {end_decompression - start_decompression:.4f} sec")

    # ✅ Download Button
    img_byte_arr = io.BytesIO()
    ai_decompressed.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button("Download AI Super-Resolved Image", img_byte_arr, file_name="ai_superresolved.png", mime="image/png")
