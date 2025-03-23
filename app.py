import streamlit as st
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

# ✅ Define SRGAN Generator Model
class SRGenerator(nn.Module):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
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

# ✅ Load Model
@st.cache_resource
def load_model():
    model = SRGenerator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ✅ Preprocess Image (as done during training)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# ✅ AI Decompression (WebP-aware Super-Resolution)
def ai_decompress_webp(compressed_image, model):
    decompressed_np = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
    decompressed_image = Image.fromarray(cv2.cvtColor(decompressed_np, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess_image(decompressed_image)

    with torch.no_grad():
        sr_tensor = model(image_tensor)

    sr_image = sr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    sr_image = np.clip((sr_image + 1) / 2, 0, 1)
    sr_image = (sr_image * 255).astype(np.uint8)

    return Image.fromarray(sr_image)

# ✅ Streamlit UI
st.title("🔗 WebP-Aware SRGAN for Adaptive Image Restoration")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # ✅ WebP Compression
    _, compressed_image = cv2.imencode(".webp", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 80])

    # ✅ AI Decompression
    ai_image = ai_decompress_webp(compressed_image, model)

    # ✅ Metrics Calculation
    original_resized = cv2.resize(image_np, (256, 256))
    ai_resized = np.array(ai_image.resize((256, 256)))

    psnr_ai = psnr(original_resized, ai_resized, data_range=255)
    ssim_ai = ssim(original_resized, ai_resized, channel_axis=2, data_range=255)

    # ✅ Display Results
    st.image([image, ai_image], caption=["Original", "AI Super-Resolved (WebP-aware)"])
    st.write(f"🎯 AI PSNR: {psnr_ai:.2f} dB | 🔍 AI SSIM: {ssim_ai:.4f}")

    # ✅ Download Button
    img_byte_arr = io.BytesIO()
    ai_image.save(img_byte_arr, format="PNG")
    st.download_button("Download AI Restored Image", img_byte_arr.getvalue(), file_name="webp_ai_restored.png", mime="image/png")
