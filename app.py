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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ✅ Model Path
MODEL_PATH = "models/srgan_generator_webp_block5.pth"

# ✅ Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.01)

# ✅ SRGAN Generator Model Definition
class SRGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

# ✅ Image Preprocessing (Matching Training Setup)
def preprocess_webp_image(webp_buffer):
    # Decode WebP compressed image
    img_np = cv2.imdecode(np.frombuffer(webp_buffer, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    # Resize and normalize as done during training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img_rgb).unsqueeze(0)

# ✅ AI Super-Resolution from WebP
def ai_super_resolve_webp(webp_image, model, original_size):
    image_tensor = preprocess_webp_image(webp_image)

    with torch.no_grad():
        sr_tensor = model(image_tensor)

    # Convert output tensor [-1, 1] to image [0, 255]
    sr_image = sr_tensor.squeeze().permute(1, 2, 0).numpy()
    sr_image = np.clip((sr_image + 1) / 2, 0, 1)
    sr_image = (sr_image * 255).astype(np.uint8)

    # Resize SR image to original dimensions
    sr_image_resized = cv2.resize(sr_image, original_size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(sr_image_resized)

# ✅ Streamlit Interface
st.title("🌐 WebP-Aware AI Adaptive Image Restoration (SRGAN)")

uploaded_file = st.file_uploader("Upload your image:", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_image)
    original_size = original_image.size  # (width, height)

    # ✅ WebP Compression (simulate your storage scenario)
    encode_params = [cv2.IMWRITE_WEBP_QUALITY, 80]
    _, webp_buffer = cv2.imencode(".webp", cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), encode_params)
    webp_size_kb = len(webp_buffer) / 1024

    # ✅ Decode WebP Compressed Image
    webp_decoded_np = cv2.imdecode(np.frombuffer(webp_buffer, np.uint8), cv2.IMREAD_COLOR)
    webp_decoded_rgb = cv2.cvtColor(webp_decoded_np, cv2.COLOR_BGR2RGB)
    webp_image_pil = Image.fromarray(webp_decoded_rgb)

    # ✅ AI Super-Resolution from compressed WebP
    ai_restored_image = ai_super_resolve_webp(webp_buffer, model, original_size)

    # ✅ Metrics Calculation (PSNR & SSIM)
    original_resized = np.array(original_image.resize((256, 256)))
    webp_resized = np.array(webp_image_pil.resize((256, 256)))
    ai_resized = np.array(ai_restored_image.resize((256, 256)))

    # ✅ PSNR & SSIM Calculation
    psnr_webp = psnr(original_resized, webp_resized, data_range=255)
    ssim_webp = ssim(original_resized, webp_resized, channel_axis=2, data_range=255)
    psnr_value = psnr(original_resized, ai_resized, data_range=255)
    ssim_value = ssim(original_resized, ai_resized, channel_axis=2, data_range=255)

    # ✅ Display Results
    st.image([original_image, webp_image_pil, ai_restored_image], caption=["Original Image", f"WebP Compressed ({webp_size_kb:.2f} KB)", "AI Restored Image"])
    st.write("📊 **Quality Metrics (256x256 resized):**")
    st.write(f"📦 **WebP Size:** {webp_size_kb:.2f} KB")
    st.write(f"🎯 **WebP PSNR:** {psnr_webp:.2f} dB | 🔍 SSIM: {ssim_webp:.4f}")
    st.write(f"📊 **AI PSNR:** {psnr_value:.2f} dB | **AI SSIM:** {ssim_value:.4f}")

    # ✅ Download Button
    img_byte_arr = io.BytesIO()
    ai_restored_image.save(img_byte_arr, format="PNG")
    st.download_button("Download Restored Image", img_byte_arr.getvalue(), "ai_restored.png", "image/png")