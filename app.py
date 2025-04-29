import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ✅ Model Path (Your Colab-trained model)
MODEL_PATH = "models/srgan_generator_webp.pth"

# ✅ Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + res, 0.01)

# ✅ SRGAN Generator
class SRGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(64, 256, 3, padding=1), nn.PixelShuffle(2), nn.PReLU()
        )
        self.conv2 = nn.Conv2d(64, 3, 9, padding=4)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.upsample(x)
        return self.tanh(self.conv2(x))

# ✅ Load Model
@st.cache_resource
def load_model():
    model = SRGenerator()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ✅ Preprocess function
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# ✅ AI Decompression pipeline
def ai_decompress_webp(compressed_bytes, model):
    decoded = cv2.imdecode(np.frombuffer(compressed_bytes, np.uint8), cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(decoded)

    tensor = preprocess(pil)
    with torch.no_grad():
        out = model(tensor)
    out_img = out.squeeze(0).permute(1, 2, 0).numpy()
    out_img = ((out_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out_img)

# ✅ Streamlit App
st.title("🖼️ WebP-Aware SRGAN - AI Image Restoration")

uploaded_file = st.file_uploader("Upload an image (JPEG/PNG/WebP)", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ✅ Compress to WebP
    webp_buffer = cv2.imencode(".webp", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 60])[1]

    # ✅ AI Restore
    ai_image = ai_decompress_webp(webp_buffer, model)

    # ✅ Resize for Metrics (Match shapes)
    original = cv2.resize(image_np, (256, 256))
    ai_resized = cv2.resize(np.array(ai_image), (256, 256))

    psnr_val = psnr(original, ai_resized, data_range=255)
    ssim_val = ssim(original, ai_resized, channel_axis=2, data_range=255)

    # ✅ Show Results
    st.image([image, ai_image], caption=["Original", "AI Restored"])
    st.write(f"📊 **PSNR**: {psnr_val:.2f} dB | **SSIM**: {ssim_val:.4f}")

    # ✅ Download
    img_bytes = io.BytesIO()
    ai_image.save(img_bytes, format="PNG")
    st.download_button("⬇️ Download Restored Image", img_bytes.getvalue(), file_name="ai_restored.png", mime="image/png")
