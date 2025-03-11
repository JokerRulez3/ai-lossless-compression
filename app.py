import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import io
import time
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from transformers import AutoModelForVision2Seq, AutoImageProcessor
import nest_asyncio

# Fix Streamlit async conflicts
nest_asyncio.apply()

# Title
st.title("🤖 AI-Based Image Compression & Decompression using VAE")

# Load Pretrained Model from Hugging Face
@st.cache_resource
def load_vae_model():
    model_name = "madebyollin/karlo-v1-alpha"
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

vae, processor = load_vae_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB mode
    image_np = np.array(image)
    end_upload = time.time()

    # Convert to PyTorch Tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Compress with VAE
    start_compression = time.time()
    with torch.no_grad():
        latent_representation = vae.encoder(image_tensor).sample()
        compressed_image = latent_representation.numpy()
    end_compression = time.time()

    compressed_size = compressed_image.nbytes / 1024  # KB

    # Decompress with VAE
    start_decompression = time.time()
    with torch.no_grad():
        reconstructed_image = vae.decoder(latent_representation).sample()
    end_decompression = time.time()

    # Convert reconstructed tensor to NumPy and PIL Image
    decompressed_np = reconstructed_image.squeeze().permute(1, 2, 0).numpy()
    decompressed_np = (decompressed_np * 255).astype(np.uint8)
    decompressed_image = Image.fromarray(decompressed_np)

    # Convert decompressed image to buffer for file download
    decompressed_image_bytes = io.BytesIO()
    decompressed_image.save(decompressed_image_bytes, format="PNG")
    decompressed_image_bytes.seek(0)

    # Compute image quality metrics
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_decompressed = cv2.cvtColor(decompressed_np, cv2.COLOR_RGB2GRAY)

    psnr_value = psnr(gray_original, gray_decompressed, data_range=255)
    ssim_value = ssim(gray_original, gray_decompressed, data_range=255)

    # Time calculations
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    decompression_time = end_decompression - start_decompression
    simulated_download_time = compressed_size / (5 * 1024)  # Assuming 5MB/s speed

    # Display results
    st.image([image, decompressed_image], caption=["Original", "Decompressed (VAE)"])
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")
    st.write(f"🎯 PSNR: {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM: {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"♻️ Decompression Time: {decompression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")

    # Download button for decompressed image
    st.download_button("⬇️ Download Decompressed Image", decompressed_image_bytes, file_name="decompressed_image.png", mime="image/png")
