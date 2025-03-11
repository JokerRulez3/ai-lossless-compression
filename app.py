import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Load pretrained autoencoder model
@st.cache_resource
def load_autoencoder():
    model_url = "https://github.com/mtobeiyf/image-compression-autoencoder/releases/download/v1.0/autoencoder_model.h5"
    model_path = "autoencoder_model.h5"
    model = load_model(model_path)
    return model

autoencoder = load_autoencoder()

st.title("🤖 AI-Based Image Compression Using Autoencoder")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize
    end_upload = time.time()
    
    # Resize image to match model input size
    image_resized = cv2.resize(image_np, (128, 128))
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    
    # Compress using autoencoder
    start_compression = time.time()
    compressed_image = autoencoder.predict(image_resized)
    compressed_image = np.clip(compressed_image[0], 0, 1)  # Ensure valid pixel range
    end_compression = time.time()
    
    # Resize back to original size
    decompressed_image_np = cv2.resize(compressed_image, (image_np.shape[1], image_np.shape[0]))
    decompressed_image_np = (decompressed_image_np * 255).astype(np.uint8)
    
    # Convert decompressed image to PIL format
    decompressed_image = Image.fromarray(decompressed_image_np)
    
    # Save decompressed image for download
    decompressed_buffer = io.BytesIO()
    decompressed_image.save(decompressed_buffer, format="PNG")
    decompressed_buffer.seek(0)
    
    # Compute quality metrics
    gray_original = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_decompressed = cv2.cvtColor(decompressed_image_np, cv2.COLOR_RGB2GRAY)
    
    psnr_value = psnr(gray_original, gray_decompressed, data_range=255)
    ssim_value = ssim(gray_original, gray_decompressed, data_range=255)
    
    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    
    # Display results
    st.image([image, decompressed_image], caption=["Original", "Decompressed (AI-Based)"])
    st.write(f"🎯 PSNR: {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM: {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    
    # Download button
    st.download_button("⬇️ Download Decompressed Image", decompressed_buffer, file_name="decompressed_image.png", mime="image/png")
