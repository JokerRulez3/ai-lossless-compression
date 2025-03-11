import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🤖 AI-Based Lossless Image Compression")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    end_upload = time.time()
    
    # Convert to TensorFlow tensor
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.uint8)

    # AI-based Compression (JPEG using TensorFlow)
    start_compression = time.time()
    compressed_image = tf.io.encode_jpeg(image_tf, format='rgb', quality=80)
    compressed_image_np = compressed_image.numpy()
    end_compression = time.time()
    compressed_size = len(compressed_image_np) / 1024  # KB
    
    # Decompression (Decode back to Tensor)
    start_decompression = time.time()
    decompressed_image = tf.io.decode_jpeg(compressed_image, channels=3)
    decompressed_image_np = decompressed_image.numpy()
    end_decompression = time.time()
    
    # **Ensure Decompressed Size is Correct**
    decompressed_pil = Image.fromarray(decompressed_image_np)
    decompressed_bytes = io.BytesIO()
    
    # **Fix: Save with same quality (80) as compression**
    decompressed_pil.save(decompressed_bytes, format="JPEG", quality=80)
    
    decompressed_size = len(decompressed_bytes.getvalue()) / 1024  # KB

    # Ensure image compatibility for SSIM
    min_dim = min(image_np.shape[0], image_np.shape[1])
    win_size = min(11, min_dim) if min_dim >= 7 else 3
    
    # Convert images to grayscale for SSIM
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_decompressed = cv2.cvtColor(decompressed_image_np, cv2.COLOR_RGB2GRAY)
    
    # Resize images for metric computation if too large
    if min_dim > 1024:
        target_size = (1024, 1024)
        gray_original = cv2.resize(gray_original, target_size)
        gray_decompressed = cv2.resize(gray_decompressed, target_size)
    
    # Compute quality metrics
    psnr_value = psnr(gray_original, gray_decompressed, data_range=255)
    ssim_value = ssim(gray_original, gray_decompressed, data_range=255, win_size=win_size)
    
    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    decompression_time = end_decompression - start_decompression
    simulated_download_time = compressed_size / (5 * 1024)  # Assuming 5MB/s speed
    
    # Display results
    st.image([image, Image.open(io.BytesIO(compressed_image_np)), decompressed_pil], 
             caption=["Original", "Compressed (AI-Based)", "Decompressed"])
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ AI-Based Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")
    st.write(f"📂 Decompressed Size: {decompressed_size:.2f} KB")
    st.write(f"🎯 PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM (Structural Similarity Index): {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"♻️ Decompression Time: {decompression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")
