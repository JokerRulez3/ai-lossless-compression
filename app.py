import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to avoid TensorFlow CUDA errors

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🤖 AI-Based Image Compression")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB mode
    image_np = np.array(image, dtype=np.uint8)  # Explicitly set dtype
    end_upload = time.time()
    
    original_size_kb = uploaded_file.size / 1024  # KB
    original_shape = image_np.shape  # Get original shape
    original_dtype = image_np.dtype  # Get original dtype
    
    # Convert to TensorFlow tensor
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    
    # AI-Based Compression (WebP)
    start_compression = time.time()
    compressed_image = tf.io.encode_jpeg(image_tf, format='rgb', quality=85)  # WebP or JPEG
    compressed_image_np = compressed_image.numpy()
    end_compression = time.time()
    compressed_size_kb = len(compressed_image_np) / 1024  # KB
    
    # Decompression
    start_decompression = time.time()
    decompressed_image = tf.io.decode_jpeg(compressed_image, channels=3)
    decompressed_image_np = decompressed_image.numpy().astype(np.uint8)  # Ensure correct dtype
    end_decompression = time.time()
    
    decompressed_shape = decompressed_image_np.shape  # Verify shape
    decompressed_dtype = decompressed_image_np.dtype  # Verify dtype

    # ✅ **Final Fix for Correct Decompressed Size Calculation**
    decompressed_size_kb = (
        decompressed_image_np.shape[0] * decompressed_image_np.shape[1] * decompressed_image_np.shape[2] * decompressed_image_np.itemsize
    ) / 1024  # Convert bytes to KB
    
    # Ensure image compatibility for SSIM
    min_dim = min(image_np.shape[0], image_np.shape[1])
    win_size = min(11, min_dim) if min_dim >= 7 else 3
    
    # Convert images to grayscale for SSIM
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_decompressed = cv2.cvtColor(decompressed_image_np, cv2.COLOR_RGB2GRAY)
    
    # Resize if too large
    if min_dim > 1024:
        target_size = (1024, 1024)
        gray_original = cv2.resize(gray_original, target_size)
        gray_decompressed = cv2.resize(gray_decompressed, target_size)
    
    # Compute quality metrics (Avoid division by zero)
    psnr_value = psnr(gray_original + 1e-10, gray_decompressed + 1e-10, data_range=255)
    ssim_value = ssim(gray_original, gray_decompressed, data_range=255, win_size=win_size)
    
    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    decompression_time = end_decompression - start_decompression
    simulated_download_time = compressed_size_kb / (5 * 1024)  # 5MB/s speed assumption
    
    # 🛠️ **Debugging Prints**
    print(f"Original Shape: {original_shape}, Decompressed Shape: {decompressed_shape}")
    print(f"Original Dtype: {original_dtype}, Decompressed Dtype: {decompressed_dtype}")
    print(f"Original Size: {original_size_kb:.2f} KB, Decompressed Size: {decompressed_size_kb:.2f} KB")
    
    # Display results
    st.image([image, Image.open(io.BytesIO(compressed_image_np)), Image.fromarray(decompressed_image_np)], 
             caption=["Original", "Compressed", "Decompressed"])
    st.write(f"📏 Original Size: {original_size_kb:.2f} KB")
    st.write(f"✅ Compressed Size: {compressed_size_kb:.2f} KB ({compressed_size_kb / original_size_kb * 100:.2f}% of original)")
    st.write(f"📂 Decompressed Size: {decompressed_size_kb:.2f} KB (should match original)")
    st.write(f"🎯 PSNR: {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM: {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"♻️ Decompression Time: {decompression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")
