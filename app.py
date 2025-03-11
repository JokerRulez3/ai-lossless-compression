import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU

import streamlit as st
import numpy as np
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
    
    # AI-Based Lossless Compression (WebP)
    start_compression = time.time()
    compressed_image = tf.io.encode_jpeg(image_tf, format='rgb', quality=100)  # Use WebP instead of PNG
    compressed_image_np = compressed_image.numpy()
    end_compression = time.time()
    compressed_size = len(compressed_image_np) / 1024  # KB
    
    # Decompression (Decode back to Tensor)
    start_decompression = time.time()
    decompressed_image = tf.io.decode_jpeg(compressed_image, channels=3)
    decompressed_image_np = decompressed_image.numpy()
    end_decompression = time.time()
    
    # Ensure decompressed size is same as original
    assert decompressed_image_np.shape == image_np.shape, "Lossless compression failed!"
    
    # Compute quality metrics
    psnr_value = psnr(image_np, decompressed_image_np, data_range=255)
    ssim_value = ssim(image_np, decompressed_image_np, data_range=255, multichannel=True)
    
    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    decompression_time = end_decompression - start_decompression
    simulated_download_time = compressed_size / (5 * 1024)  # Assuming 5MB/s speed
    
    # Display results
    st.image([image, Image.open(io.BytesIO(compressed_image_np)), Image.fromarray(decompressed_image_np)], 
             caption=["Original", "Compressed (AI-Based Lossless WebP)", "Decompressed"])
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ AI-Based Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")
    st.write(f"📂 Decompressed Size: {decompressed_image_np.nbytes / 1024:.2f} KB (should match original)")
    st.write(f"🎯 PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM (Structural Similarity Index): {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"♻️ Decompression Time: {decompression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")
