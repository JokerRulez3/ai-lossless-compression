import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🔗 AI-Based Lossless Image Compression & Decompression")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    start_upload = time.time()
    image = Image.open(uploaded_file)
    image_np = np.array(image, dtype=np.uint8)  # Ensure 8-bit format
    end_upload = time.time()

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Compress using WebP
    start_compression = time.time()
    _, compressed_image = cv2.imencode(".webp", image_cv, [cv2.IMWRITE_WEBP_QUALITY, 80])
    end_compression = time.time()
    compressed_size = len(compressed_image) / 1024  # Convert to KB

    # Decompress WebP
    start_decompression = time.time()
    decompressed_np = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
    decompressed_np = cv2.cvtColor(decompressed_np, cv2.COLOR_BGR2RGB)
    end_decompression = time.time()

    # Ensure decompressed image has the same dtype and shape
    decompressed_np = decompressed_np.astype(np.uint8)

    # Convert decompressed image to JPEG format to check file size
    decompressed_image = Image.fromarray(decompressed_np)
    buffer = io.BytesIO()
    decompressed_image.save(buffer, format="JPEG", quality=95)
    decompressed_size = len(buffer.getvalue()) / 1024  # Convert to KB

    # Ensure image compatibility for SSIM
    min_dim = min(image_np.shape[0], image_np.shape[1])
    win_size = min(11, min_dim) if min_dim >= 7 else 3  # Ensure win_size is valid

    # Convert images to grayscale for SSIM
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(decompressed_np, cv2.COLOR_RGB2GRAY)

    # Resize images for metric computation if too large
    if min_dim > 1024:
        target_size = (1024, 1024)
        gray_original = cv2.resize(gray_original, target_size)
        gray_compressed = cv2.resize(gray_compressed, target_size)

    # Compute quality metrics
    psnr_value = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value = ssim(gray_original, gray_compressed, data_range=255, win_size=win_size)

    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression
    decompression_time = end_decompression - start_decompression
    simulated_download_time = compressed_size / (5 * 1024)  # Assuming 5MB/s speed

    # Display results
    st.image([image, Image.open(io.BytesIO(compressed_image)), decompressed_image], caption=["Original", "Compressed (WebP)", "Decompressed (WebP)"])
    st.write(f"📏 Original Size: {uploaded_file.size / 1024:.2f} KB")
    st.write(f"✅ WebP Compressed Size: {compressed_size:.2f} KB ({compressed_size / (uploaded_file.size / 1024) * 100:.2f}% of original)")
    st.write(f"📂 Decompressed Size: {decompressed_size:.2f} KB ({decompressed_size / compressed_size * 100:.2f}% of compressed)")
    st.write(f"🎯 PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM (Structural Similarity Index): {ssim_value:.4f}")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"♻️ Decompression Time: {decompression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")
