import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🔗 AI-Based Lossless Image Compression")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    start_upload_time = time.time()
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    original_size = uploaded_file.size / 1024  # KB
    upload_time = time.time() - start_upload_time
    
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Compress using JPEG2000 (lossless)
    start_compress_time = time.time()
    _, compressed_image = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 0])
    compressed_bytes = io.BytesIO(compressed_image)
    compressed_size = compressed_bytes.getbuffer().nbytes / 1024  # KB
    compress_time = time.time() - start_compress_time
    
    # Convert back to image for quality comparison
    compressed_np = cv2.imdecode(np.frombuffer(compressed_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Ensure grayscale images are handled properly
    if len(compressed_np.shape) == 2:
        compressed_np = cv2.cvtColor(compressed_np, cv2.COLOR_GRAY2RGB)
    
    # Compute quality metrics
    psnr_value = psnr(image_np, compressed_np, data_range=255)
    ssim_value = ssim(image_np, compressed_np, multichannel=True, data_range=255)
    
    # Compression stats
    compression_ratio = (compressed_size / original_size) * 100
    simulated_download_time = compressed_size / (500 * 1024)  # Assuming 500 KB/s download speed
    
    st.write(f"\U0001F4CF Original Size: {original_size:.2f} KB")
    st.write(f"✅ Compressed Size: {compressed_size:.2f} KB")
    st.write(f"📉 Compression Ratio: {compression_ratio:.2f}% of original size")
    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compress_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {simulated_download_time:.4f} sec")
    st.write(f"🖼️ PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    st.write(f"🧐 SSIM (Structural Similarity Index): {ssim_value:.4f}")
    
    # Display compressed image
    st.image(compressed_np, caption="Compressed Image", use_column_width=True)
    
    # Download button
    st.download_button("💾 Download Compressed Image", compressed_bytes, "compressed.jp2", "image/jp2")
