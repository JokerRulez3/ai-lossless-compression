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
    start_upload = time.time()
    
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    end_upload = time.time()

    # Show original image
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Compute original image size
    original_size = uploaded_file.size / 1024  # KB

    # Convert to grayscale for quality metrics
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 🔹 Compress using JPEG2000 (lossless)
    start_compress = time.time()
    _, compressed_jp2 = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
    compressed_jp2_bytes = io.BytesIO(compressed_jp2)
    compressed_jp2_size = compressed_jp2_bytes.getbuffer().nbytes / 1024  # KB

    # 🔹 Compress using WebP (lossy)
    _, compressed_webp = cv2.imencode(".webp", image_np, [cv2.IMWRITE_WEBP_QUALITY, 80])
    compressed_webp_bytes = io.BytesIO(compressed_webp)
    compressed_webp_size = compressed_webp_bytes.getbuffer().nbytes / 1024  # KB
    end_compress = time.time()

    # Convert compressed images back to numpy
    compressed_jp2_np = cv2.imdecode(np.frombuffer(compressed_jp2_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    compressed_webp_np = cv2.imdecode(np.frombuffer(compressed_webp_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Convert to grayscale for quality metrics
    gray_compressed_jp2 = cv2.cvtColor(compressed_jp2_np, cv2.COLOR_RGB2GRAY)
    gray_compressed_webp = cv2.cvtColor(compressed_webp_np, cv2.COLOR_RGB2GRAY)

    # 🔹 Compute quality metrics
    psnr_jp2 = psnr(gray_original, gray_compressed_jp2, data_range=255)
    psnr_webp = psnr(gray_original, gray_compressed_webp, data_range=255)

    # Fix SSIM error by setting win_size dynamically
    min_win_size = min(7, min(image_np.shape[:2]))  # Ensures SSIM works on small images
    if min_win_size % 2 == 0:
        min_win_size -= 1  # Ensure odd value for SSIM

    ssim_jp2 = ssim(image_np, compressed_jp2_np, data_range=image_np.max() - image_np.min(), win_size=min_win_size, channel_axis=2)
    ssim_webp = ssim(image_np, compressed_webp_np, data_range=image_np.max() - image_np.min(), win_size=min_win_size, channel_axis=2)

    # Simulate download time (assuming 100 Mbps)
    download_time_jp2 = compressed_jp2_size / (12.5 * 1024)  # in sec
    download_time_webp = compressed_webp_size / (12.5 * 1024)  # in sec

    # ⏳ Time calculations
    upload_time = end_upload - start_upload
    compression_time = end_compress - start_compress

    # 📊 Display Results
    st.write(f"📏 Original Size: **{original_size:.2f} KB**")
    
    st.write(f"✅ **JPEG2000 Compressed Size:** {compressed_jp2_size:.2f} KB (**{(compressed_jp2_size / original_size) * 100:.2f}%** of original)")
    st.write(f"🎯 **JPEG2000 PSNR:** {psnr_jp2:.2f} dB")
    st.write(f"🔍 **JPEG2000 SSIM:** {ssim_jp2:.4f}")
    
    st.write(f"✅ **WebP Compressed Size:** {compressed_webp_size:.2f} KB (**{(compressed_webp_size / original_size) * 100:.2f}%** of original)")
    st.write(f"🎯 **WebP PSNR:** {psnr_webp:.2f} dB")
    st.write(f"🔍 **WebP SSIM:** {ssim_webp:.4f}")

    st.write(f"⏳ Upload Time: {upload_time:.4f} sec")
    st.write(f"⚡ Compression Time: {compression_time:.4f} sec")
    st.write(f"⬇️ Simulated Download Time (JPEG2000): {download_time_jp2:.4f} sec")
    st.write(f"⬇️ Simulated Download Time (WebP): {download_time_webp:.4f} sec")

    # 📥 Download buttons
    st.download_button("💾 Download JPEG2000", compressed_jp2_bytes, "compressed.jp2", "image/jp2")
    st.download_button("💾 Download WebP", compressed_webp_bytes, "compressed.webp", "image/webp")
