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

    # Load image and display
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Original Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Calculate original size
    uploaded_file.seek(0, io.SEEK_END)  # Move to end to get size
    original_size = uploaded_file.tell() / 1024  # Convert bytes to KB

    start_compression = time.time()

    # Compress using JPEG2000 (lossless)
    _, compressed_jp2 = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
    compressed_jp2_size = len(compressed_jp2) / 1024  # Convert bytes to KB

    # Compress using WebP (lossy, 80% quality)
    _, compressed_webp = cv2.imencode(".webp", image_np, [cv2.IMWRITE_WEBP_QUALITY, 80])
    compressed_webp_size = len(compressed_webp) / 1024  # Convert bytes to KB

    end_compression = time.time()

    # Convert compressed images back to numpy for comparison
    compressed_jp2_np = cv2.imdecode(np.frombuffer(compressed_jp2, np.uint8), cv2.IMREAD_COLOR)
    compressed_webp_np = cv2.imdecode(np.frombuffer(compressed_webp, np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale for SSIM (required for fair comparison)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    compressed_jp2_gray = cv2.cvtColor(compressed_jp2_np, cv2.COLOR_RGB2GRAY)
    compressed_webp_gray = cv2.cvtColor(compressed_webp_np, cv2.COLOR_RGB2GRAY)

    # Compute PSNR & SSIM
    psnr_jp2 = psnr(image_gray, compressed_jp2_gray, data_range=255)
    ssim_jp2 = ssim(image_gray, compressed_jp2_gray, data_range=255)

    psnr_webp = psnr(image_gray, compressed_webp_gray, data_range=255)
    ssim_webp = ssim(image_gray, compressed_webp_gray, data_range=255)

    # Timing calculations
    upload_time = start_compression - start_upload
    compression_time = end_compression - start_compression

    # Display results
    st.write(f"📏 **Original Size:** {original_size:.2f} KB")
    st.write(f"✅ **JPEG2000 Compressed Size:** {compressed_jp2_size:.2f} KB ({(compressed_jp2_size/original_size)*100:.2f}% of original)")
    st.write(f"🎯 **JPEG2000 PSNR:** {psnr_jp2:.2f} dB")
    st.write(f"🔍 **JPEG2000 SSIM:** {ssim_jp2:.4f}")

    st.write(f"✅ **WebP Compressed Size:** {compressed_webp_size:.2f} KB ({(compressed_webp_size/original_size)*100:.2f}% of original)")
    st.write(f"🎯 **WebP PSNR:** {psnr_webp:.2f} dB")
    st.write(f"🔍 **WebP SSIM:** {ssim_webp:.4f}")

    st.write(f"⏳ **Upload Time:** {upload_time:.4f} sec")
    st.write(f"⚡ **Compression Time:** {compression_time:.4f} sec")

    # Download buttons
    st.download_button("💾 Download JPEG2000", io.BytesIO(compressed_jp2), "compressed.jp2", "image/jp2")
    st.download_button("💾 Download WebP", io.BytesIO(compressed_webp), "compressed.webp", "image/webp")
