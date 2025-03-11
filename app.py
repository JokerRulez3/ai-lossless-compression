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
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Calculate original file size
    uploaded_file.seek(0)
    original_size = len(uploaded_file.read())  # in bytes
    uploaded_file.seek(0)

    # Start compression
    start_compression = time.time()

    # ✅ Compress using JPEG2000 (optimized settings)
    _, jp2_image = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 50])
    jp2_bytes = io.BytesIO(jp2_image)
    jp2_size = jp2_bytes.getbuffer().nbytes  # in bytes

    # ✅ Compress using WebP (80% quality)
    _, webp_image = cv2.imencode(".webp", image_np, [cv2.IMWRITE_WEBP_QUALITY, 80])
    webp_bytes = io.BytesIO(webp_image)
    webp_size = webp_bytes.getbuffer().nbytes  # in bytes

    end_compression = time.time()

    # Convert compressed images back to numpy for quality assessment
    compressed_jp2 = cv2.imdecode(np.frombuffer(jp2_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    compressed_webp = cv2.imdecode(np.frombuffer(webp_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    # Convert to grayscale for quality metrics
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_jp2 = cv2.cvtColor(compressed_jp2, cv2.COLOR_RGB2GRAY)
    gray_webp = cv2.cvtColor(compressed_webp, cv2.COLOR_RGB2GRAY)

    # ✅ Compute quality metrics (handling SSIM errors for small images)
    min_win_size = min(7, min(image_np.shape[:2]))  # Ensures SSIM works on small images
    psnr_jp2 = psnr(gray_original, gray_jp2, data_range=255)
    ssim_jp2 = ssim(image_np, compressed_jp2, data_range=image_np.max() - image_np.min(), win_size=min_win_size)

    psnr_webp = psnr(gray_original, gray_webp, data_range=255)
    ssim_webp = ssim(image_np, compressed_webp, data_range=image_np.max() - image_np.min(), win_size=min_win_size)

    # ✅ Calculate compression ratios
    jp2_ratio = (jp2_size / original_size) * 100
    webp_ratio = (webp_size / original_size) * 100

    # ✅ Simulate download time (assuming 50 Mbps speed)
    download_speed_mbps = 50
    download_speed_bps = download_speed_mbps * (1024 ** 2) / 8  # Convert to bytes/sec
    jp2_download_time = jp2_size / download_speed_bps
    webp_download_time = webp_size / download_speed_bps

    # ✅ Display results
    st.write(f"📏 **Original Size:** {original_size / 1024:.2f} KB")
    st.write(f"✅ **JPEG2000 Compressed Size:** {jp2_size / 1024:.2f} KB ({jp2_ratio:.2f}% of original)")
    st.write(f"🎯 **JPEG2000 PSNR:** {psnr_jp2:.2f} dB")
    st.write(f"🔍 **JPEG2000 SSIM:** {ssim_jp2:.4f}")

    st.write(f"✅ **WebP Compressed Size:** {webp_size / 1024:.2f} KB ({webp_ratio:.2f}% of original)")
    st.write(f"🎯 **WebP PSNR:** {psnr_webp:.2f} dB")
    st.write(f"🔍 **WebP SSIM:** {ssim_webp:.4f}")

    # ✅ Timing Stats
    upload_time = start_compression - start_upload
    compression_time = end_compression - start_compression
    st.write(f"⏳ **Upload Time:** {upload_time:.4f} sec")
    st.write(f"⚡ **Compression Time:** {compression_time:.4f} sec")
    st.write(f"⬇️ **JPEG2000 Simulated Download Time:** {jp2_download_time:.4f} sec")
    st.write(f"⬇️ **WebP Simulated Download Time:** {webp_download_time:.4f} sec")

    # ✅ Download buttons
    st.download_button("💾 Download JPEG2000", jp2_bytes, "compressed.jp2", "image/jp2")
    st.download_button("💾 Download WebP", webp_bytes, "compressed.webp", "image/webp")
