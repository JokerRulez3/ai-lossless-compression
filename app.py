import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🔗 AI-Based Lossless Image Compression with Quality Comparison")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    start_upload = time.time()
    
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Original Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)
    
    # Measure original size
    uploaded_bytes = uploaded_file.getbuffer().nbytes
    original_size_kb = uploaded_bytes / 1024  # Convert bytes to KB

    end_upload = time.time()
    
    # Compression Step
    start_compression = time.time()
    
    # Compress using JPEG2000 (lossless)
    _, compressed_image = cv2.imencode(".webp", image_np, [cv2.IMWRITE_WEBP_QUALITY, 80])

    compressed_bytes = io.BytesIO(compressed_image)
    compressed_size_kb = compressed_bytes.getbuffer().nbytes / 1024  # Convert bytes to KB

    end_compression = time.time()
    
    # Decompress to compare quality
    compressed_np = cv2.imdecode(np.frombuffer(compressed_bytes.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Ensure images have same shape
    if image_np.shape != compressed_np.shape:
        compressed_np = cv2.resize(compressed_np, (image_np.shape[1], image_np.shape[0]))

    # Convert to grayscale for SSIM comparison
    gray_original = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_compressed = cv2.cvtColor(compressed_np, cv2.COLOR_RGB2GRAY)

    # Compute quality metrics
    psnr_value = psnr(gray_original, gray_compressed, data_range=255)
    ssim_value = ssim(image_np, compressed_np, data_range=image_np.max() - image_np.min())

    # Calculate times
    upload_time = end_upload - start_upload
    compression_time = end_compression - start_compression

    # Simulated download time (assume 50 Mbps speed)
    download_time = (compressed_size_kb * 8) / (50 * 1024)

    # Display compression stats
    st.markdown(f"""
    - 📏 **Original Size:** {original_size_kb:.2f} KB  
    - ✅ **Compressed Size:** {compressed_size_kb:.2f} KB  
    - 📉 **Compression Ratio:** {100 * (compressed_size_kb / original_size_kb):.2f}% of original size  
    - 🎯 **PSNR (Peak Signal-to-Noise Ratio):** {psnr_value:.2f} dB  
    - 🔍 **SSIM (Structural Similarity Index):** {ssim_value:.4f}  
    - ⏳ **Upload Time:** {upload_time:.4f} sec  
    - ⚡ **Compression Time:** {compression_time:.4f} sec  
    - ⬇️ **Simulated Download Time:** {download_time:.4f} sec  
    """)

    # Show compressed image
    st.image(compressed_np, caption="🖼️ Compressed Image", use_column_width=True)

    # Download button
    st.download_button("💾 Download Compressed Image", compressed_bytes, "compressed.jp2", "image/jp2")
