import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

st.title("🔗 AI-Based Lossless Image Compression")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Start timer for upload time
    upload_start_time = time.time()
    
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Original Image", use_column_width=True)
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Get original size
    uploaded_file.seek(0, io.SEEK_END)
    original_size = uploaded_file.tell()
    uploaded_file.seek(0)
    
    upload_time = time.time() - upload_start_time
    
    # Start timer for compression
    compression_start_time = time.time()
    
    # Compress using optimized JPEG2000 settings
    _, compressed_image = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 0])
    
    compression_time = time.time() - compression_start_time
    
    # Convert to bytes
    compressed_bytes = io.BytesIO(compressed_image)
    compressed_size = compressed_bytes.getbuffer().nbytes
    
    # Compute compression ratio
    compression_ratio = (compressed_size / original_size) * 100
    
    # Simulated download time (assume 10 Mbps network speed)
    network_speed_mbps = 10
    simulated_download_time = (compressed_size * 8) / (network_speed_mbps * 1_000_000)
    
    # Show compression stats
    st.write(f"📏 **Original Size:** {original_size / 1024:.2f} KB")
    st.write(f"✅ **Compressed Size:** {compressed_size / 1024:.2f} KB")
    st.write(f"📉 **Compression Ratio:** {compression_ratio:.2f}% of original size")
    st.write(f"⏳ **Upload Time:** {upload_time:.4f} sec")
    st.write(f"⚡ **Compression Time:** {compression_time:.4f} sec")
    st.write(f"⬇️ **Simulated Download Time:** {simulated_download_time:.4f} sec")
    
    # Download button
    st.download_button("💾 Download Compressed Image", compressed_bytes, "compressed.jp2", "image/jp2")