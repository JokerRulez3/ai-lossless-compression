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
    # Measure upload time
    upload_start = time.time()
    image = Image.open(uploaded_file).convert("RGB")
    image_bytes = uploaded_file.getvalue()
    upload_end = time.time()
    
    original_size_kb = len(image_bytes) / 1024  # Convert to KB
    st.image(image, caption=f"Original Image ({original_size_kb:.2f} KB)", use_column_width=True)
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Measure compression time
    compress_start = time.time()
    _, compressed_image = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
    compressed_bytes = io.BytesIO(compressed_image)
    compress_end = time.time()
    
    compressed_size_kb = compressed_bytes.getbuffer().nbytes / 1024  # Convert to KB
    compression_ratio = (compressed_size_kb / original_size_kb) * 100
    
    # Measure download time (simulated)
    download_start = time.time()
    compressed_bytes.seek(0)
    download_end = time.time()
    
    # Convert back to JPG for visualization
    compressed_bytes.seek(0)
    compressed_image_np = cv2.imdecode(np.frombuffer(compressed_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    compressed_image_pil = Image.fromarray(cv2.cvtColor(compressed_image_np, cv2.COLOR_BGR2RGB))
    
    # Display compression stats
    st.write(f"📏 Original Size: {original_size_kb:.2f} KB")
    st.write(f"✅ Compressed Size: {compressed_size_kb:.2f} KB")
    st.write(f"📉 Compression Ratio: {compression_ratio:.2f}% of original size")
    st.write(f"⏳ Upload Time: {upload_end - upload_start:.4f} sec")
    st.write(f"⚡ Compression Time: {compress_end - compress_start:.4f} sec")
    st.write(f"⬇️ Simulated Download Time: {download_end - download_start:.4f} sec")
    
    # Display compressed image
    st.image(compressed_image_pil, caption="Compressed Image (Converted back to JPG for viewing)", use_column_width=True)
    
    # Download buttons
    st.download_button("💾 Download Compressed JP2", compressed_bytes, "compressed.jp2", "image/jp2")
    st.download_button("💾 Download Converted JPG", io.BytesIO(compressed_image_pil.tobytes()), "compressed.jpg", "image/jpeg")
