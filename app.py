import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("🔗 AI-Based Lossless Image Compression")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # Compress using JPEG2000 (lossless)
    _, compressed_image = cv2.imencode(".jp2", image_np, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])

    # Convert to bytes
    compressed_bytes = io.BytesIO(compressed_image)

    # Show compression stats
    st.write(f"✅ Compressed Size: {compressed_bytes.getbuffer().nbytes} bytes")

    # Download button
    st.download_button("💾 Download Compressed Image", compressed_bytes, "compressed.jp2", "image/jp2")
