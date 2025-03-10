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

    # Get original file size
    uploaded_file.seek(0)  # Reset pointer
    original_size = len(uploaded_file.getvalue())

    # Convert to numpy array
    image_np = np.array(image)

    # Apply JPEG2000 compression with quality factor
    compression_quality = 50  # 0-100 (lower means more compression)
    encode_param = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_quality]
    success, compressed_image = cv2.imencode(".jp2", image_np, encode_param)

    if success:
        # Convert to bytes
        compressed_bytes = io.BytesIO(compressed_image)
        compressed_size = compressed_bytes.getbuffer().nbytes

        # Calculate compression ratio
        compression_ratio = (compressed_size / original_size) * 100 if original_size > 0 else 0

        # Show original and compressed sizes
        st.write(f"📏 **Original Size:** {original_size / 1024:.2f} KB")
        st.write(f"✅ **Compressed Size:** {compressed_size / 1024:.2f} KB")
        st.write(f"📉 **Compression Ratio:** {compression_ratio:.2f}% of original size")

        # Download button for JP2
        st.download_button("💾 Download Compressed Image (JP2)", compressed_bytes, "compressed.jp2", "image/jp2")

        # --- Convert JP2 back to JPG ---
        compressed_bytes.seek(0)  # Reset file pointer
        compressed_image_np = np.frombuffer(compressed_bytes.read(), dtype=np.uint8)
        decompressed_image = cv2.imdecode(compressed_image_np, cv2.IMREAD_COLOR)

        if decompressed_image is not None:
            # Convert OpenCV image to PIL for Streamlit
            decompressed_pil = Image.fromarray(cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB))

            # Show decompressed image
            st.image(decompressed_pil, caption="Decompressed Image (JPG)", use_column_width=True)

            # Convert decompressed image to bytes for downloading
            decompressed_io = io.BytesIO()
            decompressed_pil.save(decompressed_io, format="JPEG", quality=90)  # High-quality JPG
            decompressed_io.seek(0)

            # Download button for JPG
            st.download_button("📥 Download Decompressed Image (JPG)", decompressed_io, "decompressed.jpg", "image/jpeg")
        else:
            st.error("⚠️ Decompression failed. The compressed file may not be valid.")
    else:
        st.error("⚠️ Compression failed. Try a different image or settings.")
