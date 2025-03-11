import streamlit as st
import numpy as np
import cv2
import io
import time
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

st.title("🔗 AI-Based Lossless Image Compression")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    original_size = len(uploaded_file.getvalue()) / 1024  # KB
    
    # Convert to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Start compression timer
    start_compression = time.time()
    
    # Compress to WebP
    _, compressed_image = cv2.imencode(".webp", image_bgr, [cv2.IMWRITE_WEBP_QUALITY, 80])
    compressed_size = len(compressed_image) / 1024  # KB
    
    # End compression timer
    end_compression = time.time()
    
    # Decompress the WebP image
    decompressed_image = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
    decompressed_size = decompressed_image.size * decompressed_image.itemsize / 1024  # KB
    
    # Convert decompressed image to RGB for metric calculations
    decompressed_rgb = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for SSIM/PSNR calculation
    gray_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_decompressed = cv2.cvtColor(decompressed_image, cv2.COLOR_BGR2GRAY)
    
    # Compute quality metrics
    psnr_value = psnr(gray_original, gray_decompressed, data_range=255)
    ssim_value = ssim(gray_original, gray_decompressed, data_range=255)
    
    # Display results
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(decompressed_rgb, caption="Decompressed Image", use_column_width=True)
    
    st.markdown(f"""
    📏 **Original Size**: {original_size:.2f} KB  
    ✅ **Compressed Size**: {compressed_size:.2f} KB ({(compressed_size/original_size)*100:.2f}% of original)  
    🔄 **Decompressed Size**: {decompressed_size:.2f} KB  
    🎯 **PSNR (Peak Signal-to-Noise Ratio)**: {psnr_value:.2f} dB  
    🔍 **SSIM (Structural Similarity Index)**: {ssim_value:.4f}  
    ⏳ **Compression Time**: {end_compression - start_compression:.4f} sec  
    """)
