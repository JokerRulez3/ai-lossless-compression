import streamlit as st
from compression import ai_compress, ai_decompress

st.title("AI-Based Lossless Compression Demo")

uploaded_file = st.file_uploader("Upload an image for compression", type=["png", "jpg"])

if uploaded_file:
    compressed_data = ai_compress(uploaded_file)
    st.download_button("Download Compressed File", compressed_data, "compressed.ai")

    decompressed_data = ai_decompress(compressed_data)
    st.download_button("Download Decompressed File", decompressed_data, "original_file.jpg")
