import streamlit as st
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Load the VAE model
@st.cache_resource
def load_vae():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    return vae

vae = load_vae()
vae.eval()

# Image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize([0.5], [0.5]),
])

postprocess = transforms.Compose([
    transforms.Normalize([-1], [2]),
    transforms.ToPILImage(),
])

st.title("🔗 AI-Based Image Compression & Decompression")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess the image
    input_image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        # Encode (compress) the image
        latent_representation = vae.encode(input_image).latent_dist.sample()

        # Decode (decompress) the image
        reconstructed_image = vae.decode(latent_representation).sample()

    # Post-process the reconstructed image
    reconstructed_image = reconstructed_image.squeeze(0)
    reconstructed_image = postprocess(reconstructed_image)

    st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

    # Calculate metrics
    original_np = np.array(image.resize((512, 512)))
    reconstructed_np = np.array(reconstructed_image)

    psnr_value = psnr(original_np, reconstructed_np, data_range=255)
    ssim_value = ssim(original_np, reconstructed_np, multichannel=True, data_range=255)

    st.write(f"🎯 PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    st.write(f"🔍 SSIM (Structural Similarity Index): {ssim_value:.4f}")
