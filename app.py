import streamlit as st
from PIL import Image
import torch
import cv2
import os
from scripts.detect import detect_image
from scripts.train import Autoencoder

st.set_page_config(page_title="Detector de Livros", layout="wide")
st.title("📚 Detector de Livros Fora do Lugar")

# Carregar modelo
@st.cache_resource
def load_model():
    model = Autoencoder()
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()
    return model

model = load_model()

# Upload de múltiplas imagens
uploaded_files = st.file_uploader(
    "Envie imagens das estantes", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

threshold = st.slider("Sensibilidade da detecção", 0.001, 0.05, 0.01, 0.001)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Salvar temporariamente
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Detectar anomalias
        image_marked, score = detect_image(model, temp_path, threshold=threshold)

        # Exibir resultados
        st.subheader(uploaded_file.name)
        st.image(cv2.cvtColor(image_marked, cv2.COLOR_BGR2RGB), caption=f"Score de anomalia: {score:.6f}")
