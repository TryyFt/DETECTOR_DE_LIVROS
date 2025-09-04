import torch
import cv2
from scripts.detect import detect_image
from scripts.train import Autoencoder  # mesma classe do train.py

# Carregar modelo
model = Autoencoder()
model.load_state_dict(torch.load("models/autoencoder.pth"))

image_path = "C:/Users/Gamer/Desktop/DETECTOR DE LIVROS/data/este/colecao-de-livros-empilhados-em-ambientes-fechados.jpg"

 # exemplo
image_marked, score = detect_image(model, image_path)

print(f"Score de anomalia: {score:.6f}")
cv2.imshow("Detecção", image_marked)
cv2.waitKey(0)
cv2.destroyAllWindows()
