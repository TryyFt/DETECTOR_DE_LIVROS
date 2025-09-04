import cv2
import os

input_folder = "data/normal"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    # Redimensionar para 128x128
    img = cv2.resize(img, (128, 128))
    cv2.imwrite(os.path.join(output_folder, file), img)

print("Pré-processamento concluído!")
