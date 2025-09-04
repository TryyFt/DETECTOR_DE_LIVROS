import torch
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def detect_image(model, image_path, threshold=0.01):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        loss_map = (output - img_tensor).pow(2).mean(dim=1).squeeze().numpy()
        anomaly_score = loss_map.mean()

    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    mask = (loss_map > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img_cv, (x,y), (x+w, y+h), (0,0,255), 2)

    return img_cv, anomaly_score
