import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

# Transformações
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Dataset customizado
class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [os.path.join(folder, f) 
                       for f in os.listdir(folder) 
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

# Carregar dataset
dataset = CustomDataset("data/processed", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128*128*3),
            nn.Sigmoid(),
            nn.Unflatten(1, (3,128,128))
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Treinamento
for epoch in range(10):
    for imgs, _ in dataloader:
        output = model(imgs)
        loss = criterion(output, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Salvar modelo
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/autoencoder.pth")
print("Modelo salvo em models/autoencoder.pth")
