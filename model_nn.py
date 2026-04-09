import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 80  # Increased for better reconstruction
LEARNING_RATE = 1e-3
TRAIN_DIR = "cable/train/good"  # Path to your 224 good images
TEST_IMG_PATH = "cable/test/missing_wire/003.png" # Path to a cable with a defect

# --- DATASET ---
import torchvision.transforms as T
from PIL import Image

class CableDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        
        # Definicja losowych transformacji (augmentacja)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(90), # Kable obracają się o losowy kąt
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor() # Zamienia na tensor [0, 1] w formacie (C, H, W)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Wczytanie obrazu
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplikacja augmentacji
        img_tensor = self.transform(img)
        
        # Zwracamy ten sam zmodyfikowany obraz jako wejście i cel
        return img_tensor, img_tensor

# --- MODEL INITIALIZATION ---
class ExtremeBottleneckAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Koder (Splotowy) - zmniejsza rozdzielczość do 8x8
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        
        # EKSTREMALNE WĄSKIE GARDŁO: 256 kanałów * 8 * 8 pikseli = 16384 -> 128
        self.encoder_linear = nn.Linear(16384, 128)
        
        # Dekoder - odbudowa
        self.decoder_linear = nn.Linear(128, 16384)
        self.unflatten = nn.Unflatten(1, (256, 8, 8))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Kompresja
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_linear(x) # Zostaje 128 liczb
        
        # Rekonstrukcja
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

model = ExtremeBottleneckAutoencoder().to(DEVICE)

def train():
    dataset = CableDataset(TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training started on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.6f}")
    
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("Model saved as autoencoder_model.pth")

def generate_anomaly_mask(image_path, threshold=80):
    model.load_state_dict(torch.load("autoencoder_model.pth", map_location=DEVICE))
    model.eval()
    
    # Wczytanie obrazu
    original_bgr = cv2.imread(image_path)
    h, w, _ = original_bgr.shape
    img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Przygotowanie wejścia do sieci (256x256)
    img_input = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.tensor(img_input.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Rekonstrukcja
    with torch.no_grad():
        reconstructed_tensor = model(img_tensor)
    
    # Powrót do numpy (rozdzielczość 256x256)
    reco = reconstructed_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    reco = (reco * 255).astype(np.uint8)
    
    # --- ZMIANA: Obliczanie różnicy w 256x256 ---
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
    gray_reco = cv2.cvtColor(reco, cv2.COLOR_RGB2GRAY)
    
    # Mapa różnic 
    diff = cv2.absdiff(gray_input, gray_reco)
    
    # Rozmycie mapy różnic, aby zignorować drobny szum rekonstrukcji
    diff = cv2.GaussianBlur(diff, (11, 11), 0)
    
    # Progowanie (binaryzacja) - nadal w 256x256
    _, mask_256 = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # --- SKALOWANIE W GÓRĘ ---
    # Skalujemy dopiero gotową maskę przy użyciu najbliższego sąsiada (aby nie tworzyć szarych pikseli)
    mask_full = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # --- MASKOWANIE KABLA (Usuwanie szumu z tła) ---
    gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Wyznaczamy obszar kabla (zakładamy, że tło jest najciemniejsze)
    _, cable_area = cv2.threshold(gray_orig, 20, 255, cv2.THRESH_BINARY)
    
    # Zostawiamy anomalie tylko wewnątrz fizycznego obszaru kabla
    final_mask = cv2.bitwise_and(mask_full, mask_full, mask=cable_area)
    
    # Morfologiczne czyszczenie końcowe (usuwa kropki o średnicy mniejszej niż jądro)
    kernel = np.ones((15, 15), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Skalujemy zrekonstruowany obraz tylko na potrzeby wizualizacji (do wykresu)
    reco_vis = cv2.resize(reco, (w, h))
    
    return img_rgb, reco_vis, final_mask

# --- RUN EVERYTHING ---
if __name__ == "__main__":
    # 1. Train if weights don't exist
    if not os.path.exists("autoencoder_model.pth"):
        train()
    else:
        print("Weights found, skipping training.")

    # 2. Run detection
    if os.path.exists(TEST_IMG_PATH):
        orig, reco, mask = generate_anomaly_mask(TEST_IMG_PATH)
        
        # 3. Visual Results
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(orig); plt.title("Original (with defect)")
        plt.subplot(1, 3, 2); plt.imshow(reco); plt.title("Reconstructed (clean version)")
        plt.subplot(1, 3, 3); plt.imshow(mask, cmap='gray'); plt.title("Anomaly Mask (0-255)")
        plt.show()
    else:
        print(f"Test image not found at {TEST_IMG_PATH}")