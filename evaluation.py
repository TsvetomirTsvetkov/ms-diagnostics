import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T

# Настройки
TEST_DIRS = [f"test0{i}" for i in range(1, 14)]
ROOT_DIR = "."
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset Class ===
class MRIDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.files = []
        for folder in root_dirs:
            path = os.path.join(ROOT_DIR, folder, "orig")
            self.files.extend(glob.glob(os.path.join(path, "*.nii.gz")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = nib.load(file).get_fdata()
        img = np.array(img, dtype=np.float32)

        # Избираме централна 2D срезка
        slice_idx = img.shape[2] // 2
        img = img[:, :, slice_idx]
        img = np.expand_dims(img, axis=0)  # [1, H, W]

        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        if self.transform:
            img = self.transform(torch.tensor(img))
        return img, file  # Връщаме и пътя към файла за зареждане на маската


# === Autoencoder Model ===
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# === Load Mask Function ===
def load_mask(file_path):
    # Създаване на път за маската (например, ако името на маската съвпада с изображението, но е в различна директория)
    mask_path = file_path.replace("orig", "mask")
    if os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
        return np.array(mask, dtype=np.float32)
    else:
        print(f"Mask not found for {file_path}")
        return np.zeros((128, 128))  # Връща празна маска, ако няма налична маска


# === Evaluate Function ===
def evaluate_model(model, dataloader):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_losses = []
    all_labels = []
    with torch.no_grad():
        for imgs, file_paths in tqdm(dataloader, desc="Evaluating"):  # Добавяме file_paths
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss_per_image = loss.mean(dim=(1, 2, 3)).cpu().numpy()
            all_losses.extend(loss_per_image)

            # Зареждаме маската за изображението (ако имаме маска за лезия)
            labels = []
            for img_path in file_paths:  # Преминаваме през пътищата на файловете
                mask = load_mask(img_path)  # Зареждаме маската
                label = np.any(mask > 0)  # Ако маската има стойности, значи има лезия
                labels.append(label)
            all_labels.extend(labels)

    # Изчисляване на точността:
    all_losses = np.array(all_losses)
    avg_loss = np.mean(all_losses)
    print(f"\nAvg reconstruction error: {avg_loss:.4f}")
    
    threshold = 0.05  # Праг за дефиниране на anomaly
    preds = (all_losses > threshold).astype(int)
    
    TP = np.sum((preds == 1) & (all_labels == 1))
    TN = np.sum((preds == 0) & (all_labels == 0))
    FP = np.sum((preds == 1) & (all_labels == 0))
    FN = np.sum((preds == 0) & (all_labels == 1))
    
    accuracy = (TP + TN) / len(all_labels)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


# === Main Function ===
def main():
    transform = T.Compose([T.Resize((128, 128))])

    test_dataset = MRIDataset(TEST_DIRS, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = Autoencoder().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Training model needed...")

    print("Evaluating on test data...")
    avg_loss, accuracy = evaluate_model(model, test_loader)
    print(f"Evaluation complete. Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
