import os
import glob
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn

# Настройки
ROOT_DIR = "."
MODEL_PATH = "model.pth"
TEST_DIRS = [f"test0{i}" for i in range(1, 14)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return img


# === Train Function ===
def train_model(model, dataloader, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


# === Evaluate Function ===
def evaluate_model(model, dataloader):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_losses = []
    patient_errors = defaultdict(list)

    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss_per_image = loss.mean(dim=(1, 2, 3)).cpu().numpy()
            all_losses.extend(loss_per_image)

            # За всеки пациент
            for i, loss_value in enumerate(loss_per_image):
                patient_id = dataloader.dataset.files[i].split('/')[-2]  # Извеждаме ID на пациента от пътя
                patient_errors[patient_id].append(loss_value)
            
    # Извеждаме средната грешка за всеки пациент
    avg_patient_errors = {patient_id: np.mean(errors) for patient_id, errors in patient_errors.items()}
    return avg_patient_errors


# === Visualization Function ===
def visualize_diff(original, reconstructed):
    original = original.squeeze().detach().cpu().numpy()  # Добавяме .detach() и .cpu() за преобразуване в NumPy
    reconstructed = reconstructed.squeeze().detach().cpu().numpy()  # Добавяме .detach() и .cpu()

    diff = np.abs(original - reconstructed)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Difference (Anomaly Map)")
    plt.imshow(diff, cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# === Main ===
def main():
    transform = T.Compose([T.Resize((128, 128))])
    test_dataset = MRIDataset(TEST_DIRS, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = Autoencoder().to(DEVICE)
    
    # Зареждаме модел, ако съществува
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Training model needed...")

    print("Evaluating on test data...")
    patient_errors = evaluate_model(model, test_loader)

    # Търсим пациента с най-висока грешка
    max_error_patient = min(patient_errors, key=patient_errors.get)
    print(f"\nPatient with highest reconstruction error (potential MS): {max_error_patient}")

    # Визуализираме изображения от пациента с най-висока грешка
    patient_files = [file for file in test_dataset.files if max_error_patient in file]
    for file in patient_files:
        img = nib.load(file).get_fdata()
        slice_idx = img.shape[2] // 2
        img_slice = img[:, :, slice_idx]
        img_tensor = torch.tensor(np.expand_dims(img_slice, axis=0), dtype=torch.float32).to(DEVICE)
        output = model(img_tensor)

        visualize_diff(img_tensor.cpu(), output.cpu())


if __name__ == "__main__":
    main()
