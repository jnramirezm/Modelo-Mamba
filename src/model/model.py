class HepaticVessel2DDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, include_empty=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.include_empty = include_empty
        self.slices = []  # lista de (img_path, mask_path, slice_index)

        # Recorremos todos los archivos
        for filename in sorted(os.listdir(images_dir)):
            if not filename.endswith(".nii.gz"):
                continue

            img_path = os.path.join(images_dir, filename)
            msk_path = os.path.join(masks_dir, filename)

            # Cargar volumen y máscara
            img = nib.load(img_path).get_fdata()
            msk = nib.load(msk_path).get_fdata()

            # Verificamos cortes útiles
            for i in range(img.shape[2]):
                if self.include_empty or np.max(msk[:, :, i]) > 0:
                    self.slices.append((img_path, msk_path, i))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, msk_path, slice_idx = self.slices[idx]

        img = nib.load(img_path).get_fdata()
        msk = nib.load(msk_path).get_fdata()

        img_slice = img[:, :, slice_idx]
        msk_slice = msk[:, :, slice_idx]

        # Normalizar imagen entre 0 y 1
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

        # Expandir dimensiones → [C, H, W]
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).float()
        msk_tensor = torch.from_numpy((msk_slice > 0).astype(np.float32)).unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)
            msk_tensor = self.transform(msk_tensor)

        # Reduccion
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        msk_tensor = F.interpolate(msk_tensor.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        return img_tensor, msk_tensor


from torch.utils.data import DataLoader
from config import IMAGES_DIR, MASKS_DIR
from preprocessing.dataset import HepaticVessel2DDataset

# Usar las rutas del config que son multiplataforma
images_dir = IMAGES_DIR
masks_dir = MASKS_DIR

dataset = HepaticVessel2DDataset(images_dir, masks_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Probar una muestra
for img, msk in loader:
    print("Imagen shape:", img.shape)  # [B, 1, H, W]
    print("Máscara shape:", msk.shape)  # [B, 1, H, W]
    break


import torch
import torch.nn as nn
import torch.nn.functional as F



def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


from torch.utils.data import DataLoader
import torch.optim as optim

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Dataset y dataloader (usando el que ya armaste)
dataset = HepaticVessel2DDataset(images_dir, masks_dir)
# loader = DataLoader(dataset, batch_size=4, shuffle=True)

small_dataset = torch.utils.data.Subset(dataset, range(200))
loader = DataLoader(small_dataset, batch_size=4, shuffle=True)

# Entrenamiento
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    epoch_dice = 0

    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        output = model(img)
        loss = criterion(output, mask)
        dice = dice_score(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice.item()

    print(f"🟢 Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss / len(loader):.4f} | Dice: {epoch_dice / len(loader):.4f}")


with torch.no_grad():
    model.eval()
    for img, msk in loader:
        img = img.to(device)
        pred = model(img)
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        print("Pred max:", pred_bin.max())
        break

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for img, mask in loader:
        img = img.to(device)
        pred = model(img)
        pred_sigmoid = torch.sigmoid(pred)
        pred_bin = (pred_sigmoid > 0.5).float()

        # Mostrar primer ejemplo del batch
        i = 0
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img[i][0].cpu(), cmap='gray')
        plt.title("Imagen")

        plt.subplot(1, 3, 2)
        plt.imshow(mask[i][0].cpu(), cmap='gray')
        plt.title("Máscara real")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_bin[i][0].cpu(), cmap='gray')
        plt.title("Predicción")

        plt.tight_layout()
        plt.show()
        break