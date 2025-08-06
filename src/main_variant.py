import os
from config import RANDOM_SEED, N_EPOCHS, BATCH_SIZE, IMAGES_DIR, MASKS_DIR
from model.unet_mamba_variants import UNetMamba
from utils.metrics import dice_score
from preprocessing.dataset import HepaticVessel2DDataset
from utils.plotting import load_results, plot_metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import csv

def train_variant(variant, mode):
    print(f"\n🔁 Entrenando modelo: {variant.upper()} con modo: {mode.upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from model import mamba_block
    mamba_block.MAMBA_VARIANT = variant

    model = UNetMamba(mode=mode, strategy="integrate").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = HepaticVessel2DDataset(IMAGES_DIR, MASKS_DIR)
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, random_state=RANDOM_SEED
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    results = []
    best_dice = 0.0

    for epoch in range(N_EPOCHS):
        # ENTRENAMIENTO
        model.train()
        epoch_loss, epoch_dice = 0, 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            dice = dice_score(output, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice.item()

        # VALIDACIÓN
        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = criterion(output, mask)
                dice = dice_score(output, mask)
                
                val_loss += loss.item()
                val_dice += dice.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        results.append([epoch + 1, avg_loss, avg_dice, avg_val_loss, avg_val_dice])
        print(f"📊 {variant}-{mode} | Epoch {epoch+1}/{N_EPOCHS}")
        print(f"    Train - Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")
        print(f"    Val   - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

        # Guardar mejor modelo
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_model_path = os.path.join("outputs", "models", f"best_hepatic_model_{variant}_{mode}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    💾 Nuevo mejor modelo! Dice: {best_dice:.4f}")

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # Guardar modelo final
    model_path = os.path.join("outputs", "models", f"final_hepatic_model_{variant}_{mode}.pth")
    torch.save(model.state_dict(), model_path)

    csv_path = os.path.join("outputs", "results", f"results_{variant}_{mode}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Train_Dice", "Val_Loss", "Val_Dice"])
        writer.writerows(results)


if __name__ == "__main__":
    combinations = [
        ("simple", "enc"),
        ("simple", "dec"),
        ("simple", "full"),
        # ("full", "enc"),
        # ("full", "dec"),
        # ("full", "full"),
        ("v", "enc"),
        ("v", "dec"),
        ("v", "full"),
    ]

    for variant, mode in combinations:
        train_variant(variant, mode)

    df = load_results([f"{v}_{m}" for v, m in combinations])
    plot_metrics(df)

    print("\n🔎 Mejores resultados por modelo:")
    best_results = []
    for variant in df["Variant"].unique():
        sub = df[df["Variant"] == variant]
        best = sub.loc[sub["Dice"].idxmax()]
        print(f"🟢 {variant.upper()} → Epoch {int(best['Epoch'])} | Dice: {best['Dice']:.4f} | Loss: {best['Loss']:.4f}")
        best_results.append([variant, int(best["Epoch"]), best["Loss"], best["Dice"]])

    with open(os.path.join("outputs", "results", "compare_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "BestEpoch", "BestLoss", "BestDice"])
        writer.writerows(best_results)

    print("\n📁 Resumen guardado en outputs/results/compare_results.csv")
