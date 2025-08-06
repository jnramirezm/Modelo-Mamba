import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from config import (IMAGES_DIR, MASKS_DIR, RANDOM_SEED, N_EPOCHS, BATCH_SIZE, 
                   MAMBA_VARIANT, IMAGE_SIZE, NORMALIZE_METHOD, USE_DATA_AUGMENTATION, NUM_WORKERS,
                   MAX_SAMPLES, USE_CACHE, LOG_INTERVAL, USE_LIGHTWEIGHT_MODEL, IS_TESTING)
from model.unet_mamba_variants import UNetMamba
from utils.metrics import dice_score
from preprocessing.dataset_v2 import HepaticVesselDatasetV2
import time

def show_config():
    """Muestra la configuración actual de manera organizada"""
    print("="*60)
    if IS_TESTING:
        print("🧪 ENTRENAMIENTO UNet + Mamba - MODO TESTING/DESARROLLO")
        print("   (Configuración optimizada para hardware limitado)")
    else:
        print("🚀 ENTRENAMIENTO UNet + Mamba - MODO PRODUCCIÓN")
        print("   (Configuración para entrenamiento completo)")
    print("="*60)
    print(f"🔁 Variante Mamba: {MAMBA_VARIANT}")
    print(f"🖼️  Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔢 Épocas: {N_EPOCHS}")
    print(f"🎯 Max muestras: {MAX_SAMPLES}")
    print(f"🔧 Normalización: {NORMALIZE_METHOD}")
    print(f"🎲 Data Augmentation: {USE_DATA_AUGMENTATION}")
    print(f"💾 Cache: {USE_CACHE}")
    print(f"⚡ Modelo ligero: {USE_LIGHTWEIGHT_MODEL}")
    print("="*60)

def main():
    # Mostrar configuración
    show_config()
    
    # Verificar disponibilidad de CUDA
    print("🔍 Verificando dispositivos disponibles...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Número de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        device = torch.device("cuda")
    else:
        print("   ⚠️  CUDA no está disponible. Posibles causas:")
        print("   - PyTorch instalado sin soporte CUDA")
        print("   - Drivers de NVIDIA no instalados")
        print("   - GPU no compatible")
        device = torch.device("cpu")
    
    print(f"🖥️  Dispositivo seleccionado: {device}")
    
    model = UNetMamba(mode="full", strategy="integrate").to(device)
    print(f"🔁 Usando modelo: UNet + Mamba ({MAMBA_VARIANT})")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Crear datasets separados para entrenamiento y validación
    print("\n📁 Creando datasets...")
    if IS_TESTING:
        print(f"   🧪 Modo testing: limitando a {MAX_SAMPLES} muestras")
    else:
        print(f"   🚀 Modo producción: usando hasta {MAX_SAMPLES} muestras")
    
    train_dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR, 
        image_size=IMAGE_SIZE,
        normalize_method=NORMALIZE_METHOD,
        is_training=True,  # Con augmentation
        include_empty=False,
        cache_data=USE_CACHE,
        max_samples=MAX_SAMPLES
    )
    
    val_dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR, 
        image_size=IMAGE_SIZE,
        normalize_method=NORMALIZE_METHOD,
        is_training=False,  # Sin augmentation
        include_empty=False,
        cache_data=USE_CACHE,
        max_samples=MAX_SAMPLES
    )

    # Dividir índices
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Mostrar estadísticas del dataset
    print(f"\n📊 Estadísticas del dataset:")
    print(f"   📚 Entrenamiento: {len(train_loader)} batches ({len(train_idx)} muestras)")
    print(f"   🔍 Validación: {len(val_loader)} batches ({len(val_idx)} muestras)")
    
    train_stats = train_dataset.get_dataset_stats()
    for key, value in train_stats.items():
        print(f"   {key}: {value}")

    print(f"\n🚀 Iniciando entrenamiento...")
    start_time = time.time()
    best_dice = 0.0
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # ENTRENAMIENTO
        model.train()
        train_loss, train_dice = 0, 0
        batch_count = 0
        
        print(f"\n🔄 Época {epoch+1}/{N_EPOCHS} - Entrenamiento:")
        
        for batch_idx, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            
            # Forward pass
            output = model(img)
            loss = criterion(output, mask)
            dice = dice_score(output, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice.item()
            batch_count += 1
            
            # Log cada LOG_INTERVAL batches
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                current_loss = train_loss / batch_count
                current_dice = train_dice / batch_count
                print(f"   Batch {batch_idx+1}/{len(train_loader)} - Loss: {current_loss:.4f}, Dice: {current_dice:.4f}")

        # VALIDACIÓN
        print(f"   🔍 Validando...")
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

        # Calcular promedios
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time

        print(f"\n   📊 Resultados Época {epoch+1}:")
        print(f"      🚂 Train - Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"      🔍 Val   - Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")
        print(f"      ⏱️  Tiempo época: {epoch_time:.1f}s | Total: {elapsed_time:.1f}s")

        # Guardar mejor modelo
        if val_dice > best_dice:
            best_dice = val_dice
            model_path = f"best_hepatic_model_{MAMBA_VARIANT}_opt.pth"
            torch.save(model.state_dict(), model_path)
            print(f"      💾 ¡Nuevo mejor modelo guardado! Dice: {best_dice:.4f}")
            
        # Limpiar memoria GPU
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Guardar modelo final
    final_model_path = f"final_hepatic_model_{MAMBA_VARIANT}_opt.pth"
    torch.save(model.state_dict(), final_model_path)
    
    total_time = time.time() - start_time
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
    print(f"   ⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   🏆 Mejor Dice Score: {best_dice:.4f}")
    print(f"   💾 Modelo final guardado: {final_model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
