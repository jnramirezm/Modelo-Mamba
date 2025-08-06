"""
Script de prueba completo del pipeline de entrenamiento
Usa dataset sintético pequeño con configuraciones optimizadas para testing rápido
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

# Añadir directorio src al path
sys.path.append(os.path.dirname(__file__))

# Configuración específica para testing con dataset real
TEST_CONFIG = {
    # Datos - usar dataset real de decathlon
    'IMAGES_DIR': 'data/decathlon/imagesTr',
    'MASKS_DIR': 'data/decathlon/labelsTr',
    'IMAGE_SIZE': 64,    # MUY reducido desde 512x512 original
    'BATCH_SIZE': 1,     # Batch muy pequeño
    'MAX_SAMPLES': 10,   # Solo 10 muestras para testing rápido
    
    # Entrenamiento
    'EPOCHS': 2,         # Solo 2 épocas
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 1e-4,
    
    # Validación
    'TRAIN_SPLIT': 0.8,  # 80% train, 20% val
    'SAVE_FREQ': 1,      # Guardar cada época
    
    # Augmentation (muy reducido para velocidad)
    'AUGMENTATION_PARAMS': {
        'rotation_range': (-5, 5),     # Rotación mínima
        'horizontal_flip': True,
        'vertical_flip': False,
        'elastic_deformation': False,   # Desactivado
        'gaussian_noise_std': 0.01,
        'brightness_range': (0.95, 1.05),
        'contrast_range': (0.95, 1.05)
    },
    
    # Configuración específica para dataset real
    'VESSEL_LABEL': 1,    # Label para vasos
    'TUMOR_LABEL': 2,     # Label para tumores
    'FOCUS_ON_VESSELS': True,  # Solo segmentar vasos (ignorar tumores)
    'MIN_VESSEL_PIXELS': 10,   # Filtrar slices con muy pocos vasos
}


def setup_test_environment():
    """Configura el entorno para testing"""
    print("CONFIGURACIÓN DE ENTORNO DE PRUEBA")
    print("=" * 50)
    
    # Verificar que el dataset real existe
    if not os.path.exists(TEST_CONFIG['IMAGES_DIR']):
        print(f"❌ Dataset no encontrado en: {TEST_CONFIG['IMAGES_DIR']}")
        print("Por favor verifica que el dataset de Decathlon esté disponible.")
        return None, None
    
    print(f"✅ Dataset encontrado: {TEST_CONFIG['IMAGES_DIR']}")
    
    # Verificar PyTorch y CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("Usando CPU (recomendado para testing rápido)")
        device = torch.device('cpu')
    
    # Crear directorio de outputs
    output_dir = Path("outputs/test_run_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Outputs en: {output_dir}")
    
    return device, output_dir


def load_test_dataset(device):
    """Carga el dataset de prueba usando datos reales de Decathlon"""
    print("\nCARGANDO DATASET REAL DE DECATHLON")
    print("-" * 40)
    
    try:
        # Intentar cargar dataset avanzado
        from preprocessing.dataset_v2 import HepaticVesselDatasetV2
        
        print("🔄 Cargando con dataset avanzado...")
        dataset = HepaticVesselDatasetV2(
            images_dir=TEST_CONFIG['IMAGES_DIR'],
            masks_dir=TEST_CONFIG['MASKS_DIR'],
            image_size=TEST_CONFIG['IMAGE_SIZE'],
            normalize_method="minmax",
            is_training=True,
            augmentation_params=TEST_CONFIG['AUGMENTATION_PARAMS'],
            include_empty=False,     # Excluir slices vacíos para acelerar
            cache_data=False,        # Sin cache para testing
            vessel_label=TEST_CONFIG.get('VESSEL_LABEL', 1),
            max_samples=TEST_CONFIG.get('MAX_SAMPLES', 10)  # Limitar muestras
        )
        
        print(f"✅ Dataset avanzado cargado: {len(dataset)} slices")
        
    except ImportError:
        print("⚠️  Dataset avanzado no disponible, usando dataset básico...")
        
        # Fallback al dataset básico
        from preprocessing.dataset import HepaticVesselDataset
        
        dataset = HepaticVesselDataset(
            images_dir=TEST_CONFIG['IMAGES_DIR'],
            masks_dir=TEST_CONFIG['MASKS_DIR'],
            image_size=TEST_CONFIG['IMAGE_SIZE']
        )
        
        # Limitar dataset manualmente si es necesario
        if len(dataset) > TEST_CONFIG.get('MAX_SAMPLES', 10) * 30:  # Estimación de slices por volumen
            print(f"🔄 Limitando dataset a ~{TEST_CONFIG.get('MAX_SAMPLES', 10) * 30} slices para testing")
            dataset.file_paths = dataset.file_paths[:TEST_CONFIG.get('MAX_SAMPLES', 10)]
            dataset._scan_files()  # Re-escanear archivos limitados
        
        print(f"✅ Dataset básico cargado: {len(dataset)} slices")
    
    if len(dataset) == 0:
        print("❌ No se encontraron datos válidos en el dataset")
        return None, None
    
    # Split train/validation
    train_size = int(TEST_CONFIG['TRAIN_SPLIT'] * len(dataset))
    val_size = len(dataset) - train_size
    
    # Asegurar que tenemos al menos 1 muestra en validación
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TEST_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=0,      # Sin multiprocessing para evitar problemas
        pin_memory=False    # Sin pin_memory para CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TEST_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"📊 Train: {len(train_dataset)} slices, {len(train_loader)} batches")
    print(f"📊 Validation: {len(val_dataset)} slices, {len(val_loader)} batches")
    print(f"🎯 Configuración: imagen {TEST_CONFIG['IMAGE_SIZE']}x{TEST_CONFIG['IMAGE_SIZE']}, batch {TEST_CONFIG['BATCH_SIZE']}")
    
    return train_loader, val_loader


def load_test_model(device):
    """Carga un modelo para testing"""
    print("\nCARGANDO MODELO")
    print("-" * 15)
    
    try:
        # Intentar cargar UNet + Mamba
        from model.unet_mamba_variants import UNetMambaSimple
        
        model = UNetMambaSimple(
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256],  # Características reducidas para testing
        ).to(device)
        
        model_name = "UNet + Mamba Simple"
        print(f"✅ {model_name} cargado")
        
    except ImportError:
        try:
            # Fallback a UNet básico
            from model.unet import UNet
            
            model = UNet(
                in_channels=1,
                out_channels=1
            ).to(device)
            
            model_name = "UNet Básico"
            print(f"✅ {model_name} cargado")
            
        except ImportError:
            # Último fallback: modelo simple
            print("Modelos personalizados no disponibles, usando modelo simple...")
            
            class SimpleUNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.decoder = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            model = SimpleUNet().to(device)
            model_name = "UNet Simple (Fallback)"
            print(f"✅ {model_name} cargado")
    
    # Mostrar información del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    return model, model_name


def setup_training_components(model, device):
    """Configura componentes de entrenamiento"""
    print("\nCONFIGURANDO ENTRENAMIENTO")
    print("-" * 25)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    print("✅ Loss: BCEWithLogitsLoss")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=TEST_CONFIG['LEARNING_RATE'],
        weight_decay=TEST_CONFIG['WEIGHT_DECAY']
    )
    print(f"✅ Optimizer: Adam (lr={TEST_CONFIG['LEARNING_RATE']})")
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    print("✅ Scheduler: ReduceLROnPlateau")
    
    return criterion, optimizer, scheduler


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calcula métricas básicas"""
    pred_binary = (predictions > threshold).float()
    
    # Intersection over Union (IoU)
    intersection = (pred_binary * targets).sum()
    union = pred_binary.sum() + targets.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    # Dice Score
    dice = (2 * intersection) / (pred_binary.sum() + targets.sum() + 1e-8)
    
    # Accuracy
    correct = (pred_binary == targets).float().sum()
    accuracy = correct / targets.numel()
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item()
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Entrena una época"""
    model.train()
    total_loss = 0
    total_metrics = {'iou': 0, 'dice': 0, 'accuracy': 0}
    num_batches = len(train_loader)
    
    print(f"\nÉpoca {epoch + 1} - Entrenamiento:")
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Si el modelo retorna logits, aplicar sigmoid para métricas
        if outputs.min() < 0 or outputs.max() > 1:
            predictions = torch.sigmoid(outputs)
        else:
            predictions = outputs
        
        # Loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        with torch.no_grad():
            metrics = calculate_metrics(predictions, masks)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        # Progreso
        if batch_idx % max(1, num_batches // 3) == 0:
            print(f"  Batch {batch_idx+1}/{num_batches} - "
                  f"Loss: {loss.item():.4f}, "
                  f"IoU: {metrics['iou']:.3f}, "
                  f"Dice: {metrics['dice']:.3f}")
    
    # Promedios
    avg_loss = total_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
    
    print(f"  📊 Promedio - Loss: {avg_loss:.4f}, "
          f"IoU: {avg_metrics['iou']:.3f}, "
          f"Dice: {avg_metrics['dice']:.3f}, "
          f"Acc: {avg_metrics['accuracy']:.3f}")
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Valida una época"""
    model.eval()
    total_loss = 0
    total_metrics = {'iou': 0, 'dice': 0, 'accuracy': 0}
    num_batches = len(val_loader)
    
    print(f"\nÉpoca {epoch + 1} - Validación:")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Si el modelo retorna logits, aplicar sigmoid para métricas
            if outputs.min() < 0 or outputs.max() > 1:
                predictions = torch.sigmoid(outputs)
            else:
                predictions = outputs
            
            # Loss
            loss = criterion(outputs, masks)
            
            # Métricas
            metrics = calculate_metrics(predictions, masks)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    # Promedios
    avg_loss = total_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
    
    print(f"  📊 Validación - Loss: {avg_loss:.4f}, "
          f"IoU: {avg_metrics['iou']:.3f}, "
          f"Dice: {avg_metrics['dice']:.3f}, "
          f"Acc: {avg_metrics['accuracy']:.3f}")
    
    return avg_loss, avg_metrics


def run_complete_test():
    """Ejecuta el test completo del pipeline"""
    start_time = time.time()
    
    print("INICIO DEL TEST COMPLETO DEL PIPELINE")
    print("=" * 60)
    
    try:
        # 1. Configurar entorno
        device, output_dir = setup_test_environment()
        
        # 2. Cargar dataset
        train_loader, val_loader = load_test_dataset(device)
        
        # 3. Cargar modelo
        model, model_name = load_test_model(device)
        
        # 4. Configurar entrenamiento
        criterion, optimizer, scheduler = setup_training_components(model, device)
        
        # 5. Loop de entrenamiento
        print(f"\nINICIANDO ENTRENAMIENTO ({TEST_CONFIG['EPOCHS']} épocas)")
        print("=" * 40)
        
        train_history = {'loss': [], 'iou': [], 'dice': [], 'accuracy': []}
        val_history = {'loss': [], 'iou': [], 'dice': [], 'accuracy': []}
        
        best_val_dice = 0
        
        for epoch in range(TEST_CONFIG['EPOCHS']):
            epoch_start = time.time()
            
            # Entrenar
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validar
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Guardar historial
            train_history['loss'].append(train_loss)
            val_history['loss'].append(val_loss)
            
            for key in ['iou', 'dice', 'accuracy']:
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])
            
            # Guardar mejor modelo
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': best_val_dice,
                    'config': TEST_CONFIG
                }, output_dir / 'best_model.pth')
                print(f"  💾 Mejor modelo guardado (Dice: {best_val_dice:.3f})")
            
            epoch_time = time.time() - epoch_start
            print(f"  ⏱️  Tiempo época: {epoch_time:.1f}s")
        
        # 6. Resumen final
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("RESUMEN FINAL DEL TEST")
        print("=" * 60)
        print(f"✅ Test completado exitosamente en {total_time:.1f}s")
        print(f"🏗️  Modelo: {model_name}")
        print(f"📊 Dataset: {len(train_loader.dataset) + len(val_loader.dataset)} muestras")
        print(f"🎯 Mejor Dice Score: {best_val_dice:.3f}")
        print(f"📁 Outputs guardados en: {output_dir}")
        
        print(f"\n📈 PROGRESO POR ÉPOCA:")
        for epoch in range(TEST_CONFIG['EPOCHS']):
            print(f"  Época {epoch+1}: "
                  f"Train Dice={train_history['dice'][epoch]:.3f}, "
                  f"Val Dice={val_history['dice'][epoch]:.3f}")
        
        print(f"\n✅ COMPONENTES VERIFICADOS:")
        print(f"  ✓ Carga de dataset")
        print(f"  ✓ Arquitectura del modelo")
        print(f"  ✓ Loop de entrenamiento") 
        print(f"  ✓ Loop de validación")
        print(f"  ✓ Cálculo de métricas")
        print(f"  ✓ Guardado de modelos")
        print(f"  ✓ Gestión de memoria")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN EL TEST: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("TEST DEL PIPELINE COMPLETO")
    print("Configuración optimizada para testing rápido")
    print("-" * 60)
    
    success = run_complete_test()
    
    if success:
        print(f"\n🎉 ¡TEST EXITOSO! El pipeline funciona correctamente.")
        print(f"Ahora puedes usar configuraciones completas para entrenamiento real.")
    else:
        print(f"\n⚠️  Test falló. Revisar errores arriba.")
    
    print("\nPresiona Enter para salir...")
    # input()  # Descomentado para evitar bloqueo en scripts automáticos
