"""
Script para comparar diferentes configuraciones de preprocesamiento
Prueba diferentes tamaños de imagen y métodos de normalización
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import time
import os
import pandas as pd

from config import IMAGES_DIR, MASKS_DIR, RANDOM_SEED, BATCH_SIZE
from model.unet_mamba_variants import UNetMamba
from utils.metrics import dice_score
from preprocessing.dataset_v2 import HepaticVesselDatasetV2


def test_configuration(image_size, normalize_method, use_augmentation, epochs=3):
    """
    Prueba una configuración específica
    """
    print(f"\n🧪 Probando configuración:")
    print(f"   - Tamaño: {image_size}x{image_size}")
    print(f"   - Normalización: {normalize_method}")
    print(f"   - Augmentation: {use_augmentation}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear datasets
    train_dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=image_size,
        normalize_method=normalize_method,
        is_training=use_augmentation,
        include_empty=False
    )
    
    val_dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=image_size,
        normalize_method=normalize_method,
        is_training=False,
        include_empty=False
    )
    
    # Usar subset pequeño para pruebas rápidas
    train_indices = list(range(min(200, len(train_dataset))))
    val_indices = list(range(min(50, len(val_dataset))))
    
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    # Modelo
    model = UNetMamba(mode="simple", strategy="integrate").to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Entrenamiento rápido
    start_time = time.time()
    best_dice = 0.0
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss, train_dice = 0, 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            
            output = model(img)
            loss = criterion(output, mask)
            dice = dice_score(output, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice.item()
        
        # Validación
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
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        best_dice = max(best_dice, val_dice)
        
        print(f"   Epoch {epoch+1}/{epochs} - Val Dice: {val_dice:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Calcular tiempo por epoch
    time_per_epoch = training_time / epochs
    
    # Limpiar GPU
    del model, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    
    return {
        'image_size': image_size,
        'normalize_method': normalize_method,
        'use_augmentation': use_augmentation,
        'best_dice': best_dice,
        'training_time': training_time,
        'time_per_epoch': time_per_epoch
    }


def main():
    print("🔬 COMPARACIÓN DE CONFIGURACIONES DE PREPROCESAMIENTO")
    print("=" * 60)
    
    # Configuraciones a probar
    configurations = [
        # Tamaño 224
        (224, "minmax", False),
        (224, "minmax", True),
        (224, "zscore", False),
        (224, "clahe", False),
        
        # Tamaño 256
        (256, "minmax", False),
        (256, "minmax", True),
        (256, "zscore", False),
        (256, "clahe", False),
        
        # Tamaño 512 (si hay memoria suficiente)
        # (512, "minmax", False),
        # (512, "minmax", True),
    ]
    
    results = []
    
    for config in configurations:
        try:
            result = test_configuration(*config, epochs=3)
            results.append(result)
            
            print(f"✅ Completado - Dice: {result['best_dice']:.4f}, Tiempo: {result['time_per_epoch']:.1f}s/epoch")
            
        except Exception as e:
            print(f"❌ Error con configuración {config}: {e}")
            continue
    
    # Crear DataFrame con resultados
    df = pd.DataFrame(results)
    
    # Mostrar resultados
    print("\n📊 RESULTADOS FINALES:")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Guardar resultados
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/preprocessing_comparison.csv", index=False)
    
    # Análisis y recomendaciones
    print("\n🎯 ANÁLISIS Y RECOMENDACIONES:")
    print("=" * 50)
    
    # Mejor configuración por Dice score
    best_dice_config = df.loc[df['best_dice'].idxmax()]
    print(f"🥇 Mejor Dice Score: {best_dice_config['best_dice']:.4f}")
    print(f"   Configuración: {best_dice_config['image_size']}px, {best_dice_config['normalize_method']}, Aug: {best_dice_config['use_augmentation']}")
    
    # Mejor configuración por velocidad
    fastest_config = df.loc[df['time_per_epoch'].idxmin()]
    print(f"⚡ Más rápido: {fastest_config['time_per_epoch']:.1f}s/epoch")
    print(f"   Configuración: {fastest_config['image_size']}px, {fastest_config['normalize_method']}, Aug: {fastest_config['use_augmentation']}")
    
    # Comparar tamaños de imagen
    print(f"\n📏 Comparación por tamaño:")
    for size in df['image_size'].unique():
        subset = df[df['image_size'] == size]
        avg_dice = subset['best_dice'].mean()
        avg_time = subset['time_per_epoch'].mean()
        print(f"   {size}px: Dice promedio = {avg_dice:.4f}, Tiempo promedio = {avg_time:.1f}s")
    
    # Comparar métodos de normalización
    print(f"\n🔧 Comparación por normalización:")
    for method in df['normalize_method'].unique():
        subset = df[df['normalize_method'] == method]
        avg_dice = subset['best_dice'].mean()
        print(f"   {method}: Dice promedio = {avg_dice:.4f}")
    
    # Impacto del data augmentation
    print(f"\n🎯 Impacto del Data Augmentation:")
    with_aug = df[df['use_augmentation'] == True]['best_dice'].mean()
    without_aug = df[df['use_augmentation'] == False]['best_dice'].mean()
    print(f"   Con augmentation: {with_aug:.4f}")
    print(f"   Sin augmentation: {without_aug:.4f}")
    print(f"   Mejora: {with_aug - without_aug:.4f}")
    
    print(f"\n💾 Resultados guardados en: experiments/preprocessing_comparison.csv")


if __name__ == "__main__":
    main()
