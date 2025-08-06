"""
Script para visualizar transformaciones de data augmentation
Muestra ejemplos de las transformaciones aplicadas a las imágenes
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import IMAGES_DIR, MASKS_DIR, IMAGE_SIZE, AUGMENTATION_PARAMS
from preprocessing.dataset_v2 import HepaticVesselDatasetV2
from preprocessing.transforms import MedicalDataAugmentation


def visualize_augmentations():
    """Visualiza ejemplos de data augmentation"""
    
    # Crear dataset sin augmentation
    dataset_orig = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=IMAGE_SIZE,
        is_training=False,  # Sin augmentation
        include_empty=False
    )
    
    # Crear augmentator
    augmentator = MedicalDataAugmentation(AUGMENTATION_PARAMS)
    
    # Seleccionar algunas muestras
    sample_indices = [0, 10, 20, 30]  # Primeras muestras disponibles
    
    fig, axes = plt.subplots(len(sample_indices), 6, figsize=(18, 3*len(sample_indices)))
    fig.suptitle("🎯 Ejemplos de Data Augmentation para Segmentación de Vasos Hepáticos", fontsize=16)
    
    for row, idx in enumerate(sample_indices):
        if idx >= len(dataset_orig):
            continue
            
        # Obtener imagen y máscara originales
        orig_img, orig_mask = dataset_orig[idx]
        
        # Convertir a numpy para augmentation
        img_np = orig_img.squeeze().cpu().numpy()
        mask_np = orig_mask.squeeze().cpu().numpy()
        
        # Aplicar augmentation
        aug_img, aug_mask = augmentator(img_np, mask_np)
        
        # Convertir de vuelta para visualización
        aug_img_display = aug_img.squeeze().cpu().numpy()
        aug_mask_display = aug_mask.squeeze().cpu().numpy()
        
        # Obtener información del slice
        slice_info = dataset_orig.get_slice_info(idx)
        vessel_pixels = slice_info['vessel_pixels']
        filename = slice_info['filename']
        
        # Configurar títulos
        if row == 0:
            titles = [
                "Imagen Original", 
                "Máscara Original", 
                "Imagen + Augmentation", 
                "Máscara + Augmentation",
                "Overlay Original",
                "Overlay Augmentado"
            ]
        else:
            titles = ["", "", "", "", "", ""]
        
        # Subplot 1: Imagen original
        axes[row, 0].imshow(img_np, cmap='gray')
        axes[row, 0].set_title(titles[0])
        axes[row, 0].axis('off')
        
        # Subplot 2: Máscara original
        axes[row, 1].imshow(mask_np, cmap='red', alpha=0.8)
        axes[row, 1].set_title(titles[1])
        axes[row, 1].axis('off')
        
        # Subplot 3: Imagen augmentada
        axes[row, 2].imshow(aug_img_display, cmap='gray')
        axes[row, 2].set_title(titles[2])
        axes[row, 2].axis('off')
        
        # Subplot 4: Máscara augmentada
        axes[row, 3].imshow(aug_mask_display, cmap='red', alpha=0.8)
        axes[row, 3].set_title(titles[3])
        axes[row, 3].axis('off')
        
        # Subplot 5: Overlay original
        axes[row, 4].imshow(img_np, cmap='gray')
        axes[row, 4].imshow(mask_np, cmap='red', alpha=0.3)
        axes[row, 4].set_title(titles[4])
        axes[row, 4].axis('off')
        
        # Subplot 6: Overlay augmentado
        axes[row, 5].imshow(aug_img_display, cmap='gray')
        axes[row, 5].imshow(aug_mask_display, cmap='red', alpha=0.3)
        axes[row, 5].set_title(titles[5])
        axes[row, 5].axis('off')
        
        # Añadir información del slice
        axes[row, 0].text(5, 15, f"{filename}\nVasos: {vessel_pixels}px", 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                          fontsize=8, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Ejemplos guardados en: augmentation_examples.png")


def compare_normalization_methods():
    """Compara diferentes métodos de normalización"""
    
    from preprocessing.transforms import NormalizationMethods
    
    # Crear dataset básico
    dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=IMAGE_SIZE,
        normalize_method="minmax",  # Método base
        is_training=False
    )
    
    if len(dataset) == 0:
        print("❌ No se encontraron datos en el dataset")
        return
    
    # Obtener una imagen de ejemplo
    slice_info = dataset.get_slice_info(0)
    img_slice, _ = dataset._load_slice_data(slice_info)
    
    # Aplicar diferentes normalizaciones
    methods = {
        "Original": img_slice,
        "MinMax": NormalizationMethods.minmax_normalize(img_slice),
        "Z-Score": NormalizationMethods.zscore_normalize(img_slice),
        "CLAHE": NormalizationMethods.clahe_normalize(img_slice),
        "Percentile": NormalizationMethods.percentile_normalize(img_slice)
    }
    
    fig, axes = plt.subplots(1, len(methods), figsize=(20, 4))
    fig.suptitle("🔧 Comparación de Métodos de Normalización", fontsize=16)
    
    for i, (method_name, normalized_img) in enumerate(methods.items()):
        axes[i].imshow(normalized_img, cmap='gray')
        axes[i].set_title(f"{method_name}\nMin: {np.min(normalized_img):.3f}\nMax: {np.max(normalized_img):.3f}")
        axes[i].axis('off')
        
        # Añadir estadísticas
        axes[i].text(0.02, 0.98, f"Mean: {np.mean(normalized_img):.3f}\nStd: {np.std(normalized_img):.3f}",
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig("normalization_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Comparación guardada en: normalization_comparison.png")\n\n\ndef analyze_dataset_distribution():\n    \"\"\"Analiza la distribución de píxeles de vasos en el dataset\"\"\"\n    \n    dataset = HepaticVesselDatasetV2(\n        IMAGES_DIR, MASKS_DIR,\n        image_size=IMAGE_SIZE,\n        is_training=False\n    )\n    \n    stats = dataset.get_dataset_stats()\n    \n    print(\"📊 ESTADÍSTICAS DEL DATASET:\")\n    print(\"=\" * 40)\n    for key, value in stats.items():\n        print(f\"{key}: {value}\")\n    \n    # Distribución de píxeles de vasos\n    vessel_pixels = [dataset.get_slice_info(i)['vessel_pixels'] for i in range(len(dataset))]\n    \n    plt.figure(figsize=(12, 8))\n    \n    # Subplot 1: Histograma\n    plt.subplot(2, 2, 1)\n    plt.hist(vessel_pixels, bins=50, alpha=0.7, color='blue')\n    plt.title(\"Distribución de Píxeles de Vasos\")\n    plt.xlabel(\"Número de Píxeles\")\n    plt.ylabel(\"Frecuencia\")\n    plt.grid(True, alpha=0.3)\n    \n    # Subplot 2: Box plot\n    plt.subplot(2, 2, 2)\n    plt.boxplot(vessel_pixels)\n    plt.title(\"Box Plot - Píxeles de Vasos\")\n    plt.ylabel(\"Número de Píxeles\")\n    plt.grid(True, alpha=0.3)\n    \n    # Subplot 3: Distribución acumulativa\n    plt.subplot(2, 2, 3)\n    sorted_pixels = np.sort(vessel_pixels)\n    cumulative = np.arange(1, len(sorted_pixels) + 1) / len(sorted_pixels)\n    plt.plot(sorted_pixels, cumulative)\n    plt.title(\"Distribución Acumulativa\")\n    plt.xlabel(\"Número de Píxeles\")\n    plt.ylabel(\"Percentil\")\n    plt.grid(True, alpha=0.3)\n    \n    # Subplot 4: Estadísticas por archivo\n    plt.subplot(2, 2, 4)\n    files = [dataset.get_slice_info(i)['filename'] for i in range(len(dataset))]\n    unique_files = list(set(files))\n    file_vessel_counts = []\n    \n    for file in unique_files[:10]:  # Primeros 10 archivos\n        file_pixels = [vessel_pixels[i] for i, f in enumerate(files) if f == file]\n        file_vessel_counts.append(np.mean(file_pixels))\n    \n    plt.bar(range(len(file_vessel_counts)), file_vessel_counts)\n    plt.title(\"Promedio de Píxeles por Archivo (Top 10)\")\n    plt.xlabel(\"Archivo\")\n    plt.ylabel(\"Píxeles Promedio\")\n    plt.xticks(range(len(file_vessel_counts)), [f\"F{i+1}\" for i in range(len(file_vessel_counts))])\n    plt.grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig(\"dataset_analysis.png\", dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(\"💾 Análisis guardado en: dataset_analysis.png\")\n\n\ndef main():\n    print(\"🎨 VISUALIZACIÓN DE PREPROCESAMIENTO\")\n    print(\"=\" * 50)\n    \n    try:\n        print(\"\\n1️⃣ Generando ejemplos de data augmentation...\")\n        visualize_augmentations()\n        \n        print(\"\\n2️⃣ Comparando métodos de normalización...\")\n        compare_normalization_methods()\n        \n        print(\"\\n3️⃣ Analizando distribución del dataset...\")\n        analyze_dataset_distribution()\n        \n        print(\"\\n✅ Visualización completada!\")\n        print(\"\\n📁 Archivos generados:\")\n        print(\"   - augmentation_examples.png\")\n        print(\"   - normalization_comparison.png\")\n        print(\"   - dataset_analysis.png\")\n        \n    except Exception as e:\n        print(f\"❌ Error durante la visualización: {e}\")\n        print(\"\\n💡 Asegúrate de que:\")\n        print(\"   - Los datos están en las rutas correctas\")\n        print(\"   - Las dependencias están instaladas\")\n        print(\"   - Hay suficiente memoria disponible\")\n\n\nif __name__ == \"__main__\":\n    main()
