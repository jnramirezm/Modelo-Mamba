"""
Script para visualizar transformaciones de data augmentation
Muestra ejemplos de las transformaciones aplicadas a las imágenes
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# Añadir el directorio src al path para imports
sys.path.append(os.path.dirname(__file__))

# Configuración por defecto (puede ser sobrescrita desde config.py si existe)
DEFAULT_CONFIG = {
    'IMAGES_DIR': '../data/images',
    'MASKS_DIR': '../data/masks', 
    'IMAGE_SIZE': 256,
    'AUGMENTATION_PARAMS': {
        'rotation_range': (-15, 15),
        'horizontal_flip': True,
        'vertical_flip': False,
        'elastic_deformation': True,
        'elastic_alpha': 100,
        'elastic_sigma': 10,
        'gaussian_noise_std': 0.01,
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2)
    }
}

# Intentar importar configuración real, usar por defecto si no existe
try:
    from config import IMAGES_DIR, MASKS_DIR, IMAGE_SIZE, AUGMENTATION_PARAMS
    print("✓ Configuración importada desde config.py")
except ImportError:
    print("Archivo config.py no encontrado, usando configuración por defecto")
    IMAGES_DIR = DEFAULT_CONFIG['IMAGES_DIR']
    MASKS_DIR = DEFAULT_CONFIG['MASKS_DIR']
    IMAGE_SIZE = DEFAULT_CONFIG['IMAGE_SIZE']
    AUGMENTATION_PARAMS = DEFAULT_CONFIG['AUGMENTATION_PARAMS']

# Verificar que AUGMENTATION_PARAMS tenga la estructura correcta
if not isinstance(AUGMENTATION_PARAMS.get('rotation_range'), (tuple, list)):
    AUGMENTATION_PARAMS['rotation_range'] = (-15, 15)
if not isinstance(AUGMENTATION_PARAMS.get('brightness_range'), (tuple, list)):
    AUGMENTATION_PARAMS['brightness_range'] = (0.8, 1.2)
if not isinstance(AUGMENTATION_PARAMS.get('contrast_range'), (tuple, list)):
    AUGMENTATION_PARAMS['contrast_range'] = (0.8, 1.2)


def create_sample_data():
    """Crea datos sintéticos para demostración si no hay datos reales"""
    print("Creando datos sintéticos para demostración...")
    
    # Crear imagen sintética que simula una imagen médica
    img = np.random.randn(IMAGE_SIZE, IMAGE_SIZE) * 0.3 + 0.5
    img = np.clip(img, 0, 1).astype(np.float32)
    
    # Crear máscara sintética que simula vasos
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    
    # Añadir algunas estructuras tipo "vasos"
    center = IMAGE_SIZE // 2
    for i in range(5):
        # Líneas verticales
        x = center + np.random.randint(-50, 50)
        mask[20:IMAGE_SIZE-20, x-2:x+3] = 1
        
        # Líneas horizontales
        y = center + np.random.randint(-50, 50)
        mask[y-2:y+3, 20:IMAGE_SIZE-20] = 1
    
    # Añadir algo de ruido a la máscara
    noise_mask = np.random.rand(IMAGE_SIZE, IMAGE_SIZE) > 0.98
    mask[noise_mask] = 1
    
    return img, mask


def apply_basic_augmentations(img, mask):
    """Aplica augmentaciones básicas usando OpenCV y SciPy"""
    import cv2
    from scipy import ndimage
    
    augmented_samples = []
    
    # 1. Rotación
    angle = np.random.uniform(AUGMENTATION_PARAMS['rotation_range'][0], 
                             AUGMENTATION_PARAMS['rotation_range'][1])
    center = (img.shape[1]//2, img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    mask_rot = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    augmented_samples.append((img_rot, mask_rot, f"Rotación ({angle:.1f}°)"))
    
    # 2. Flip horizontal
    if AUGMENTATION_PARAMS['horizontal_flip'] and np.random.rand() > 0.5:
        img_flip = np.fliplr(img)
        mask_flip = np.fliplr(mask)
        augmented_samples.append((img_flip, mask_flip, "Flip Horizontal"))
    
    # 3. Ruido Gaussiano
    if 'gaussian_noise_std' in AUGMENTATION_PARAMS:
        noise = np.random.normal(0, AUGMENTATION_PARAMS['gaussian_noise_std'], img.shape)
        img_noisy = np.clip(img + noise, 0, 1)
        augmented_samples.append((img_noisy, mask, "Ruido Gaussiano"))
    
    # 4. Ajuste de brillo
    if 'brightness_range' in AUGMENTATION_PARAMS:
        brightness_factor = np.random.uniform(*AUGMENTATION_PARAMS['brightness_range'])
        img_bright = np.clip(img * brightness_factor, 0, 1)
        augmented_samples.append((img_bright, mask, f"Brillo ({brightness_factor:.2f}x)"))
    
    return augmented_samples


def visualize_augmentations():
    """Visualiza ejemplos de data augmentation"""
    print("Generando visualización de augmentaciones...")
    
    # Crear o cargar datos
    img, mask = create_sample_data()
    
    # Aplicar augmentaciones
    augmented_samples = apply_basic_augmentations(img, mask)
    
    # Crear visualización
    n_samples = min(4, len(augmented_samples) + 1)  # +1 para la imagen original
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    fig.suptitle("Ejemplos de Data Augmentation para Segmentación de Vasos Hepáticos", fontsize=16)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Mostrar imagen original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Imagen Original")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='Reds', alpha=0.8)
    axes[0, 1].set_title("Máscara Original")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img, cmap='gray')
    axes[0, 2].imshow(mask, cmap='Reds', alpha=0.3)
    axes[0, 2].set_title("Overlay Original")
    axes[0, 2].axis('off')
    
    # Mostrar augmentaciones
    for i, (aug_img, aug_mask, aug_name) in enumerate(augmented_samples[:n_samples-1], 1):
        axes[i, 0].imshow(aug_img, cmap='gray')
        axes[i, 0].set_title(f"Imagen - {aug_name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(aug_mask, cmap='Reds', alpha=0.8)
        axes[i, 1].set_title(f"Máscara - {aug_name}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(aug_img, cmap='gray')
        axes[i, 2].imshow(aug_mask, cmap='Reds', alpha=0.3)
        axes[i, 2].set_title(f"Overlay - {aug_name}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Ejemplos guardados en: augmentation_examples.png")


def compare_normalization_methods():
    """Compara diferentes métodos de normalización"""
    print("Comparando métodos de normalización...")
    
    # Crear imagen de ejemplo
    img = create_sample_data()[0]
    
    # Aplicar diferentes normalizaciones
    methods = {}
    
    # Original
    methods["Original"] = img
    
    # MinMax
    methods["MinMax"] = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Z-Score
    methods["Z-Score"] = (img - img.mean()) / (img.std() + 1e-8)
    
    # CLAHE (si OpenCV está disponible)
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_uint8 = (img * 255).astype(np.uint8)
        img_clahe = clahe.apply(img_uint8) / 255.0
        methods["CLAHE"] = img_clahe
    except ImportError:
        pass
    
    # Percentile
    p5, p95 = np.percentile(img, [5, 95])
    img_percentile = np.clip((img - p5) / (p95 - p5 + 1e-8), 0, 1)
    methods["Percentile (5-95)"] = img_percentile
    
    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods), 4))
    fig.suptitle("Comparación de Métodos de Normalización", fontsize=16)
    
    if len(methods) == 1:
        axes = [axes]
    
    for i, (method_name, normalized_img) in enumerate(methods.items()):
        axes[i].imshow(normalized_img, cmap='gray')
        title_text = f"{method_name}\nMin: {np.min(normalized_img):.3f}\nMax: {np.max(normalized_img):.3f}"
        axes[i].set_title(title_text)
        axes[i].axis('off')
        
        # Añadir estadísticas
        stats_text = f"Mean: {np.mean(normalized_img):.3f}\nStd: {np.std(normalized_img):.3f}"
        axes[i].text(0.02, 0.98, stats_text,
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig("normalization_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Comparación guardada en: normalization_comparison.png")


def analyze_synthetic_dataset():
    """Analiza datos sintéticos para demostración"""
    print("Analizando dataset sintético...")
    
    # Generar múltiples muestras sintéticas
    vessel_pixels = []
    images = []
    masks = []
    
    for i in range(50):
        img, mask = create_sample_data()
        images.append(img)
        masks.append(mask)
        vessel_pixels.append(np.sum(mask))
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Histograma
    plt.subplot(2, 2, 1)
    plt.hist(vessel_pixels, bins=20, alpha=0.7, color='blue')
    plt.title("Distribución de Píxeles de Vasos")
    plt.xlabel("Número de Píxeles")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(vessel_pixels)
    plt.title("Box Plot - Píxeles de Vasos")
    plt.ylabel("Número de Píxeles")
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Ejemplos de imágenes
    plt.subplot(2, 2, 3)
    example_img = images[0]
    plt.imshow(example_img, cmap='gray')
    plt.title("Ejemplo de Imagen Sintética")
    plt.axis('off')
    
    # Subplot 4: Ejemplo de máscara
    plt.subplot(2, 2, 4)
    example_mask = masks[0]
    plt.imshow(example_mask, cmap='Reds')
    plt.title("Ejemplo de Máscara Sintética")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Análisis guardado en: dataset_analysis.png")
    print(f"  - Promedio de píxeles de vasos: {np.mean(vessel_pixels):.1f}")
    print(f"  - Desviación estándar: {np.std(vessel_pixels):.1f}")
    print(f"  - Rango: {np.min(vessel_pixels):.0f} - {np.max(vessel_pixels):.0f}")


def main():
    print("VISUALIZACIÓN DE PREPROCESAMIENTO")
    print("=" * 50)
    print(f"Configuración:")
    print(f"  - Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Parámetros de augmentation: {len(AUGMENTATION_PARAMS)} configurados")
    
    try:
        print("\n1. Generando ejemplos de data augmentation...")
        visualize_augmentations()
        
        print("\n2. Comparando métodos de normalización...")
        compare_normalization_methods()
        
        print("\n3. Analizando dataset sintético...")
        analyze_synthetic_dataset()
        
        print("\n✅ ¡Visualización completada exitosamente!")
        print("\nArchivos generados:")
        print("   - augmentation_examples.png")
        print("   - normalization_comparison.png") 
        print("   - dataset_analysis.png")
        
        print("\nNota: Este script usa datos sintéticos para demostración.")
        print("Para usar con datos reales, asegúrate de que estén en las rutas correctas:")
        print(f"   - Imágenes: {IMAGES_DIR}")
        print(f"   - Máscaras: {MASKS_DIR}")
        
    except Exception as e:
        print(f"❌ Error durante la visualización: {e}")
        import traceback
        traceback.print_exc()
        print("\nPosibles soluciones:")
        print("   - Verificar que las dependencias estén instaladas")
        print("   - Asegurar que hay suficiente memoria disponible")
        print("   - Revisar los permisos de escritura en el directorio")


if __name__ == "__main__":
    main()
