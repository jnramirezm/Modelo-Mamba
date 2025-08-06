"""
Script para generar un dataset sintético pequeño para testing
Crea imágenes y máscaras NIfTI simulando datos médicos reales
"""

import numpy as np
import nibabel as nib
import os
from pathlib import Path
import cv2
from scipy import ndimage


def create_synthetic_medical_image(size=(128, 128, 32), noise_level=0.1):
    """Crea una imagen médica sintética con características realistas"""
    
    # Base: imagen con intensidades típicas de CT/MRI
    img = np.random.normal(0.3, 0.1, size).astype(np.float32)
    
    # Añadir estructuras anatómicas simuladas
    center_z = size[2] // 2
    center_y = size[1] // 2
    center_x = size[0] // 2
    
    # Estructura principal (simula hígado)
    for z in range(size[2]):
        z_factor = 1 - abs(z - center_z) / center_z * 0.3
        
        # Crear estructura ovalada
        y, x = np.ogrid[:size[1], :size[0]]
        mask_organ = ((x - center_x)**2 / (40 * z_factor)**2 + 
                     (y - center_y)**2 / (35 * z_factor)**2) <= 1
        
        img[:, :, z][mask_organ] += np.random.normal(0.4, 0.05, np.sum(mask_organ))
    
    # Añadir ruido gaussiano
    img += np.random.normal(0, noise_level, size)
    
    # Normalizar a rango típico de imágenes médicas
    img = np.clip(img, 0, 1)
    
    return img


def create_vessel_mask(size=(128, 128, 32), vessel_density=0.02):
    """Crea una máscara de vasos sintética"""
    
    mask = np.zeros(size, dtype=np.float32)
    center_z = size[2] // 2
    
    # Crear vasos principales
    num_main_vessels = np.random.randint(3, 6)
    
    for vessel_id in range(num_main_vessels):
        # Definir trayectoria del vaso
        start_z = np.random.randint(5, size[2] - 5)
        end_z = np.random.randint(start_z + 5, min(start_z + 15, size[2] - 1))
        
        start_x = np.random.randint(20, size[0] - 20)
        start_y = np.random.randint(20, size[1] - 20)
        
        end_x = start_x + np.random.randint(-15, 15)
        end_y = start_y + np.random.randint(-15, 15)
        
        # Crear línea 3D del vaso
        num_points = end_z - start_z + 1
        z_coords = np.linspace(start_z, end_z, num_points)
        x_coords = np.linspace(start_x, end_x, num_points)
        y_coords = np.linspace(start_y, end_y, num_points)
        
        # Dibujar el vaso con grosor variable
        for i, (z, x, y) in enumerate(zip(z_coords, x_coords, y_coords)):
            z, x, y = int(z), int(x), int(y)
            
            # Grosor del vaso (más grueso en el centro)
            thickness = np.random.randint(1, 4)
            
            # Dibujar círculo en el slice actual
            if 0 <= z < size[2]:
                for dx in range(-thickness, thickness + 1):
                    for dy in range(-thickness, thickness + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < size[0] and 0 <= ny < size[1] and
                            dx*dx + dy*dy <= thickness*thickness):
                            mask[nx, ny, z] = 1.0
        
        # Crear ramificaciones
        if np.random.random() > 0.5:
            branch_start = int(len(z_coords) * 0.6)
            if branch_start < len(z_coords) - 3:
                branch_length = min(8, len(z_coords) - branch_start)
                
                for i in range(branch_length):
                    bz = int(z_coords[branch_start + i])
                    bx = int(x_coords[branch_start + i] + np.random.randint(-5, 5))
                    by = int(y_coords[branch_start + i] + np.random.randint(-5, 5))
                    
                    if (0 <= bx < size[0] and 0 <= by < size[1] and 0 <= bz < size[2]):
                        mask[bx, by, bz] = 1.0
    
    # Suavizar ligeramente la máscara
    mask = ndimage.gaussian_filter(mask, sigma=0.5)
    mask = (mask > 0.1).astype(np.float32)
    
    return mask


def save_nifti_file(data, filepath, affine=None):
    """Guarda datos como archivo NIfTI"""
    if affine is None:
        affine = np.eye(4)  # Matriz identidad por defecto
    
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filepath)


def create_test_dataset(output_dir="data/test_dataset", num_samples=5):
    """Crea un dataset completo de prueba"""
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "imagesTr"
    masks_dir = output_dir / "labelsTr"
    
    # Crear directorios
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creando dataset de prueba con {num_samples} muestras...")
    print(f"Directorio: {output_dir.absolute()}")
    
    # Definir matriz de transformación espacial estándar
    affine = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    sample_info = []
    
    for i in range(num_samples):
        print(f"Generando muestra {i+1}/{num_samples}...")
        
        # Crear imagen y máscara
        img_data = create_synthetic_medical_image(size=(128, 128, 32))
        mask_data = create_vessel_mask(size=(128, 128, 32))
        
        # Nombres de archivos
        img_filename = f"hepatic_vessels_{i+1:03d}.nii.gz"
        mask_filename = f"hepatic_vessels_{i+1:03d}.nii.gz"
        
        # Rutas completas
        img_path = images_dir / img_filename
        mask_path = masks_dir / mask_filename
        
        # Guardar archivos
        save_nifti_file(img_data, str(img_path), affine)
        save_nifti_file(mask_data, str(mask_path), affine)
        
        # Estadísticas de la muestra
        vessel_pixels = np.sum(mask_data > 0)
        vessel_percentage = vessel_pixels / mask_data.size * 100
        
        sample_info.append({
            'filename': img_filename,
            'shape': img_data.shape,
            'vessel_pixels': vessel_pixels,
            'vessel_percentage': vessel_percentage,
            'img_min': img_data.min(),
            'img_max': img_data.max(),
            'img_mean': img_data.mean()
        })
        
        print(f"  - Forma: {img_data.shape}")
        print(f"  - Píxeles de vasos: {vessel_pixels} ({vessel_percentage:.2f}%)")
        print(f"  - Rango imagen: [{img_data.min():.3f}, {img_data.max():.3f}]")
    
    # Crear archivo de metadatos
    metadata_path = output_dir / "dataset_info.txt"
    with open(metadata_path, 'w') as f:
        f.write("DATASET SINTÉTICO DE PRUEBA - VASOS HEPÁTICOS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Número de muestras: {num_samples}\n")
        f.write(f"Resolución: 128x128x32\n")
        f.write(f"Formato: NIfTI (.nii.gz)\n\n")
        
        f.write("ESTADÍSTICAS POR MUESTRA:\n")
        f.write("-" * 30 + "\n")
        
        for info in sample_info:
            f.write(f"\n{info['filename']}:\n")
            f.write(f"  Forma: {info['shape']}\n")
            f.write(f"  Píxeles de vasos: {info['vessel_pixels']}\n")
            f.write(f"  Porcentaje vasos: {info['vessel_percentage']:.2f}%\n")
            f.write(f"  Imagen - Min: {info['img_min']:.3f}, Max: {info['img_max']:.3f}, Mean: {info['img_mean']:.3f}\n")
        
        # Estadísticas globales
        total_vessel_pixels = sum(info['vessel_pixels'] for info in sample_info)
        avg_vessel_percentage = np.mean([info['vessel_percentage'] for info in sample_info])
        
        f.write(f"\nESTADÍSTICAS GLOBALES:\n")
        f.write(f"Total píxeles de vasos: {total_vessel_pixels}\n")
        f.write(f"Promedio porcentaje vasos: {avg_vessel_percentage:.2f}%\n")
    
    print(f"\n✅ Dataset creado exitosamente!")
    print(f"📁 Imágenes: {images_dir}")
    print(f"📁 Máscaras: {masks_dir}")
    print(f"📄 Metadatos: {metadata_path}")
    print(f"\nESTADÍSTICAS:")
    print(f"- Total muestras: {num_samples}")
    print(f"- Resolución: 128x128x32") 
    print(f"- Promedio píxeles vasos: {np.mean([info['vessel_pixels'] for info in sample_info]):.0f}")
    print(f"- Promedio porcentaje vasos: {avg_vessel_percentage:.2f}%")
    
    return output_dir


if __name__ == "__main__":
    # Crear dataset de prueba
    dataset_path = create_test_dataset(num_samples=8)  # 8 muestras para train/val split
    
    print("\n" + "="*60)
    print("DATASET DE PRUEBA LISTO PARA USAR")
    print("="*60)
    print("\nPara usar este dataset, actualiza la configuración:")
    print(f"IMAGES_DIR = '{dataset_path}/imagesTr'")
    print(f"MASKS_DIR = '{dataset_path}/labelsTr'")
    print("IMAGE_SIZE = 64  # Resolución baja para testing")
    print("EPOCHS = 3       # Pocas épocas para testing")
