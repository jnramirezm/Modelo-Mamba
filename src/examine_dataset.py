"""
Script para examinar las dimensiones del dataset real de hepatic vessels
"""

import nibabel as nib
import numpy as np
import os

def examine_dataset_sample():
    """Examina una muestra del dataset real"""
    
    # Rutas a los archivos
    img_path = "data/decathlon/imagesTr/hepaticvessel_001.nii.gz"
    mask_path = "data/decathlon/labelsTr/hepaticvessel_001.nii.gz"
    
    print("EXAMINANDO DATASET REAL")
    print("=" * 30)
    
    if not os.path.exists(img_path):
        print(f"❌ No se encuentra: {img_path}")
        return None
    
    # Cargar imagen
    img_nii = nib.load(img_path)
    img_data = img_nii.get_fdata()
    
    # Cargar máscara
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    
    print(f"📁 Archivo: hepaticvessel_001.nii.gz")
    print(f"📐 Dimensiones imagen: {img_data.shape}")
    print(f"📐 Dimensiones máscara: {mask_data.shape}")
    print(f"🔢 Tipo de datos imagen: {img_data.dtype}")
    print(f"🔢 Tipo de datos máscara: {mask_data.dtype}")
    print(f"📊 Rango imagen: [{img_data.min():.3f}, {img_data.max():.3f}]")
    print(f"📊 Valores únicos en máscara: {np.unique(mask_data)}")
    print(f"💾 Tamaño en memoria imagen: {img_data.nbytes / 1024 / 1024:.1f} MB")
    print(f"💾 Tamaño en memoria máscara: {mask_data.nbytes / 1024 / 1024:.1f} MB")
    
    # Estadísticas de la máscara
    vessel_pixels = np.sum(mask_data == 1)  # Vessel label = 1
    tumor_pixels = np.sum(mask_data == 2)   # Tumor label = 2
    background_pixels = np.sum(mask_data == 0)
    
    total_pixels = mask_data.size
    vessel_percentage = vessel_pixels / total_pixels * 100
    tumor_percentage = tumor_pixels / total_pixels * 100
    
    print(f"\n📈 ESTADÍSTICAS DE SEGMENTACIÓN:")
    print(f"  - Background: {background_pixels:,} pixels ({100 - vessel_percentage - tumor_percentage:.2f}%)")
    print(f"  - Vessels: {vessel_pixels:,} pixels ({vessel_percentage:.2f}%)")
    print(f"  - Tumors: {tumor_pixels:,} pixels ({tumor_percentage:.2f}%)")
    
    # Información espacial
    print(f"\n🌍 INFORMACIÓN ESPACIAL:")
    print(f"  - Affine matrix:\n{img_nii.affine}")
    print(f"  - Voxel size: {img_nii.header.get_zooms()}")
    
    return {
        'shape': img_data.shape,
        'dtype': img_data.dtype,
        'vessel_percentage': vessel_percentage,
        'tumor_percentage': tumor_percentage,
        'voxel_size': img_nii.header.get_zooms()
    }

if __name__ == "__main__":
    info = examine_dataset_sample()
    
    if info:
        print(f"\n💡 RECOMENDACIONES PARA TESTING:")
        H, W, D = info['shape']
        
        # Sugerir resolución de testing
        test_size = min(64, min(H, W) // 4)
        print(f"  - Resolución original: {H}x{W}x{D}")
        print(f"  - Resolución sugerida para testing: {test_size}x{test_size}")
        print(f"  - Esto reduce la imagen ~{(H*W) // (test_size*test_size):.0f}x")
        print(f"  - Usar solo algunos slices centrales para acelerar")
        print(f"  - Batch size recomendado: 1-2")
        print(f"  - Épocas para testing: 2-3")
