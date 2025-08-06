"""
Script de prueba para verificar las funcionalidades de preprocesamiento
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# Añadir el directorio src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Verifica que todas las dependencias se puedan importar correctamente"""
    print("Verificando imports...")
    
    try:
        import numpy as np
        print("✓ NumPy:", np.__version__)
        
        import torch
        print("✓ PyTorch:", torch.__version__)
        print("  - CUDA disponible:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  - CUDA version:", torch.version.cuda)
            print("  - GPUs disponibles:", torch.cuda.device_count())
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib")
        
        import cv2
        print("✓ OpenCV:", cv2.__version__)
        
        import scipy
        print("✓ SciPy:", scipy.__version__)
        
        import nibabel as nib
        print("✓ NiBabel:", nib.__version__)
        
        from tqdm import tqdm
        print("✓ tqdm")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error importando: {e}")
        return False


def test_basic_transforms():
    """Prueba las transformaciones básicas sin usar datos reales"""
    print("\nProbando transformaciones básicas...")
    
    # Crear imagen sintética
    img = np.random.rand(256, 256).astype(np.float32)
    mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    
    print(f"Imagen sintética creada: {img.shape}, rango: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Máscara sintética creada: {mask.shape}, píxeles positivos: {mask.sum()}")
    
    # Transformaciones básicas
    img_norm = (img - img.mean()) / (img.std() + 1e-8)
    img_minmax = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    print(f"Normalización Z-score: rango: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
    print(f"Normalización MinMax: rango: [{img_minmax.min():.3f}, {img_minmax.max():.3f}]")
    
    return True


def test_augmentations():
    """Prueba transformaciones de data augmentation"""
    print("\nProbando augmentaciones...")
    
    try:
        import cv2
        from scipy import ndimage
        
        # Crear imagen de prueba
        img = np.random.rand(256, 256).astype(np.float32)
        mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
        
        # Rotación
        angle = 15
        center = (img.shape[1]//2, img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        mask_rot = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        
        print(f"✓ Rotación aplicada: {angle}°")
        
        # Flip horizontal
        img_flip = np.fliplr(img)
        mask_flip = np.fliplr(mask)
        
        print("✓ Flip horizontal aplicado")
        
        # Deformación elástica simple
        sigma = 10
        alpha = 100
        dx = ndimage.gaussian_filter(np.random.rand(*img.shape) * 2 - 1, sigma) * alpha
        dy = ndimage.gaussian_filter(np.random.rand(*img.shape) * 2 - 1, sigma) * alpha
        
        print("✓ Campos de deformación elástica generados")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en augmentaciones: {e}")
        return False


def test_visualization():
    """Prueba la funcionalidad de visualización"""
    print("\nProbando visualización...")
    
    try:
        # Crear datos sintéticos para visualizar
        img1 = np.random.rand(256, 256)
        img2 = np.random.rand(256, 256)
        mask = np.random.randint(0, 2, (256, 256))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(img1, cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        axes[1].imshow(img2, cmap='gray')
        axes[1].set_title('Imagen Transformada')
        axes[1].axis('off')
        
        axes[2].imshow(img1, cmap='gray')
        axes[2].imshow(mask, cmap='Reds', alpha=0.3)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()  # Cerrar para evitar mostrar en entorno sin display
        
        print("✓ Visualización guardada en: test_visualization.png")
        return True
        
    except Exception as e:
        print(f"✗ Error en visualización: {e}")
        return False


def test_torch_functionality():
    """Prueba funcionalidades básicas de PyTorch"""
    print("\nProbando PyTorch...")
    
    try:
        # Crear tensores
        img_tensor = torch.randn(1, 1, 256, 256)
        mask_tensor = torch.randint(0, 2, (1, 1, 256, 256)).float()
        
        print(f"✓ Tensor imagen: {img_tensor.shape}")
        print(f"✓ Tensor máscara: {mask_tensor.shape}")
        
        # Operaciones básicas
        normalized = torch.nn.functional.normalize(img_tensor, dim=0)
        interpolated = torch.nn.functional.interpolate(img_tensor, size=(224, 224), mode='bilinear')
        
        print(f"✓ Interpolación: {img_tensor.shape} -> {interpolated.shape}")
        
        # Verificar CUDA si está disponible
        if torch.cuda.is_available():
            img_cuda = img_tensor.cuda()
            print(f"✓ Tensor en GPU: {img_cuda.device}")
            img_cpu = img_cuda.cpu()
            print("✓ Transferencia GPU -> CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en PyTorch: {e}")
        return False


def test_file_system():
    """Verifica el sistema de archivos y estructura del proyecto"""
    print("\nVerificando estructura del proyecto...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print(f"Directorio actual: {current_dir}")
    print(f"Raíz del proyecto: {project_root}")
    
    # Verificar directorios esperados
    expected_dirs = ['src', 'data']
    for dir_name in expected_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}/ encontrado")
        else:
            print(f"? {dir_name}/ no encontrado")
    
    # Verificar archivos de configuración
    config_files = ['requirements.txt', 'README.md']
    for file_name in config_files:
        file_path = os.path.join(project_root, file_name)
        if os.path.exists(file_path):
            print(f"✓ {file_name} encontrado")
        else:
            print(f"? {file_name} no encontrado")
    
    return True


def main():
    """Función principal de pruebas"""
    print("PRUEBAS DE CONFIGURACIÓN DEL ENTORNO")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Transformaciones básicas", test_basic_transforms),
        ("Augmentaciones", test_augmentations),
        ("Visualización", test_visualization),
        ("PyTorch", test_torch_functionality),
        ("Sistema de archivos", test_file_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Error inesperado: {e}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASÓ" if success else "✗ FALLÓ"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("\n🎉 ¡Todos los componentes funcionan correctamente!")
        print("El entorno está listo para entrenar modelos de segmentación.")
    else:
        print(f"\n⚠️  {len(results) - passed} pruebas fallaron.")
        print("Revisa los errores arriba para solucionar los problemas.")


if __name__ == "__main__":
    main()
