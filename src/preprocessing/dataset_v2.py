"""
Dataset mejorado con preprocesamiento avanzado y data augmentation
Optimizado para segmentación de vasos hepáticos
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from preprocessing.transforms import get_transforms, NormalizationMethods
from config import IMAGE_SIZE, NORMALIZE_METHOD, USE_DATA_AUGMENTATION, AUGMENTATION_PARAMS


class HepaticVesselDatasetV2(Dataset):
    """
    Dataset mejorado para segmentación de vasos hepáticos
    
    Características:
    - Múltiples métodos de normalización
    - Data augmentation profesional
    - Diferentes tamaños de imagen (224, 256, 512)
    - Cache inteligente para mejorar rendimiento
    - Filtrado de slices vacíos mejorado
    """
    
    def __init__(self, 
                 images_dir, 
                 masks_dir, 
                 image_size=256,
                 normalize_method="minmax",
                 is_training=True,
                 include_empty=False,
                 cache_data=False,
                 min_vessel_pixels=10,
                 augmentation_params=None,
                 vessel_label=1,
                 max_samples=None):
        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.normalize_method = normalize_method
        self.is_training = is_training
        self.include_empty = include_empty
        self.cache_data = cache_data
        self.min_vessel_pixels = min_vessel_pixels
        self.vessel_label = vessel_label  # Para dataset de Decathlon que tiene multiple labels
        self.max_samples = max_samples
        
        # Usar parámetros de augmentation específicos si se proporcionan
        self.augmentation_params = augmentation_params or AUGMENTATION_PARAMS
        
        # Configurar normalizador
        self.normalizer = self._get_normalizer(normalize_method)
        
        # Configurar transformaciones
        config = {
            'USE_DATA_AUGMENTATION': USE_DATA_AUGMENTATION,
            'AUGMENTATION_PARAMS': AUGMENTATION_PARAMS
        }
        self.transform = get_transforms(config, is_training)
        
        # Cache para datos
        self.data_cache = {} if cache_data else None
        
        # Preparar lista de slices
        self.slices = []
        self._prepare_slice_list()
        
        print(f"📊 Dataset inicializado:")
        print(f"   - Total slices: {len(self.slices)}")
        print(f"   - Tamaño imagen: {image_size}x{image_size}")
        print(f"   - Normalización: {normalize_method}")
        print(f"   - Entrenamiento: {is_training}")
        print(f"   - Augmentation: {USE_DATA_AUGMENTATION and is_training}")
        print(f"   - Cache activado: {cache_data}")
    
    def _get_normalizer(self, method):
        """Obtiene función de normalización"""
        normalizers = {
            "minmax": NormalizationMethods.minmax_normalize,
            "zscore": NormalizationMethods.zscore_normalize,
            "clahe": NormalizationMethods.clahe_normalize,
            "percentile": NormalizationMethods.percentile_normalize
        }
        
        if method not in normalizers:
            print(f"⚠️  Método de normalización '{method}' no encontrado. Usando 'minmax'")
            return normalizers["minmax"]
        
        return normalizers[method]
    
    def _prepare_slice_list(self):
        """Prepara lista de slices útiles"""
        print("🔍 Analizando archivos...")
        
        file_list = sorted(os.listdir(self.images_dir))
        
        # Limitar número de archivos si se especifica max_samples
        if self.max_samples is not None:
            file_list = file_list[:self.max_samples]
            print(f"🔄 Limitando a {self.max_samples} muestras para testing rápido")
        
        for filename in file_list:
            if not filename.endswith(".nii.gz"):
                continue
            
            img_path = os.path.join(self.images_dir, filename)
            msk_path = os.path.join(self.masks_dir, filename)
            
            # Verificar que existe la máscara
            if not os.path.exists(msk_path):
                print(f"⚠️  Máscara no encontrada: {filename}")
                continue
            
            try:
                # Cargar volúmenes
                img = nib.load(img_path).get_fdata()
                msk = nib.load(msk_path).get_fdata()
                
                # Verificar dimensiones
                if img.shape != msk.shape:
                    print(f"⚠️  Dimensiones no coinciden en {filename}: {img.shape} vs {msk.shape}")
                    continue
                
                # Analizar cada slice
                for i in range(img.shape[2]):
                    # Contar píxeles específicos de vasos (label 1 en Decathlon)
                    if hasattr(self, 'vessel_label') and self.vessel_label is not None:
                        vessel_pixels = np.sum(msk[:, :, i] == self.vessel_label)
                    else:
                        vessel_pixels = np.sum(msk[:, :, i] > 0)
                    
                    # Incluir slice si:
                    # 1. include_empty=True, o
                    # 2. Tiene suficientes píxeles de vasos
                    if self.include_empty or vessel_pixels >= self.min_vessel_pixels:
                        slice_info = {
                            'img_path': img_path,
                            'msk_path': msk_path,
                            'slice_idx': i,
                            'vessel_pixels': vessel_pixels,
                            'filename': filename
                        }
                        self.slices.append(slice_info)
                        
            except Exception as e:
                print(f"❌ Error procesando {filename}: {e}")
        
        print(f"✅ Análisis completado. {len(self.slices)} slices válidos encontrados.")
    
    def _load_slice_data(self, slice_info):
        """Carga datos de un slice específico"""
        cache_key = f"{slice_info['filename']}_{slice_info['slice_idx']}"
        
        # Verificar cache
        if self.data_cache is not None and cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Cargar desde disco
        img = nib.load(slice_info['img_path']).get_fdata()
        msk = nib.load(slice_info['msk_path']).get_fdata()
        
        img_slice = img[:, :, slice_info['slice_idx']]
        msk_slice = msk[:, :, slice_info['slice_idx']]
        
        # Guardar en cache si está activado
        if self.data_cache is not None:
            self.data_cache[cache_key] = (img_slice.copy(), msk_slice.copy())
        
        return img_slice, msk_slice
    
    def _preprocess_image(self, img_slice):
        """Preprocesa la imagen"""
        # Convertir a float32
        img_slice = img_slice.astype(np.float32)
        
        # Manejar valores extremos
        img_slice = np.clip(img_slice, np.percentile(img_slice, 0.5), 
                           np.percentile(img_slice, 99.5))
        
        # Normalizar
        img_slice = self.normalizer(img_slice)
        
        return img_slice
    
    def _preprocess_mask(self, msk_slice):
        """Preprocesa la máscara - Maneja múltiples labels del dataset Decathlon"""
        if hasattr(self, 'vessel_label') and self.vessel_label is not None:
            # Para dataset Decathlon: 0=background, 1=vessel, 2=tumor
            # Solo extraer los vasos (label 1)
            msk_slice = (msk_slice == self.vessel_label).astype(np.float32)
        else:
            # Binarizar máscara (comportamiento original)
            msk_slice = (msk_slice > 0).astype(np.float32)
        return msk_slice
    
    def _resize_data(self, img_tensor, msk_tensor):
        """Redimensiona imagen y máscara al tamaño objetivo"""
        # Redimensionar imagen con interpolación bilinear
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0), 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Redimensionar máscara con interpolación nearest
        msk_tensor = F.interpolate(
            msk_tensor.unsqueeze(0), 
            size=(self.image_size, self.image_size), 
            mode='nearest'
        ).squeeze(0)
        
        return img_tensor, msk_tensor
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        """Obtiene un elemento del dataset"""
        slice_info = self.slices[idx]
        
        # Cargar datos
        img_slice, msk_slice = self._load_slice_data(slice_info)
        
        # Preprocesar
        img_slice = self._preprocess_image(img_slice)
        msk_slice = self._preprocess_mask(msk_slice)
        
        # Convertir a tensores
        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).float()
        msk_tensor = torch.from_numpy(msk_slice).unsqueeze(0).float()
        
        # Aplicar transformaciones (augmentation si es entrenamiento)
        if self.transform:
            img_tensor, msk_tensor = self.transform(img_tensor, msk_tensor)
        
        # Redimensionar al tamaño objetivo
        img_tensor, msk_tensor = self._resize_data(img_tensor, msk_tensor)
        
        return img_tensor, msk_tensor
    
    def get_slice_info(self, idx):
        """Obtiene información del slice"""
        return self.slices[idx]
    
    def get_dataset_stats(self):
        """Obtiene estadísticas del dataset"""
        vessel_pixels = [s['vessel_pixels'] for s in self.slices]
        
        stats = {
            'total_slices': len(self.slices),
            'empty_slices': sum(1 for p in vessel_pixels if p == 0),
            'non_empty_slices': sum(1 for p in vessel_pixels if p > 0),
            'avg_vessel_pixels': np.mean(vessel_pixels),
            'min_vessel_pixels': np.min(vessel_pixels),
            'max_vessel_pixels': np.max(vessel_pixels),
            'unique_files': len(set(s['filename'] for s in self.slices))
        }
        
        return stats


# Función de compatibilidad con el dataset anterior
def HepaticVessel2DDataset(images_dir, masks_dir, transform=None, include_empty=False):
    """
    Función de compatibilidad con el dataset anterior
    Ahora usa el dataset mejorado con configuración básica
    """
    return HepaticVesselDatasetV2(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=IMAGE_SIZE,
        normalize_method=NORMALIZE_METHOD,
        is_training=True,  # Por defecto entrenamiento
        include_empty=include_empty,
        cache_data=False
    )
