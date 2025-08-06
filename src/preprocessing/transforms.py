"""
Transformaciones avanzadas para data augmentation en imágenes médicas
Optimizado para segmentación de vasos hepáticos
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import random
import math


class MedicalDataAugmentation:
    """
    Clase para data augmentation específico para imágenes médicas
    Todas las transformaciones son aplicadas tanto a imagen como máscara
    """
    
    def __init__(self, config):
        self.config = config
        self.rotation_range = config.get('rotation_range', 15)
        self.horizontal_flip = config.get('horizontal_flip', True)
        self.vertical_flip = config.get('vertical_flip', False)
        self.zoom_range = config.get('zoom_range', 0.1)
        self.brightness_range = config.get('brightness_range', 0.2)
        self.contrast_range = config.get('contrast_range', 0.2)
        self.noise_factor = config.get('noise_factor', 0.05)
        self.elastic_deform = config.get('elastic_deform', True)
        self.cutout_prob = config.get('cutout_prob', 0.1)
        self.cutout_size = config.get('cutout_size', 32)
    
    def __call__(self, image, mask):
        """Aplica transformaciones aleatorias a imagen y máscara"""
        # Convertir a numpy si es tensor
        if torch.is_tensor(image):
            image = image.squeeze().cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.squeeze().cpu().numpy()
        
        # Aplicar transformaciones geométricas (ambos)
        if random.random() < 0.7:  # 70% probabilidad
            image, mask = self._apply_geometric_transforms(image, mask)
        
        # Aplicar transformaciones de intensidad (solo imagen)
        if random.random() < 0.5:  # 50% probabilidad
            image = self._apply_intensity_transforms(image)
        
        # Aplicar ruido (solo imagen)
        if random.random() < 0.3:  # 30% probabilidad
            image = self._add_noise(image)
        
        # Aplicar cutout (ambos)
        if random.random() < self.cutout_prob:
            image, mask = self._apply_cutout(image, mask)
        
        # Convertir de vuelta a tensores
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
    def _apply_geometric_transforms(self, image, mask):
        """Aplica transformaciones geométricas"""
        # Rotación
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate_image(image, angle)
            mask = self._rotate_image(mask, angle, interpolation=cv2.INTER_NEAREST)
        
        # Flip horizontal
        if self.horizontal_flip and random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Flip vertical (no recomendado para anatomía)
        if self.vertical_flip and random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Zoom/Scale
        if self.zoom_range > 0:
            zoom_factor = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            image = self._zoom_image(image, zoom_factor)
            mask = self._zoom_image(mask, zoom_factor, interpolation=cv2.INTER_NEAREST)
        
        # Deformación elástica
        if self.elastic_deform and random.random() < 0.3:
            image, mask = self._elastic_deformation(image, mask)
        
        return image, mask
    
    def _apply_intensity_transforms(self, image):
        """Aplica transformaciones de intensidad (solo para imagen)"""
        # Cambio de brillo
        if self.brightness_range > 0:
            brightness_factor = random.uniform(-self.brightness_range, self.brightness_range)
            image = np.clip(image + brightness_factor, 0, 1)
        
        # Cambio de contraste
        if self.contrast_range > 0:
            contrast_factor = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            mean_val = np.mean(image)
            image = np.clip((image - mean_val) * contrast_factor + mean_val, 0, 1)
        
        # Gamma correction
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            image = np.power(image, gamma)
        
        return image
    
    def _add_noise(self, image):
        """Añade ruido gaussiano"""
        noise = np.random.normal(0, self.noise_factor, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def _rotate_image(self, image, angle, interpolation=cv2.INTER_LINEAR):
        """Rota imagen manteniendo el tamaño"""
        h, w = image.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=interpolation)
    
    def _zoom_image(self, image, zoom_factor, interpolation=cv2.INTER_LINEAR):
        """Aplica zoom manteniendo el tamaño"""
        h, w = image.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Crop o pad para mantener tamaño original
        if zoom_factor > 1:  # Zoom in - crop
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y:start_y + h, start_x:start_x + w]
        else:  # Zoom out - pad
            result = np.zeros((h, w), dtype=image.dtype)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            result[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            return result
    
    def _elastic_deformation(self, image, mask, alpha=30, sigma=5):
        """Aplica deformación elástica"""
        h, w = image.shape
        
        # Generar campos de desplazamiento aleatorios
        dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
        dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha
        
        # Crear grillas de coordenadas
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        
        # Aplicar deformación
        image_def = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
        mask_def = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')
        
        return image_def, mask_def
    
    def _apply_cutout(self, image, mask):
        """Aplica cutout (enmascara regiones aleatorias)"""
        h, w = image.shape
        
        # Seleccionar posición y tamaño aleatorios
        cut_h = random.randint(self.cutout_size // 2, self.cutout_size)
        cut_w = random.randint(self.cutout_size // 2, self.cutout_size)
        
        top = random.randint(0, h - cut_h)
        left = random.randint(0, w - cut_w)
        
        # Aplicar cutout
        image_cut = image.copy()
        mask_cut = mask.copy()
        
        image_cut[top:top + cut_h, left:left + cut_w] = 0
        mask_cut[top:top + cut_h, left:left + cut_w] = 0
        
        return image_cut, mask_cut


class NormalizationMethods:
    """Diferentes métodos de normalización para imágenes médicas"""
    
    @staticmethod
    def minmax_normalize(image):
        """Normalización Min-Max (0-1)"""
        min_val = np.min(image)
        max_val = np.max(image)
        return (image - min_val) / (max_val - min_val + 1e-8)
    
    @staticmethod
    def zscore_normalize(image):
        """Normalización Z-score (media=0, std=1)"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        return (image - mean_val) / (std_val + 1e-8)
    
    @staticmethod
    def clahe_normalize(image):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convertir a uint8 para CLAHE
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_uint8)
        
        # Convertir de vuelta a float [0,1]
        return image_clahe.astype(np.float32) / 255.0
    
    @staticmethod
    def percentile_normalize(image, lower=1, upper=99):
        """Normalización por percentiles"""
        p_low = np.percentile(image, lower)
        p_high = np.percentile(image, upper)
        
        image_norm = np.clip(image, p_low, p_high)
        return (image_norm - p_low) / (p_high - p_low + 1e-8)


def get_transforms(config, is_training=True):
    """
    Factory function para obtener transformaciones
    
    Args:
        config: Configuración de augmentation
        is_training: Si es True, aplica augmentation. Si es False, solo normalización.
    
    Returns:
        Función de transformación
    """
    
    if is_training and config.get('USE_DATA_AUGMENTATION', True):
        augmentation = MedicalDataAugmentation(config.get('AUGMENTATION_PARAMS', {}))
        
        def train_transform(image, mask):
            return augmentation(image, mask)
        
        return train_transform
    
    else:
        # Solo normalización para validación/test
        def val_transform(image, mask):
            # Convertir a tensor si es numpy
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).unsqueeze(0).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            return image, mask
        
        return val_transform
