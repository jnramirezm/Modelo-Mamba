from pathlib import Path
import os

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Rutas a imágenes y máscaras - Configurables con variables de entorno
IMAGES_DIR = os.getenv('IMAGES_DIR', str(DATA_DIR / "decathlon" / "imagesTr"))
MASKS_DIR = os.getenv('MASKS_DIR', str(DATA_DIR / "decathlon" / "labelsTr"))

# Nombre del dataset (útil para organizar resultados)
DATASET_NAME = os.getenv('DATASET_NAME', 'decathlon')

# ====================================================================
# CONFIGURACIÓN DE MODO DE ENTRENAMIENTO
# ====================================================================
IS_TESTING = os.getenv('IS_TESTING', 'True').lower() == 'true'  # Modo testing/desarrollo

# Parámetros básicos de entrenamiento
RANDOM_SEED = 42
VAL_SPLIT = 0.2

# Variante de Mamba: "simple", "full" o "v"
MAMBA_VARIANT = os.getenv('MAMBA_VARIANT', 'simple')

# ====================================================================
# CONFIGURACIONES SEGÚN EL MODO
# ====================================================================

if IS_TESTING:
    # CONFIGURACIÓN PARA DESARROLLO/TESTING (Hardware limitado)
    print("🧪 Modo TESTING - Configuración para desarrollo/hardware limitado")
    
    N_EPOCHS = int(os.getenv('N_EPOCHS', '5'))          # Pocas épocas para pruebas rápidas
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))      # Batch pequeño para RTX 3060
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '128'))    # Imagen pequeña para menos memoria
    MAX_SAMPLES = int(os.getenv('MAX_SAMPLES', '20'))   # Máximo 20 muestras para pruebas
    MAX_SLICES_PER_VOLUME = int(os.getenv('MAX_SLICES_PER_VOLUME', '10'))  # Máximo 10 slices
    
    # Cache y memoria optimizados
    USE_CACHE = os.getenv('USE_CACHE', 'False').lower() == 'true'  # Sin cache para ahorrar RAM
    USE_LIGHTWEIGHT_MODEL = os.getenv('USE_LIGHTWEIGHT_MODEL', 'True').lower() == 'true'
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '2'))    # 2 workers para Linux/WSL
    PREFETCH_FACTOR = int(os.getenv('PREFETCH_FACTOR', '2'))  # Prefetch reducido
    
    # Logging más frecuente para desarrollo
    LOG_INTERVAL = int(os.getenv('LOG_INTERVAL', '5'))  # Log cada 5 batches
    VALIDATE_INTERVAL = int(os.getenv('VALIDATE_INTERVAL', '1'))  # Validar cada época
    
else:
    # CONFIGURACIÓN PARA PRODUCCIÓN (Hardware completo)
    print("🚀 Modo PRODUCCIÓN - Configuración para entrenamiento completo")
    
    N_EPOCHS = int(os.getenv('N_EPOCHS', '50'))         # Entrenamiento completo
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))      # Batch más grande
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '256'))    # Imagen completa
    MAX_SAMPLES = int(os.getenv('MAX_SAMPLES', '1000')) # Todas las muestras disponibles
    MAX_SLICES_PER_VOLUME = int(os.getenv('MAX_SLICES_PER_VOLUME', '50'))  # Más slices
    
    # Cache y memoria optimizados para rendimiento
    USE_CACHE = os.getenv('USE_CACHE', 'True').lower() == 'true'  # Cache activado
    USE_LIGHTWEIGHT_MODEL = os.getenv('USE_LIGHTWEIGHT_MODEL', 'False').lower() == 'true'
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))    # Más workers
    PREFETCH_FACTOR = int(os.getenv('PREFETCH_FACTOR', '4'))  # Más prefetch
    
    # Logging menos frecuente para producción
    LOG_INTERVAL = int(os.getenv('LOG_INTERVAL', '20'))  # Log cada 20 batches
    VALIDATE_INTERVAL = int(os.getenv('VALIDATE_INTERVAL', '5'))  # Validar cada 5 épocas

# Parámetros comunes
NORMALIZE_METHOD = os.getenv('NORMALIZE_METHOD', 'minmax')  # "minmax", "zscore", "clahe"
USE_DATA_AUGMENTATION = os.getenv('USE_DATA_AUGMENTATION', 'True').lower() == 'true'
SAVE_CHECKPOINTS = os.getenv('SAVE_CHECKPOINTS', 'True').lower() == 'true'

# Parámetros de Data Augmentation
AUGMENTATION_PARAMS = {
    "rotation_range": 15,        # Grados de rotación
    "horizontal_flip": True,     # Flip horizontal
    "vertical_flip": False,      # Flip vertical (no recomendado para anatomía)
    "zoom_range": 0.1,          # Zoom in/out
    "brightness_range": 0.2,     # Cambio de brillo
    "contrast_range": 0.2,       # Cambio de contraste
    "noise_factor": 0.05,        # Ruido gaussiano
    "elastic_deform": True,      # Deformación elástica
    "cutout_prob": 0.1,         # Probabilidad de cutout
    "cutout_size": 32           # Tamaño de cutout
}
