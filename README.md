# 🧠 Hepatic Vessel Segmentation with UNet + Mamba Variants

Este proyecto entrena una red UNet 2D para segmentación de venas hepáticas en imágenes médicas, integrando variantes del bloque Mamba en el encoder y/o decoder.

---

## 📁 Estructura del proyecto

```
structured_project/
│
├── src/
│   ├── main.py                  # Entrena una sola variante desde config
│   ├── main_variants.py         # Entrena todas ("simple", "full", "v") y grafica
│   ├── plot_metrics.py          # Graficar manualmente resultados
│   ├── compare_results.csv      # Tabla con mejor Dice por variante
│   ├── results_*.csv            # Resultados por epoch
│   ├── plots/                   # Imágenes de métricas
│   ├── model/
│   │   ├── mamba_block.py
│   │   ├── unet_mamba_variants.py
│   │   └── v_mamba/             # Bloque VMamba
│   ├── preprocessing/
│   │   └── dataset.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── plotting.py
│   └── config.py
```

---

## 🧪 Entorno virtual (recomendado)

### 🐧 WSL2 (Recomendado para Windows)
```bash
# Migrar proyecto a WSL2 (mejor soporte CUDA)
mkdir -p ~/tesis && cd ~/tesis
cp -r /mnt/c/Tesis/Modelo\ VH/src .
cp -r /mnt/c/Tesis/Modelo\ VH/data .
cp /mnt/c/Tesis/Modelo\ VH/requirements.txt .

# Instalar dependencias del sistema
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential

# Crear ambiente virtual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🪟 Windows Nativo
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 🍎 Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Entrenamiento

### ▶️ 1. Entrenar una variante (definida en `config.py`):

```bash
python src/main.py
```

Genera:
- `hepatic_model_<variant>.pth`
- `results_<variant>.csv`

---

### ▶️ 2. Entrenar y comparar las tres variantes (simple, full, v):

```bash
python src/main_variants.py
```

Genera:
- `results_simple.csv`, `results_full.csv`, `results_v.csv`
- `compare_results.csv` con mejores métricas
- Gráficos: `plots/dice_score.png`, `plots/loss.png`

---

## 📊 Visualización manual (opcional)

```bash
python src/plot_metrics.py
```

---

## 🐳 Ejecución con Docker (Recomendado para servidores)

### 🗂️ Gestión de Múltiples Datasets

**IMPORTANTE:** Los datasets NO se copian al contenedor Docker. Se montan como volúmenes externos para:
- ✅ Imagen Docker ligera (~2GB vs >50GB)
- ✅ Flexibilidad para cambiar datasets
- ✅ No duplicar datos grandes
- ✅ Resultados organizados por dataset

```bash
# Estructura recomendada
datasets/
├── decathlon/          # Medical Segmentation Decathlon
│   ├── imagesTr/
│   └── labelsTr/
├── ircad/              # IRCAD dataset
│   ├── images/
│   └── masks/
└── custom/             # Tu dataset personalizado
    ├── images/
    └── masks/
```

### 🏗️ Construir la imagen

```bash
# Construir imagen Docker (SIN datos)
docker-compose build
```

### 🚀 Ejecutar entrenamiento

```bash
# Método 1: Scripts automatizados (Recomendado)
chmod +x run_dataset.sh

# Entrenar con dataset específico
./run_dataset.sh decathlon train       # Dataset Decathlon
./run_dataset.sh ircad variants        # Dataset IRCAD - todas las variantes
./run_dataset.sh custom jupyter        # Dataset personalizado + Jupyter

# Método 2: Docker Compose directo
docker-compose -f docker-compose.datasets.yml up hepatic-vessel-decathlon
docker-compose -f docker-compose.datasets.yml up jupyter-dev
```

### 📜 Scripts automatizados

```bash
# Linux/macOS
./run_dataset.sh [dataset] [comando]

# Windows
.\run_dataset.bat [dataset] [comando]

# Comandos disponibles:
#   train      - Entrenar una variante
#   variants   - Entrenar todas las variantes  
#   jupyter    - Abrir Jupyter Lab
#   compare    - Comparar configuraciones
#   visualize  - Generar visualizaciones
#   shell      - Abrir shell en contenedor
```

### 🗂️ Persistencia de datos

Los resultados se organizan automáticamente por dataset:
```
results/
├── decathlon/          # Resultados dataset Decathlon
├── ircad/              # Resultados dataset IRCAD  
└── custom/             # Resultados dataset personalizado

models/
├── decathlon/          # Modelos entrenados por dataset
├── ircad/
└── custom/
```

📖 **Guía completa:** Ver [DATASETS.md](DATASETS.md) para configuración detallada.

### ⚙️ Configuración Docker

**Variables de entorno importantes:**
- `DATASET_NAME=decathlon` - Nombre del dataset
- `IMAGES_DIR=/app/data/decathlon/imagesTr` - Ruta a imágenes
- `MASKS_DIR=/app/data/decathlon/labelsTr` - Ruta a máscaras
- `CUDA_VISIBLE_DEVICES=0` - GPU a usar
- `PYTHONPATH=/app` - Path de Python

**Requisitos del servidor:**
- Docker >= 20.10
- docker-compose >= 1.29
- NVIDIA Docker (para GPU)
- Espacio suficiente para datasets en host

---

## ⚙️ Configuración rápida

Modifica `src/config.py`:

```python
# Rutas de datos
IMAGES_DIR = "path/a/imagenes"
MASKS_DIR = "path/a/máscaras"

# Modelo
MAMBA_VARIANT = "simple"  # o "full" o "v"
N_EPOCHS = 10
BATCH_SIZE = 4

# Preprocesamiento NUEVO 🆕
IMAGE_SIZE = 256  # 224 o 256 (recomendado para segmentación)
NORMALIZE_METHOD = "minmax"  # "minmax", "zscore", "clahe", "percentile"
USE_DATA_AUGMENTATION = True

# Data Augmentation personalizable
AUGMENTATION_PARAMS = {
    "rotation_range": 15,        # Rotaciones ±15°
    "horizontal_flip": True,     # Flip horizontal
    "zoom_range": 0.1,          # Zoom ±10%
    "brightness_range": 0.2,     # Brillo ±20%
    "contrast_range": 0.2,       # Contraste ±20%
    "noise_factor": 0.05,        # Ruido gaussiano
    "elastic_deform": True,      # Deformación elástica
    "cutout_prob": 0.1          # Probabilidad de cutout
}
```

---

## 🔬 Comparación de configuraciones

```bash
# Comparar diferentes tamaños y métodos de normalización
python src/compare_preprocessing.py

# Resultados se guardan en experiments/preprocessing_comparison.csv
```

### 📏 **224px vs 256px - ¿Cuál elegir?**

**256x256 (Recomendado) ✅:**
- Mejor para segmentación médica
- Preserva más detalles de vasos pequeños
- Compatible con la mayoría de arquitecturas
- Buen balance velocidad/calidad

**224x224:**
- Más rápido de entrenar
- Menos memoria GPU
- Estándar en clasificación (ImageNet)
- Puede perder detalles finos

**Recomendación:** Usa **256px** para segmentación de vasos hepáticos.

### 🔧 **Métodos de Normalización**

- **MinMax** (Por defecto): Normaliza a [0,1], preserva distribución original
- **Z-Score**: Media=0, std=1, mejor para datos con distribución normal
- **CLAHE**: Mejora contraste local, excelente para imágenes médicas
- **Percentile**: Robusta a outliers, usa percentiles 1-99

### 🎯 **Data Augmentation Inteligente**

Transformaciones optimizadas para imágenes médicas:
- ✅ Rotaciones moderadas (±15°)
- ✅ Flips horizontales
- ✅ Zoom controlado
- ✅ Cambios de brillo/contraste
- ✅ Deformación elástica
- ✅ Ruido gaussiano
- ❌ Flips verticales (anatomía)
- ❌ Rotaciones extremas

---

## 📦 Requisitos clave

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-learn>=1.3.0
mamba-ssm>=1.2.0
nibabel>=5.1.0
jupyter>=1.0.0
```

---

## 🧠 Variantes Mamba

- `simple`: bloque Mamba simple propio
- `full`: bloque Mamba original (`mamba-ssm`)
- `v`: bloque VMamba (`v-mamba`)

---

## 📌 Notas

- Trabaja con cortes 2D (`.nii.gz`)
- Normaliza intensidades por slice
- Redimensiona todo a 256×256
- Aplica validación cruzada (split train/val)
- Optimizado para GPU con CUDA

---

## 👤 Autor

Proyecto de Tesis – Segmentación de venas hepáticas con UNet + Mamba
