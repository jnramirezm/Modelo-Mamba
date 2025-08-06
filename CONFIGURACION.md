# Configuración de Modos de Entrenamiento

Este proyecto incluye dos modos de configuración para adaptarse a diferentes necesidades y hardware:

## 🧪 Modo TESTING/DESARROLLO (Por defecto)
**Ideal para:** Desarrollo, pruebas rápidas, hardware limitado (RTX 3060, etc.)

### Características:
- **Épocas:** 5 (entrenamiento rápido)
- **Batch size:** 2 (bajo uso de memoria)
- **Tamaño imagen:** 128x128 (procesamiento rápido)
- **Max muestras:** 20 (dataset pequeño)
- **Cache:** Desactivado (ahorra RAM)
- **Logging:** Cada 5 batches (detallado)
- **Modelo:** Versión ligera activada

## 🚀 Modo PRODUCCIÓN
**Ideal para:** Entrenamiento final, hardware potente, resultados óptimos

### Características:
- **Épocas:** 50 (entrenamiento completo)
- **Batch size:** 8 (mejor rendimiento)
- **Tamaño imagen:** 256x256 (máxima calidad)
- **Max muestras:** 1000 (dataset completo)
- **Cache:** Activado (mayor velocidad)
- **Logging:** Cada 20 batches (eficiente)
- **Modelo:** Versión completa

## 🔧 Cómo cambiar de modo

### Opción 1: Script de cambio de modo
```bash
# Activar modo testing (desarrollo)
python set_mode.py testing

# Activar modo producción
python set_mode.py production

# Ver modo actual
python set_mode.py status
```

### Opción 2: Variable de entorno
```bash
# Modo testing
export IS_TESTING=True
python src/main.py

# Modo producción
export IS_TESTING=False
python src/main.py
```

### Opción 3: En tiempo de ejecución
```bash
# Ejecutar directamente en modo específico
IS_TESTING=True python src/main.py    # Testing
IS_TESTING=False python src/main.py   # Producción
```

## 📋 Configuraciones personalizadas

También puedes sobrescribir configuraciones específicas usando variables de entorno:

```bash
# Ejemplo: Modo testing pero con más épocas
IS_TESTING=True N_EPOCHS=10 python src/main.py

# Ejemplo: Modo producción pero con menos muestras
IS_TESTING=False MAX_SAMPLES=100 python src/main.py
```

## 🎯 Recomendaciones de uso

### Para desarrollo y pruebas:
1. Usa modo **TESTING** por defecto
2. Perfecto para validar cambios de código
3. Entrenamiento completa en ~5-10 minutos

### Para entrenamiento final:
1. Cambia a modo **PRODUCCIÓN**
2. Asegúrate de tener suficiente memoria GPU
3. El entrenamiento puede tomar varias horas

### Para hardware intermedio:
```bash
# Configuración personalizada
IS_TESTING=True BATCH_SIZE=4 IMAGE_SIZE=192 MAX_SAMPLES=50 python src/main.py
```

## 🔍 Verificar configuración actual

```python
# Ver todas las configuraciones cargadas
python -c "from src.config import *; print(f'Modo: {"TESTING" if IS_TESTING else "PRODUCCIÓN"}'); print(f'Épocas: {N_EPOCHS}'); print(f'Batch: {BATCH_SIZE}'); print(f'Imagen: {IMAGE_SIZE}x{IMAGE_SIZE}'); print(f'Muestras: {MAX_SAMPLES}')"
```
