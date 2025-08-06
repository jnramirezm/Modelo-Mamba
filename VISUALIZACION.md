# Sistema de Visualización y Análisis de Entrenamiento

## 🎯 Funcionalidades Implementadas

### 1. **Entrenamiento con Tracking Automático**
- ✅ Guardado automático de métricas (loss, dice score) por época
- ✅ Generación automática de plots durante el entrenamiento
- ✅ Guardado organizado de modelos y resultados
- ✅ Dashboard completo con estadísticas

### 2. **Visualizaciones Generadas Automáticamente**

#### Durante el entrenamiento se crean:
- **`training_history.png`** - Gráficos de loss y dice score
- **`model_predictions.png`** - Predicciones del modelo vs ground truth
- **`model_comparison.png`** - Comparación detallada lado a lado
- **`training_dashboard.png`** - Dashboard completo con estadísticas
- **`training_metrics.csv`** - Métricas en formato CSV para análisis

## 📊 Cómo Usar

### Entrenamiento Básico (con visualizaciones automáticas):
```bash
# Modo testing (rápido, hardware limitado)
python src/main.py

# Modo producción (completo)
python set_mode.py production
python src/main.py
```

### Análisis de Resultados:
```bash
# Mostrar estadísticas detalladas en terminal
python show_stats.py outputs/simple_testing

# Generar análisis visual detallado (crea detailed_analysis.png)
python analyze_results.py -d outputs/simple_testing

# Comparar múltiples experimentos
python show_stats.py --compare
python analyze_results.py --compare
```

## 📁 Estructura de Outputs

```
outputs/
├── simple_testing/                    # Experimento en modo testing
│   ├── best_hepatic_model_simple_opt.pth     # Mejor modelo
│   ├── final_hepatic_model_simple_opt.pth    # Modelo final
│   ├── training_metrics.csv                   # Métricas en CSV
│   ├── training_history.png                   # Gráfico básico
│   ├── detailed_analysis.png                  # Análisis detallado
│   └── plots/                                 # Plots del entrenamiento
│       ├── training_dashboard.png             # Dashboard completo
│       ├── model_predictions.png              # Predicciones del modelo
│       ├── model_comparison.png               # Comparación detallada
│       └── training_history.png               # Historia de entrenamiento
└── simple_production/                # Experimento en modo producción
    └── [misma estructura]
```

## 📈 Tipos de Visualizaciones

### 1. **Training History** 📊
- Loss y Dice Score por época
- Comparación train vs validation
- Identificación del mejor modelo

### 2. **Model Predictions** 🔍
- Muestras de predicciones del modelo
- Comparación con ground truth
- Visualización de la calidad de segmentación

### 3. **Model Comparison** 🔬
- Imagen original, ground truth, predicción, overlay
- Análisis detallado de casos específicos
- Mapas de probabilidad

### 4. **Training Dashboard** 📋
- Resumen completo del entrenamiento
- Estadísticas clave
- Progreso por época
- Detección de overfitting

### 5. **Detailed Analysis** 📈
- Análisis de tendencias
- Detección de overfitting
- Gap train-validation
- Tabla de estadísticas

## 🔍 Análisis Automático

### El sistema detecta automáticamente:
- **Overfitting**: Cuando train mejora pero validation empeora
- **Mejor época**: Cuándo se obtuvo el mejor modelo
- **Tendencias**: Si el modelo está mejorando o estancado
- **Recomendaciones**: Qué hacer para mejorar el entrenamiento

### Ejemplo de output del análisis:
```
📊 ANÁLISIS DE RESULTADOS: simple_testing
📈 Información básica:
   🔄 Épocas completadas: 5
   📁 Directorio: outputs/simple_testing

📉 Estadísticas de Loss:
   🚂 Train Loss: 0.159726 → 0.010761 (93.3% mejora)
   🔍 Validation Loss: 0.012439 → 0.009556 (23.2% mejora)

🎯 Estadísticas de Dice Score:
   🚂 Train Dice: 0.013394 → 0.022323 (66.7% mejora)
   🔍 Validation Dice: 0.017858 → 0.017858 (0.0% mejora)

💡 Recomendaciones:
   📍 El mejor modelo fue en época 1, no al final
   💾 Usar el modelo guardado de época 1
   📈 Dice score bajo (<0.1), considerar más entrenamiento
```

## 🎮 Comandos Útiles

### Entrenamiento y análisis rápido:
```bash
# Entrenar en modo testing
python src/main.py

# Ver estadísticas inmediatamente
python show_stats.py outputs/simple_testing
```

### Comparar experimentos:
```bash
# Entrenar diferentes configuraciones
python set_mode.py testing && python src/main.py
python set_mode.py production && python src/main.py

# Comparar resultados
python show_stats.py --compare
```

### Análisis visual:
```bash
# Generar plots detallados
python analyze_results.py -d outputs/simple_testing

# Los plots se abren automáticamente y se guardan como PNG
```

## 🔧 Personalización

### Configurar plots en `utils/plotting.py`:
- Cambiar estilos de visualización
- Añadir nuevas métricas
- Personalizar dashboards

### Configurar análisis en `show_stats.py`:
- Cambiar umbrales de detección
- Añadir nuevas recomendaciones
- Personalizar formato de output

## 📝 Métricas Guardadas

### En `training_metrics.csv`:
```csv
epoch,train_loss,val_loss,train_dice,val_dice
1,0.159726,0.012439,0.013394,0.017858
2,0.012486,0.011525,0.013394,0.017858
...
```

### Esto permite:
- Análisis post-entrenamiento
- Comparación entre experimentos
- Generación de reports
- Integración con herramientas externas

## 🎯 Beneficios

1. **Monitoreo en tiempo real** durante entrenamiento
2. **Análisis automático** de resultados
3. **Detección temprana** de problemas
4. **Comparación fácil** entre experimentos
5. **Documentación visual** completa
6. **Reproducibilidad** mejorada

¡Ahora tienes un sistema completo para analizar y visualizar tus entrenamientos! 🚀
