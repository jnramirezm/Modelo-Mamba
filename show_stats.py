#!/usr/bin/env python3
"""
Script para mostrar estadísticas de entrenamiento en terminal
"""
import pandas as pd
import os
from pathlib import Path

def analyze_training_results(results_dir):
    """
    Analiza los resultados de entrenamiento y muestra estadísticas
    """
    results_dir = Path(results_dir)
    
    # Verificar que el directorio existe
    if not results_dir.exists():
        print(f"❌ Directorio no encontrado: {results_dir}")
        return
    
    # Cargar métricas
    csv_path = results_dir / "training_metrics.csv"
    if not csv_path.exists():
        print(f"❌ Archivo de métricas no encontrado: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print(f"📊 ANÁLISIS DE RESULTADOS: {results_dir.name}")
    print("="*80)
    
    # Información básica
    print(f"📈 Información básica:")
    print(f"   🔄 Épocas completadas: {len(df)}")
    print(f"   📁 Directorio: {results_dir}")
    print(f"   📋 Archivo métricas: {csv_path}")
    
    # Estadísticas de Loss
    print(f"\n📉 Estadísticas de Loss:")
    print(f"   🚂 Train Loss:")
    print(f"      Inicial: {df['train_loss'].iloc[0]:.6f}")
    print(f"      Final:   {df['train_loss'].iloc[-1]:.6f}")
    print(f"      Mejor:   {df['train_loss'].min():.6f} (época {df['train_loss'].idxmin() + 1})")
    print(f"      Mejora:  {((df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0] * 100):.1f}%")
    
    print(f"   🔍 Validation Loss:")
    print(f"      Inicial: {df['val_loss'].iloc[0]:.6f}")
    print(f"      Final:   {df['val_loss'].iloc[-1]:.6f}")
    print(f"      Mejor:   {df['val_loss'].min():.6f} (época {df['val_loss'].idxmin() + 1})")
    print(f"      Mejora:  {((df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]) / df['val_loss'].iloc[0] * 100):.1f}%")
    
    # Estadísticas de Dice Score
    print(f"\n🎯 Estadísticas de Dice Score:")
    print(f"   🚂 Train Dice:")
    print(f"      Inicial: {df['train_dice'].iloc[0]:.6f}")
    print(f"      Final:   {df['train_dice'].iloc[-1]:.6f}")
    print(f"      Mejor:   {df['train_dice'].max():.6f} (época {df['train_dice'].idxmax() + 1})")
    print(f"      Mejora:  {((df['train_dice'].iloc[-1] - df['train_dice'].iloc[0]) / (df['train_dice'].iloc[0] + 1e-8) * 100):.1f}%")
    
    print(f"   🔍 Validation Dice:")
    print(f"      Inicial: {df['val_dice'].iloc[0]:.6f}")
    print(f"      Final:   {df['val_dice'].iloc[-1]:.6f}")
    print(f"      Mejor:   {df['val_dice'].max():.6f} (época {df['val_dice'].idxmax() + 1})")
    print(f"      Mejora:  {((df['val_dice'].iloc[-1] - df['val_dice'].iloc[0]) / (df['val_dice'].iloc[0] + 1e-8) * 100):.1f}%")
    
    # Tabla de progreso por época
    print(f"\n📊 Progreso por época:")
    print(f"{'Época':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train Dice':<12} {'Val Dice':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{int(row['epoch']):<6} {row['train_loss']:<12.6f} {row['val_loss']:<12.6f} {row['train_dice']:<12.6f} {row['val_dice']:<12.6f}")
    
    # Análisis de tendencias
    print(f"\n🔍 Análisis de tendencias:")
    
    # Últimas 3 épocas
    if len(df) >= 3:
        recent_df = df.tail(3)
        
        val_dice_trend = "⬆️ creciente" if recent_df['val_dice'].is_monotonic_increasing else \
                        "⬇️ decreciente" if recent_df['val_dice'].is_monotonic_decreasing else \
                        "➡️ estable"
        
        train_dice_trend = "⬆️ creciente" if recent_df['train_dice'].is_monotonic_increasing else \
                          "⬇️ decreciente" if recent_df['train_dice'].is_monotonic_decreasing else \
                          "➡️ estable"
        
        print(f"   📈 Últimas 3 épocas:")
        print(f"      Val Dice: {val_dice_trend}")
        print(f"      Train Dice: {train_dice_trend}")
        
        # Detección de overfitting
        if recent_df['val_dice'].is_monotonic_decreasing and recent_df['train_dice'].is_monotonic_increasing:
            print(f"   ⚠️  POSIBLE OVERFITTING detectado")
        elif recent_df['val_dice'].is_monotonic_increasing:
            print(f"   ✅ Modelo mejorando consistentemente")
        else:
            print(f"   ℹ️  Entrenamiento estable")
    
    # Gap entre train y validation
    final_loss_gap = df['train_loss'].iloc[-1] - df['val_loss'].iloc[-1]
    final_dice_gap = df['val_dice'].iloc[-1] - df['train_dice'].iloc[-1]
    
    print(f"\n⚖️  Gap Train-Validation (final):")
    print(f"   Loss gap: {final_loss_gap:.6f} {'(⚠️ train > val)' if final_loss_gap > 0 else '(✅ val >= train)'}")
    print(f"   Dice gap: {final_dice_gap:.6f} {'(✅ val > train)' if final_dice_gap > 0 else '(⚠️ train >= val)'}")
    
    # Recomendaciones
    print(f"\n💡 Recomendaciones:")
    
    best_epoch = df['val_dice'].idxmax() + 1
    if best_epoch < len(df):
        print(f"   📍 El mejor modelo fue en época {best_epoch}, no al final")
        print(f"   💾 Usar el modelo guardado de época {best_epoch}")
    else:
        print(f"   ✅ El mejor modelo es el final")
    
    if df['val_dice'].max() < 0.1:
        print(f"   📈 Dice score bajo (<0.1), considerar:")
        print(f"      - Más épocas de entrenamiento")
        print(f"      - Ajustar learning rate")
        print(f"      - Verificar calidad de datos")
        print(f"      - Probar diferentes arquitecturas")
    
    if len(df) < 10:
        print(f"   🔄 Pocas épocas ({len(df)}), considerar entrenar más tiempo")
    
    # Archivos disponibles
    print(f"\n📁 Archivos disponibles:")
    for file_path in results_dir.glob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   📄 {file_path.name}: {size_mb:.1f} MB")
    
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        print(f"   📊 Plots en plots/:")
        for plot_path in plots_dir.glob("*.png"):
            print(f"      🖼️  {plot_path.name}")
    
    print("="*80)

def compare_experiments(experiments_dir):
    """
    Compara múltiples experimentos y muestra ranking
    """
    experiments_dir = Path(experiments_dir)
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print(f"❌ No se encontraron experimentos en: {experiments_dir}")
        return
    
    print("="*80)
    print(f"🏆 COMPARACIÓN DE EXPERIMENTOS")
    print("="*80)
    
    experiments_data = []
    
    for exp_dir in experiment_dirs:
        csv_path = exp_dir / "training_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            experiments_data.append({
                'Experimento': exp_dir.name,
                'Épocas': len(df),
                'Mejor Dice': df['val_dice'].max(),
                'Época Mejor': df['val_dice'].idxmax() + 1,
                'Dice Final': df['val_dice'].iloc[-1],
                'Loss Final (Train)': df['train_loss'].iloc[-1],
                'Loss Final (Val)': df['val_loss'].iloc[-1],
                'Mejora Dice': ((df['val_dice'].iloc[-1] - df['val_dice'].iloc[0]) / (df['val_dice'].iloc[0] + 1e-8) * 100)
            })
    
    if experiments_data:
        # Ordenar por mejor dice score
        experiments_data.sort(key=lambda x: x['Mejor Dice'], reverse=True)
        
        print(f"📊 Resumen de {len(experiments_data)} experimentos:")
        print()
        
        # Header
        print(f"{'Rank':<4} {'Experimento':<20} {'Mejor Dice':<12} {'Época':<6} {'Dice Final':<12} {'Épocas':<6}")
        print("-" * 70)
        
        # Data
        for i, exp in enumerate(experiments_data, 1):
            print(f"{i:<4} {exp['Experimento']:<20} {exp['Mejor Dice']:<12.6f} {exp['Época Mejor']:<6} {exp['Dice Final']:<12.6f} {exp['Épocas']:<6}")
        
        print()
        print(f"🥇 Mejor experimento: {experiments_data[0]['Experimento']}")
        print(f"   📈 Mejor Dice: {experiments_data[0]['Mejor Dice']:.6f}")
        print(f"   📍 Época: {experiments_data[0]['Época Mejor']}")
    
    print("="*80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_experiments(Path("outputs"))
        else:
            analyze_training_results(sys.argv[1])
    else:
        analyze_training_results("outputs/simple_testing")
