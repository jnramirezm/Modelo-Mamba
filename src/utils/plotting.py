import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configurar estilo de matplotlib
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10
})
sns.set_palette("husl")

def plot_training_history(train_losses, val_losses, train_dice_scores, val_dice_scores, 
                         save_path="training_history.png", show_plot=True):
    """
    Plotea la historia de entrenamiento con loss y dice score
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Loss durante el entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot Dice Score
    ax2.plot(epochs, train_dice_scores, 'b-', label='Train Dice', linewidth=2, marker='o')
    ax2.plot(epochs, val_dice_scores, 'r-', label='Validation Dice', linewidth=2, marker='s')
    ax2.set_title('Dice Score durante el entrenamiento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Añadir anotaciones para el mejor score
    best_val_dice_idx = np.argmax(val_dice_scores)
    best_val_dice = val_dice_scores[best_val_dice_idx]
    ax2.annotate(f'Mejor: {best_val_dice:.4f}', 
                xy=(best_val_dice_idx + 1, best_val_dice),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico de entrenamiento guardado en: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_model_predictions(model, dataloader, device, num_samples=6, 
                          save_path="model_predictions.png", show_plot=True):
    """
    Plotea predicciones del modelo vs ground truth
    """
    model.eval()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 4, 12))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            # Tomar la primera imagen del batch
            img = images[0, 0].cpu().numpy()  # Primer canal
            mask = masks[0, 0].cpu().numpy()
            pred = predictions[0, 0].cpu().numpy()
            
            # Imagen original
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Imagen {i+1}', fontweight='bold')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(mask, cmap='Reds', alpha=0.8)
            axes[1, i].set_title('Ground Truth', fontweight='bold')
            axes[1, i].axis('off')
            
            # Predicción
            axes[2, i].imshow(pred, cmap='Blues', alpha=0.8)
            axes[2, i].set_title('Predicción', fontweight='bold')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Predicciones del modelo guardadas en: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_model_comparison_predictions(model, dataloader, device, num_samples=4,
                                    save_path="model_comparison.png", show_plot=True):
    """
    Plotea comparación lado a lado: Imagen original, Ground truth, Predicción, Overlay
    """
    model.eval()
    
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 4, 16))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            pred_binary = predictions > 0.5
            
            # Tomar la primera imagen del batch
            img = images[0, 0].cpu().numpy()
            mask = masks[0, 0].cpu().numpy()
            pred_prob = predictions[0, 0].cpu().numpy()
            pred = pred_binary[0, 0].cpu().numpy()
            
            # Imagen original
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(mask, cmap='Reds')
            axes[1, i].set_title('Ground Truth', fontweight='bold')
            axes[1, i].axis('off')
            
            # Predicción probabilística
            axes[2, i].imshow(pred_prob, cmap='Blues', vmin=0, vmax=1)
            axes[2, i].set_title('Pred. Probabilidad', fontweight='bold')
            axes[2, i].axis('off')
            
            # Overlay
            axes[3, i].imshow(img, cmap='gray')
            axes[3, i].imshow(pred, cmap='Reds', alpha=0.5)
            axes[3, i].set_title('Overlay', fontweight='bold')
            axes[3, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Comparación del modelo guardada en: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_training_summary_plot(train_losses, val_losses, train_dice_scores, val_dice_scores,
                               model, val_dataloader, device, save_dir="outputs/plots"):
    """
    Crea un resumen completo del entrenamiento con múltiples visualizaciones
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot de métricas de entrenamiento
    plot_training_history(train_losses, val_losses, train_dice_scores, val_dice_scores,
                         save_path=os.path.join(save_dir, "training_history.png"),
                         show_plot=False)
    
    # 2. Plot de predicciones del modelo
    plot_model_predictions(model, val_dataloader, device, num_samples=6,
                         save_path=os.path.join(save_dir, "model_predictions.png"),
                         show_plot=False)
    
    # 3. Plot de comparación detallada
    plot_model_comparison_predictions(model, val_dataloader, device, num_samples=4,
                                    save_path=os.path.join(save_dir, "model_comparison.png"),
                                    show_plot=False)
    
    # 4. Crear dashboard combinado
    create_training_dashboard(train_losses, val_losses, train_dice_scores, val_dice_scores,
                            save_path=os.path.join(save_dir, "training_dashboard.png"))
    
    print(f"📊 Resumen completo guardado en: {save_dir}")

def create_training_dashboard(train_losses, val_losses, train_dice_scores, val_dice_scores,
                            save_path="training_dashboard.png"):
    """
    Crea un dashboard completo con todas las métricas
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Loss plot grande
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_title('Loss durante el entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice plot grande
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(epochs, train_dice_scores, 'b-', label='Train Dice', linewidth=2, marker='o')
    ax2.plot(epochs, val_dice_scores, 'r-', label='Validation Dice', linewidth=2, marker='s')
    ax2.set_title('Dice Score durante el entrenamiento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Estadísticas finales
    ax3 = fig.add_subplot(gs[:2, 2])
    ax3.axis('off')
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_train_dice = train_dice_scores[-1]
    final_val_dice = val_dice_scores[-1]
    best_val_dice = max(val_dice_scores)
    best_epoch = val_dice_scores.index(best_val_dice) + 1
    
    stats_text = f"""
RESUMEN DEL ENTRENAMIENTO

📈 Métricas Finales:
• Train Loss: {final_train_loss:.4f}
• Val Loss: {final_val_loss:.4f}
• Train Dice: {final_train_dice:.4f}
• Val Dice: {final_val_dice:.4f}

🏆 Mejores Resultados:
• Mejor Val Dice: {best_val_dice:.4f}
• Época: {best_epoch}

📊 Progreso:
• Total épocas: {len(epochs)}
• Mejora Loss: {(train_losses[0] - final_train_loss)/train_losses[0]*100:.1f}%
• Mejora Dice: {(final_train_dice - train_dice_scores[0])*100:.1f}%
"""
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Distribución de mejoras por época
    ax4 = fig.add_subplot(gs[2, :])
    dice_improvements = np.diff([0] + val_dice_scores)
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in dice_improvements]
    
    ax4.bar(epochs, dice_improvements, color=colors, alpha=0.7)
    ax4.set_title('Mejora del Dice Score por Época', fontweight='bold')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Cambio en Dice Score')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.suptitle('Dashboard de Entrenamiento - UNet + Mamba', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard guardado en: {save_path}")
    
    plt.close()

def load_results(variants, base_dir="outputs/results"):
    """Función original mantenida para compatibilidad"""
    dfs = []
    for variant in variants:
        path = os.path.join(base_dir, f"results_{variant}.csv")
        df = pd.read_csv(path)
        df["Variant"] = variant
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_metrics(df, save_dir="outputs/plots"):
    """Función original mantenida para compatibilidad"""
    os.makedirs(save_dir, exist_ok=True)

    # Gráficos individuales por variante
    variants = df["Variant"].unique()
    for variant in variants:
        sub = df[df["Variant"] == variant]
        for metric in ["Loss", "Dice"]:
            plt.figure()
            plt.plot(sub["Epoch"], sub[metric], label=variant)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} por Epoch - {variant}")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"{variant}_{metric.lower()}.png"))
            plt.close()

    # Gráficos combinados para comparación general
    for metric in ["Loss", "Dice"]:
        plt.figure()
        for variant in variants:
            sub = df[df["Variant"] == variant]
            plt.plot(sub["Epoch"], sub[metric], label=variant)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} por Epoch - Comparación Global")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{metric.lower()}_score.png"))
        plt.close()
