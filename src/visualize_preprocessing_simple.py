"""
Script para visualizar transformaciones de data augmentation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import IMAGES_DIR, MASKS_DIR, IMAGE_SIZE, AUGMENTATION_PARAMS
from preprocessing.dataset_v2 import HepaticVesselDatasetV2
from preprocessing.transforms import MedicalDataAugmentation, NormalizationMethods


def visualize_augmentations():
    """Visualiza ejemplos de data augmentation"""
    
    dataset_orig = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=IMAGE_SIZE,
        is_training=False,
        include_empty=False
    )
    
    augmentator = MedicalDataAugmentation(AUGMENTATION_PARAMS)
    
    sample_indices = [0, 10, 20, 30]
    
    fig, axes = plt.subplots(len(sample_indices), 6, figsize=(18, 3*len(sample_indices)))
    fig.suptitle("Data Augmentation Examples", fontsize=16)
    
    for row, idx in enumerate(sample_indices):
        if idx >= len(dataset_orig):
            continue
            
        orig_img, orig_mask = dataset_orig[idx]
        
        img_np = orig_img.squeeze().cpu().numpy()
        mask_np = orig_mask.squeeze().cpu().numpy()
        
        aug_img, aug_mask = augmentator(img_np, mask_np)
        
        aug_img_display = aug_img.squeeze().cpu().numpy()
        aug_mask_display = aug_mask.squeeze().cpu().numpy()
        
        slice_info = dataset_orig.get_slice_info(idx)
        vessel_pixels = slice_info['vessel_pixels']
        filename = slice_info['filename']
        
        if row == 0:
            titles = [
                "Original Image", 
                "Original Mask", 
                "Augmented Image", 
                "Augmented Mask",
                "Original Overlay",
                "Augmented Overlay"
            ]
        else:
            titles = ["", "", "", "", "", ""]
        
        axes[row, 0].imshow(img_np, cmap='gray')
        axes[row, 0].set_title(titles[0])
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(mask_np, cmap='red', alpha=0.8)
        axes[row, 1].set_title(titles[1])
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(aug_img_display, cmap='gray')
        axes[row, 2].set_title(titles[2])
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(aug_mask_display, cmap='red', alpha=0.8)
        axes[row, 3].set_title(titles[3])
        axes[row, 3].axis('off')
        
        axes[row, 4].imshow(img_np, cmap='gray')
        axes[row, 4].imshow(mask_np, cmap='red', alpha=0.3)
        axes[row, 4].set_title(titles[4])
        axes[row, 4].axis('off')
        
        axes[row, 5].imshow(aug_img_display, cmap='gray')
        axes[row, 5].imshow(aug_mask_display, cmap='red', alpha=0.3)
        axes[row, 5].set_title(titles[5])
        axes[row, 5].axis('off')
        
        info_text = f"{filename}\nVessels: {vessel_pixels}px"
        axes[row, 0].text(5, 15, info_text, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                          fontsize=8, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: augmentation_examples.png")


def compare_normalization_methods():
    """Compara diferentes métodos de normalización"""
    
    dataset = HepaticVesselDatasetV2(
        IMAGES_DIR, MASKS_DIR,
        image_size=IMAGE_SIZE,
        normalize_method="minmax",
        is_training=False
    )
    
    if len(dataset) == 0:
        print("No data found in dataset")
        return
    
    slice_info = dataset.get_slice_info(0)
    img_slice, _ = dataset._load_slice_data(slice_info)
    
    methods = {
        "Original": img_slice,
        "MinMax": NormalizationMethods.minmax_normalize(img_slice),
        "Z-Score": NormalizationMethods.zscore_normalize(img_slice),
        "CLAHE": NormalizationMethods.clahe_normalize(img_slice),
        "Percentile": NormalizationMethods.percentile_normalize(img_slice)
    }
    
    fig, axes = plt.subplots(1, len(methods), figsize=(20, 4))
    fig.suptitle("Normalization Methods Comparison", fontsize=16)
    
    for i, (method_name, normalized_img) in enumerate(methods.items()):
        axes[i].imshow(normalized_img, cmap='gray')
        title_text = f"{method_name}\nMin: {np.min(normalized_img):.3f}\nMax: {np.max(normalized_img):.3f}"
        axes[i].set_title(title_text)
        axes[i].axis('off')
        
        stats_text = f"Mean: {np.mean(normalized_img):.3f}\nStd: {np.std(normalized_img):.3f}"
        axes[i].text(0.02, 0.98, stats_text,
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig("normalization_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved: normalization_comparison.png")


if __name__ == "__main__":
    print("PREPROCESSING VISUALIZATION")
    print("=" * 50)
    
    try:
        print("1. Generating augmentation examples...")
        visualize_augmentations()
        
        print("2. Comparing normalization methods...")
        compare_normalization_methods()
        
        print("Visualization completed!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
