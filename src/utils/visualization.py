import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def save_comparison_grid(lr, sr, hr, filename, metrics=None):
    """
    Saves a grid with LR, SR, and HR images.
    Expects tensors in [C, H, W] or [1, C, H, W] range [0, 1]
    """
    def to_numpy(t):
        if t.dim() == 4: t = t[0]
        return np.clip(t.permute(1, 2, 0).cpu().numpy(), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LR
    axes[0].imshow(to_numpy(lr))
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    # SR
    axes[1].imshow(to_numpy(sr))
    title = 'Super Resolved'
    if metrics:
        title += f"\nPSNR: {metrics.get('psnr', 0):.2f} | Steps: {metrics.get('steps', 'N/A')}"
    axes[1].set_title(title)
    axes[1].axis('off')
    
    # HR
    axes[2].imshow(to_numpy(hr))
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
