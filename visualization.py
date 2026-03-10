import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_radiology_heatmap(ct_volume, pred_heatmap, slice_idx=None):
    """
    ct_volume: 3D numpy array (normalized HU units)
    pred_heatmap: 3D numpy array (Sigmoid outputs 0.0 to 1.0)
    slice_idx: The specific axial slice to view (defaults to middle)
    """
    if slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original CT Scan (The Anatomy)
    ax[0].imshow(ct_volume[slice_idx], cmap='gray')
    ax[0].set_title(f"Original CT (Slice {slice_idx})")
    ax[0].axis('off')

    # 2. Probability Heatmap (The AI's Confidence)
    # We use 'jet' or 'hot' colormap to show intensity
    im2 = ax[1].imshow(pred_heatmap[slice_idx], cmap='jet', vmin=0, vmax=1)
    ax[1].set_title("AI Confidence Heatmap")
    ax[1].axis('off')
    plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

    # 3. Radiologist Overlay (The Combined View)
    # We overlay the heatmap on the CT with transparency (alpha)
    ax[2].imshow(ct_volume[slice_idx], cmap='gray')
    # Mask out very low confidence areas to keep the view clean
    masked_heatmap = np.ma.masked_where(pred_heatmap[slice_idx] < 0.2, pred_heatmap[slice_idx])
    ax[2].imshow(masked_heatmap, cmap='Reds', alpha=0.5) 
    ax[2].set_title("Radiologist Overlay (Threshold > 0.2)")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

# Assuming 'cubes' is your input and 'outputs' is your model(cubes)
# slice_to_show = 16 # Middle of a 32x32x32 cube
# visualize_radiology_heatmap(cubes[0,0].cpu().numpy(), outputs[0,0].detach().cpu().numpy(), slice_idx=slice_to_show)