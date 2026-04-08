import matplotlib.pyplot as plt
import numpy as np
import torch
import os

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


def visualize_full_comparison(ct_volume, pred_heatmap, gt_mask, slice_idx=None, save_path=None):
    """
    Four-panel comparison figure for a single axial slice.

    Panels:
      1. Raw CT (grayscale)
      2. Ground-truth mask overlay (green, alpha=0.5, threshold=0.5)
      3. Model probability heatmap overlay (jet, alpha=0.5, threshold=0.2)
      4. Diff panel — GT in green, predictions in red; disagreements visible at a glance

    Args:
        ct_volume   : np.ndarray (D, H, W), normalised HU in [0, 1]
        pred_heatmap: np.ndarray (D, H, W), sigmoid probabilities in [0, 1]
        gt_mask     : np.ndarray (D, H, W), binary ground-truth mask
        slice_idx   : int, axial slice to render (defaults to middle)
        save_path   : str or None — if given, saves the figure; otherwise plt.show()
    """
    if slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2

    ct_slice   = ct_volume[slice_idx]
    pred_slice = pred_heatmap[slice_idx]
    gt_slice   = gt_mask[slice_idx]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1 — Raw CT
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title(f"Raw CT (slice {slice_idx})")
    axes[0].axis("off")

    # Panel 2 — GT overlay (green)
    axes[1].imshow(ct_slice, cmap="gray")
    gt_overlay = np.ma.masked_where(gt_slice < 0.5, gt_slice)
    axes[1].imshow(gt_overlay, cmap="Greens", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Overlay")
    axes[1].axis("off")

    # Panel 3 — Prediction heatmap overlay (jet)
    axes[2].imshow(ct_slice, cmap="gray")
    pred_overlay = np.ma.masked_where(pred_slice < 0.2, pred_slice)
    im = axes[2].imshow(pred_overlay, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Model Prediction Overlay")
    axes[2].axis("off")

    # Panel 4 — Diff: GT=green, Pred=red
    diff_rgb = np.zeros((*ct_slice.shape, 3), dtype=np.float32)
    gt_bin   = gt_slice >= 0.5
    pred_bin = pred_slice >= 0.2
    diff_rgb[gt_bin,   1] = 1.0   # green channel for GT
    diff_rgb[pred_bin, 0] = 1.0   # red channel for predictions
    axes[3].imshow(ct_slice, cmap="gray")
    axes[3].imshow(diff_rgb, alpha=0.6)
    axes[3].set_title("Diff (GT=green, Pred=red)")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_slice_strip(ct_volume, pred_heatmap, gt_mask, num_slices=8, save_path=None):
    """
    Horizontal strip of evenly spaced axial slices across the full depth.

    Each column shows: CT (top row) and prediction overlay (bottom row).
    Lets you see at a glance whether the model detects the nodule across all slices.

    Args:
        ct_volume   : np.ndarray (D, H, W)
        pred_heatmap: np.ndarray (D, H, W)
        gt_mask     : np.ndarray (D, H, W)
        num_slices  : int, how many evenly spaced slices to show
        save_path   : str or None
    """
    depth = ct_volume.shape[0]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 3, 9))

    for col, idx in enumerate(indices):
        ct_slice   = ct_volume[idx]
        pred_slice = pred_heatmap[idx]
        gt_slice   = gt_mask[idx]

        # Row 0 — Raw CT
        axes[0, col].imshow(ct_slice, cmap="gray")
        axes[0, col].set_title(f"z={idx}", fontsize=8)
        axes[0, col].axis("off")

        # Row 1 — GT overlay
        axes[1, col].imshow(ct_slice, cmap="gray")
        gt_overlay = np.ma.masked_where(gt_slice < 0.5, gt_slice)
        axes[1, col].imshow(gt_overlay, cmap="Greens", alpha=0.5, vmin=0, vmax=1)
        axes[1, col].axis("off")

        # Row 2 — Prediction overlay
        axes[2, col].imshow(ct_slice, cmap="gray")
        pred_overlay = np.ma.masked_where(pred_slice < 0.2, pred_slice)
        axes[2, col].imshow(pred_overlay, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("CT", fontsize=9)
    axes[1, 0].set_ylabel("GT", fontsize=9)
    axes[2, 0].set_ylabel("Pred", fontsize=9)

    plt.suptitle("Slice Strip (CT / GT / Prediction)", fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_heatmap_gif(ct_volume, pred_heatmap, save_path, threshold=0.2, fps=6):
    """
    Saves an animated GIF cycling through all axial slices with the prediction overlay.

    Args:
        ct_volume   : np.ndarray (D, H, W)
        pred_heatmap: np.ndarray (D, H, W)
        save_path   : str, must end in .gif
        threshold   : float, mask predictions below this value
        fps         : int, frames per second
    """
    try:
        import imageio
        import io
    except ImportError:
        raise ImportError("imageio is required for GIF export: pip install imageio")

    frames = []
    depth = ct_volume.shape[0]

    for z in range(depth):
        ct_slice   = ct_volume[z]
        pred_slice = pred_heatmap[z]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ct_slice, cmap="gray")
        masked = np.ma.masked_where(pred_slice < threshold, pred_slice)
        ax.imshow(masked, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        frames.append(frame)
        plt.close(fig)
        buf.close()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)