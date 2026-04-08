"""
inference.py — Run the trained UNet3D on the validation split and produce
               per-sample visualizations (comparison, slice strip, heatmap GIF).

Outputs are written to outputs/ and a summary line is printed per sample.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

np.random.seed(42)

# ── Config
CHECKPOINT     = "best_nodule_model.pth"
MANIFEST       = "./scan_manifest.csv"
OUTPUT_DIR     = "outputs"
NUM_SAMPLES_TO_VISUALIZE = 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_DIR       = "LungVoxels/NoduleMasks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

from model       import UNet3D
from data_loader import NEG_DIR, POS_DIR, LungNoduleDataset
from visualization import (
    visualize_full_comparison,
    visualize_slice_strip,
    save_heatmap_gif,
)


def load_model(checkpoint_path: str) -> UNet3D:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = UNet3D().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def build_val_indices(manifest_df):
    """Reproduce the exact 80/20 split from main_train.py (same seed)."""
    indices = np.random.permutation(len(manifest_df)).tolist()
    train_size = int(0.8 * len(manifest_df))
    return indices[train_size:]


def run():
    # ── 1. Load & clean manifest (mirrors main_train.py logic)
    base_dataset = LungNoduleDataset(
        csv_file=MANIFEST,
        pos_dir=POS_DIR,
        neg_dir=NEG_DIR,
    )

    def file_exists(row) -> bool:
        folder = POS_DIR if row["type"] == "positive" else NEG_DIR
        if not os.path.exists(os.path.join(folder, row["file"])):
            return False
        if row["type"] == "positive":
            return os.path.exists(os.path.join(MASK_DIR, str(row["mask_file"])))
        return True

    valid_df = (
        base_dataset.df[base_dataset.df.apply(file_exists, axis=1)]
        .reset_index(drop=True)
    )

    # ── 2. Reproduce val split
    val_indices = build_val_indices(valid_df)
    val_df = valid_df.iloc[val_indices].reset_index(drop=True)

    # ── 3. Select samples — at least one positive and one negative
    pos_rows = val_df[val_df["type"] == "positive"]
    neg_rows = val_df[val_df["type"] == "negative"]

    selected = []
    if len(pos_rows):
        selected.append(pos_rows.iloc[0])
    if len(neg_rows):
        selected.append(neg_rows.iloc[0])

    # Fill remaining slots from the rest of val set
    remaining = val_df.drop(index=[r.name for r in selected])
    extra_needed = NUM_SAMPLES_TO_VISUALIZE - len(selected)
    if extra_needed > 0:
        selected += [remaining.iloc[i] for i in range(min(extra_needed, len(remaining)))]

    # ── 4. Load model
    model = load_model(CHECKPOINT)

    # ── 5. Iterate and visualize
    val_dataset = LungNoduleDataset(MANIFEST, POS_DIR, NEG_DIR, train=False)
    val_dataset.df = valid_df.copy()

    print(f"\nRunning inference on {len(selected)} samples  |  device={DEVICE}\n")

    for row in selected:
        pat_id   = str(row["patient"]).replace("/", "-")
        row_idx  = valid_df[valid_df["file"] == row["file"]].index[0]

        cube, mask = val_dataset[row_idx]          # (1,32,32,32) tensors
        cube_t = cube.unsqueeze(0).to(DEVICE)      # (1,1,32,32,32)

        with torch.no_grad():
            pred = model(cube_t)                   # (1,1,32,32,32) in [0,1]

        ct_vol   = cube[0].numpy()                 # (32,32,32)
        pred_vol = pred[0, 0].cpu().numpy()        # (32,32,32)
        gt_vol   = mask[0].numpy()                 # (32,32,32)

        gt_present = bool(gt_vol.max() > 0.5)
        max_prob   = float(pred_vol.max())

        # Paths
        comp_path  = os.path.join(OUTPUT_DIR, f"pat{pat_id}_comparison.png")
        strip_path = os.path.join(OUTPUT_DIR, f"pat{pat_id}_strip.png")
        gif_path   = os.path.join(OUTPUT_DIR, f"pat{pat_id}_heatmap.gif")

        visualize_full_comparison(ct_vol, pred_vol, gt_vol, save_path=comp_path)
        visualize_slice_strip(ct_vol, pred_vol, gt_vol, save_path=strip_path)
        save_heatmap_gif(ct_vol, pred_vol, save_path=gif_path)

        print(
            f"patient={pat_id}  "
            f"max_pred={max_prob:.3f}  "
            f"gt_nodule={'yes' if gt_present else 'no '}  "
            f"→  {comp_path}  {strip_path}  {gif_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    run()
