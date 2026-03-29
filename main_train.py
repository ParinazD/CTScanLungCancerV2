# IMPORTS
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# Local imports
from model import UNet3D
from losses import UnifiedFocalLoss
from data_loader import NEG_DIR, POS_DIR, LungNoduleDataset


# ── Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2          # 32³ volumes × 256 channels are VRAM-heavy; start small
ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE × ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
MASK_DIR = "LungVoxels/NoduleMasks"


# ── Dice metric (threshold-based, used for evaluation only)
def dice_metric(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Hard Dice computed after binarising predictions at `threshold`.
    Use this for evaluation; not for training loss.
    """
    preds_bin = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds_bin * targets).sum()
    return ((2.0 * intersection + smooth) / (preds_bin.sum() + targets.sum() + smooth)).item()


# ── Validation
def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """
    Returns mean hard-Dice over the validation set.
    Augmentation is already disabled because val_dataset was built with train=False.
    """
    model.eval()
    total_dice = 0.0

    with torch.no_grad():
        for cubes, masks in loader:
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)
            preds = model(cubes)                    # (B, 1, 32, 32, 32) in [0, 1]
            total_dice += dice_metric(preds, masks)

    return total_dice / max(len(loader), 1)


# ── Training loop
def train():

    # ── 1. Load & clean manifest
    base_dataset = LungNoduleDataset(
        csv_file="./scan_manifest.csv",
        pos_dir=POS_DIR,
        neg_dir=NEG_DIR,
    )
    print(f"Initial manifest size: {len(base_dataset.df)}")

    def file_exists(row) -> bool:
        folder = POS_DIR if row["type"] == "positive" else NEG_DIR
        if not os.path.exists(os.path.join(folder, row["file"])):
            return False
        if row["type"] == "positive":
            mask_path = os.path.join(MASK_DIR, str(row["mask_file"]))
            return os.path.exists(mask_path)
        return True

    valid_df = (
        base_dataset.df[base_dataset.df.apply(file_exists, axis=1)]
        .reset_index(drop=True)
    )
    print(f"Cleaned manifest size: {len(valid_df)} (removed missing files)")

    # ── 2. Train / val split
    indices = np.random.permutation(len(valid_df)).tolist()
    train_size = int(0.8 * len(valid_df))
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    # Two dataset instances so augmentation is ON for train, OFF for val
    train_dataset = LungNoduleDataset("./scan_manifest.csv", POS_DIR, NEG_DIR, train=True)
    train_dataset.df = valid_df.copy()

    val_dataset = LungNoduleDataset("./scan_manifest.csv", POS_DIR, NEG_DIR, train=False)
    val_dataset.df = valid_df.copy()

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(val_dataset,   val_indices)

    # ── 3. Balanced sampler (10× weight for positive samples)
    train_types = valid_df.iloc[train_indices]["type"].tolist()
    weights = [10.0 if t == "positive" else 1.0 for t in train_types]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              shuffle=False, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── 4. Model, loss, optimiser
    model = UNet3D().to(DEVICE)

    # UnifiedFocalLoss handles the ~0.1% foreground imbalance best.
    criterion = UnifiedFocalLoss(gamma=2.0, lambda_focal=0.5, smooth=1e-6)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Reduce LR when val Dice stops improving (mode='max')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4)

    # ── 5. Epoch loop
    best_val_dice = 0.0
    print(f"\nTraining on {DEVICE}  |  {len(train_ds)} train  /  {len(val_ds)} val samples\n")

    for epoch in range(EPOCHS):

        # ── LR warm-up for first 2 epochs
        if epoch < 2:
            warmup_lr = LEARNING_RATE * 0.1 * (epoch + 1)   # 10% → 20%
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr

        # ── Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (cubes, masks) in enumerate(train_loader):
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)

            outputs = model(cubes)                  # (B, 1, 32, 32, 32) ∈ [0, 1]

            # Loss divided by ACCUMULATION_STEPS so gradients scale correctly
            loss = criterion(outputs, masks) / ACCUMULATION_STEPS
            loss.backward()

            # Step + zero_grad only every ACCUMULATION_STEPS batches
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATION_STEPS   # undo the /ACCUM for logging

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation (once per epoch, outside batch loop)
        val_dice = evaluate_model(model, val_loader)

        # Scheduler steps on val Dice (higher = better)
        scheduler.step(val_dice)

        print(
            f"Epoch [{epoch+1:>3}/{EPOCHS}]  "
            f"Train Loss: {avg_train_loss:.4f}  |  "
            f"Val Dice: {val_dice:.4f}  |  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── Checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_nodule_model.pth")
            print(f"Improved — saved checkpoint (Dice: {val_dice:.4f})")

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train()
