#IMPORTS 
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import copy
import numpy as np

#local imports
from model import UNet3D
from losses import DiceLoss
from data_loader import NEG_DIR, POS_DIR, LungNoduleDataset


# configuration & MLOps
DEVICE = torch.device("cpu")
BATCH_SIZE = 8  # Optimized for 16GB VRAM
LEARNING_RATE = 1e-4
EPOCHS = 20

def train():
    # 1. Load the initial manifest for filtering
    # We do this once to clean the data before creating train/val sets
    base_dataset = LungNoduleDataset(csv_file="./scan_manifest.csv", pos_dir=POS_DIR, neg_dir=NEG_DIR)
    
    # --- START CLEAN-UP CODE ---
    print(f"Initial manifest size: {len(base_dataset.df)}")

    def file_exists(row):
        folder = POS_DIR if row['type'] == 'positive' else NEG_DIR
        cube_path = os.path.join(folder, row['file'])
        if not os.path.exists(cube_path):
            return False
        if row['type'] == 'positive':
            mask_path = os.path.join("LungVoxels/NoduleMasks", str(row['mask_file']))
            return os.path.exists(mask_path)
        return True

    # Filter the DataFrame
    valid_df = base_dataset.df[base_dataset.df.apply(file_exists, axis=1)].reset_index(drop=True)
    print(f"Cleaned manifest size: {len(valid_df)} (Filtered out missing files)")
    # --- END CLEAN-UP CODE ---

    # 2. Create TWO dataset instances to separate Training and Validation behavior
    # Training dataset: train=True (Augmentation ON)
    train_dataset = LungNoduleDataset(csv_file="./scan_manifest.csv", pos_dir=POS_DIR, neg_dir=NEG_DIR, train=True)
    train_dataset.df = valid_df.copy() # Use the filtered data

    # Validation dataset: train=False (Augmentation OFF)
    val_dataset = LungNoduleDataset(csv_file="./scan_manifest.csv", pos_dir=POS_DIR, neg_dir=NEG_DIR, train=False)
    val_dataset.df = valid_df.copy() # Use the filtered data

    # 3. train-test-Split indices
    indices = list(range(len(valid_df)))
    train_size = int(0.8 * len(valid_df))
    
    # Manually split indices to ensure no overlap
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets using the separate dataset objects
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = UNet3D().to(DEVICE)
    criterion_dice = DiceLoss()
    criterion_bce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for cubes, masks in train_loader:
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(cubes)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Step (Evaluation)
        val_dice = evaluate_model(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f} - Val Dice: {val_dice:.4f}")

        # Save Checkpoint (MLOps)
        torch.save(model.state_dict(), "model_with_augmentation.pth")

def evaluate_model(model, loader):
   
    model.eval()
    total_dice = 0
    smooth = 1e-6

    with torch.no_grad():
        for cubes, masks in loader:
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)
            preds = model(cubes)
            
            # Flatten tensors to calculate overlap across the whole batch
            preds_f = preds.view(-1)
            masks_f = masks.view(-1)
            
            intersection = (preds_f * masks_f).sum()
            dice_score = (2. * intersection + smooth) / (preds_f.sum() + masks_f.sum() + smooth)
            
            total_dice += dice_score.item()
            
    return total_dice / len(loader)

if __name__ == "__main__":
    print(DEVICE)
    train()