import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet3D
from losses import DiceLoss
from data_loader import NEG_DIR, POS_DIR, LungNoduleDataset
import torch.nn as nn

# configuration & MLOps
DEVICE = torch.device("cpu")
BATCH_SIZE = 8  # Optimized for 16GB VRAM
LEARNING_RATE = 1e-4
EPOCHS = 20

def train():
    # 1. Load the initial manifest
    full_dataset = LungNoduleDataset(csv_file="./scan_manifest.csv", pos_dir=POS_DIR, neg_dir=NEG_DIR)
    
    # --- START CLEAN-UP CODE ---
    print(f"Initial manifest size: {len(full_dataset.df)}")

    def file_exists(row):
        # Determine the correct cube folder
        folder = POS_DIR if row['type'] == 'positive' else NEG_DIR
        cube_path = os.path.join(folder, row['file'])

        # Check if cube exists
        if not os.path.exists(cube_path):
            return False
            
        # If positive, also check if the mask exists
        if row['type'] == 'positive':
            mask_path = os.path.join("LungVoxels/NoduleMasks", str(row['mask_file']))
            return os.path.exists(mask_path)
            
        return True

    # Filter the internal DataFrame of the dataset object
    full_dataset.df = full_dataset.df[full_dataset.df.apply(file_exists, axis=1)].reset_index(drop=True)
    print(f"Cleaned manifest size: {len(full_dataset.df)} (Filtered out missing files)")
    # --- END CLEAN-UP CODE ---

    # 2. train-test-Split (Now uses only validated files)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

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
        torch.save(model.state_dict(), "best_model.pth")

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