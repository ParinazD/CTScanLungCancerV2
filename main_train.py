import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet3D
from losses import DiceLoss
from data_loader import LungNoduleDataset

# --- Configuration & MLOps ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Optimized for 16GB VRAM
LEARNING_RATE = 1e-4
EPOCHS = 20

def train():
    # 1. Load Dataset using your manifest
    full_dataset = LungNoduleDataset(csv_file="scan_manifest.csv")
    
    # Split: 80% Train, 20% Validation (Evaluation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model, Loss, and Optimizer
    model = UNet3D().to(DEVICE)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for cubes, masks in train_loader:
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(cubes)
            loss = criterion(outputs, masks)
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
    """
    Evaluation function to calculate the Dice Coefficient on unseen data.
    """
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for cubes, masks in loader:
            cubes, masks = cubes.to(DEVICE), masks.to(DEVICE)
            preds = model(cubes)
            
            # Use 1 - DiceLoss to get the actual Dice Coefficient
            dice_score = 1 - DiceLoss()(preds, masks)
            total_dice += dice_score.item()
            
    return total_dice / len(loader)

if __name__ == "__main__":
    train()