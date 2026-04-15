import os
import sys
from datetime import datetime
import torch
import yaml
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader
import wandb
from dataset import CloudSEN12Dataset

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def train():
    """
    Training pipeline for CloudMask semantic segmentation

    Flow:
        1. Load config.yaml - all hyperparameters and paths
        2. Build datasets - train (with augmentation) and val (clean, no augmentation)
        3. Wrap in DataLoaders - handles batching, shuffling, and parallel disk loading
        4. Build U-Net model with pretrained ResNet34 encoder (transfer learning)
        5. Define BCEWithLogitsLoss and Adam optimiser
        6. Training loop - for each epoch: forward -> loss -> backward -> update weights
        7. Validation loop - evaluate on unseen data after each epoch, no weight updates
        8. Log train/val loss per epoch to monitor for overfitting
        9. Save best checkpoint (cloudmask_best.pth) when val loss improves, and final checkpoint (cloudmask_last.pth) at end
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("results", exist_ok=True)
    sys.stdout = Tee(f"results/train_{timestamp}.txt")
    
    # Config
    config = yaml.safe_load(open("config.yaml"))
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Device
    # Move everything to GPU if available, fall back to CPU
    # ROCm implements AMD's HIP runtime as a drop-in replacement for CUDA, so ROCm/CUDU all surfaces through the same CUDA interface
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        # groups all runs under one project on the dashboard
        project="cloudmask-ml",
        # human-readable run name
        name=f"unet-resnet34-e{train_cfg['epochs']}-b{train_cfg['batch_size']}",
        # logs entire config.yaml as hyperparameters
        config=config                 
    )

    # Augmentation
    # Training only - val/test get no augmentation for honest evaluation
    # p= is probability of applying each transform per sample
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # 90-degree rotations only to preserve pixel grid
        A.RandomRotate90(p=0.5),
        # Mild radiometric variation - simulates atmospheric differences
        A.RandomBrightnessContrast(p=0.3)
    ])

    # Datasets
    # Train gets augmentation, val gets none
    train_dataset = CloudSEN12Dataset(
        data_dir=data_cfg["train_dir"],
        sensor_max_reflectance=data_cfg["sensor_max_reflectance"],
        transform=train_transform
    )
    val_dataset = CloudSEN12Dataset(
        data_dir=data_cfg["val_dir"],
        sensor_max_reflectance=data_cfg["sensor_max_reflectance"]
    )

    # DataLoaders
    # shuffle=True for train - prevents model learning order of samples
    # shuffle=False for val - order doesn't matter, consistency does
    # num_workers=4 - parallel workers loading data from disk while GPU trains
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=4)

    # Model
    # encoder_weights="imagenet" - start from pretrained weights, not random
    # This is called transfer learning - resnet34 already knows edges, textures, shapes
    # We fine-tune it for cloud detection rather than learning from scratch
    #resnet34 was pretrained on ImageNet (millions of photos), highly valuable with smaller datasets
    model = smp.Unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights="imagenet",
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["num_classes"]
    ).to(device)

    # Loss & Optimiser
    # BCEWithLogitsLoss - binary cross entropy, expects raw logits (no sigmoid on model output)
    # Adam - adaptive learning rate optimiser, standard choice for segmentation
    criterion = torch.nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    best_val_loss = float("inf")
    best_epoch = 0

    # Training Loop
    for epoch in range(train_cfg["epochs"]):

        # Training phase
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            # Unsqueeze adds channel dim: (B, H, W) -> (B, 1, H, W) to match model output shape
            # Mask comes out of the DataLoader as (B, H, W) but the model outputs (B, 1, H, W)
            # The shapes must match for the loss function. unsqueeze(1) inserts the missing channel dimension.
            masks = masks.unsqueeze(1).float().to(device)

            # clear gradients from last step
            optimiser.zero_grad()
            # forward pass
            outputs = model(images)
            # compare predictions to ground truth
            loss = criterion(outputs, masks)
            # backpropagation - compute gradients
            loss.backward()
            # update weights
            optimiser.step()

            train_loss += loss.item()

        # Validation phase
        # Disables dropout, freezes batchnorm
        model.eval()
        val_loss = 0.0

        # no_grad - don't track gradients, saves memory and compute
        with torch.no_grad():  
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.unsqueeze(1).float().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{train_cfg['epochs']} "
              f"| Train Loss: {avg_train:.4f} "
              f"| Val Loss: {avg_val:.4f}")

        # Logging - average loss per batch for comparability across different dataset sizes
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train,
            "val_loss": avg_val
        })

        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "models/cloudmask_best.pth")
            print(f"  Best model saved (val loss: {avg_val:.4f})")

    # Save best epoch and best value loss to W&B summary
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_epoch"] = best_epoch

    # Save final checkpoint regardless
    torch.save(model.state_dict(), "models/cloudmask_last.pth")
    wandb.finish()
    print("Training complete. Final model saved to models/cloudmask_last.pth")

if __name__ == "__main__":
    train()
