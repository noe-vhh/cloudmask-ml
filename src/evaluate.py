import torch
import yaml
from datetime import datetime
import sys
import os
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import CloudSEN12Dataset
from tqdm import tqdm
import wandb

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

def evaluate():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("results", exist_ok=True)
    sys.stdout = Tee(f"results/eval_run_{timestamp}.txt")

    # Load config and device
    config = yaml.safe_load(open("config.yaml"))
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from checkpoint
    model = smp.Unet(
        encoder_name=model_cfg["encoder"],
        # Don't re-download ImageNet weights, use own
        encoder_weights=None,        
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["num_classes"]
    ).to(device)

    # weights_only=True - PyTorch only loads the tensor weights from the file (avoids runtime warning)
    # The default False allows arbitrary Python objects to be deserialised, which is a security risk if you ever load a checkpoint from an untrusted source
    model.load_state_dict(torch.load("models/cloudmask_best.pth", map_location=device, weights_only=True))
    model.eval()
    print("Model loaded from models/cloudmask_best.pth")

    # Test dataset and loader
    test_dataset = CloudSEN12Dataset(
        data_dir=data_cfg["test_dir"],
        sensor_max_reflectance=data_cfg["sensor_max_reflectance"]
        # no transform - honest evaluation, no augmentation
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4
    )

    print(f"Test samples: {len(test_dataset)}")

    # Accumulate raw counts
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            # raw logits (B, 1, H, W)
            outputs = model(images)
            # squash to 0.0-1.0
            probs = torch.sigmoid(outputs)
            # threshold -> binary, drop channel dim -> (B, H, W)
            preds = (probs > 0.5).long().squeeze(1)
            # masks is already (B, H, W) from the DataLoader

            tp_total += ((preds == 1) & (masks == 1)).sum().item()
            fp_total += ((preds == 1) & (masks == 0)).sum().item()
            fn_total += ((preds == 0) & (masks == 1)).sum().item()
            tn_total += ((preds == 0) & (masks == 0)).sum().item()

    # Compute and print metrics
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall    = tp_total / (tp_total + fn_total + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp_total / (tp_total + fp_total + fn_total + 1e-8)
    accuracy  = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total + 1e-8)

    print("\nEvaluation Results (Test Set)")
    print(f"IoU:       {iou:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"\nRaw counts - TP: {tp_total}, FP: {fp_total}, FN: {fn_total}, TN: {tn_total}")

    wandb.init(project="cloudmask-ml", name="evaluate-test-set", job_type="eval")
    wandb.log({
        "test/iou": iou,
        "test/f1": f1,
        "test/precision": precision,
        "test/recall": recall,
        "test/accuracy": accuracy,
        "test/tp": tp_total,
        "test/fp": fp_total,
        "test/fn": fn_total,
        "test/tn": tn_total
    })
    wandb.finish()

if __name__ == "__main__":
    evaluate()
