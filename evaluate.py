import os
import torch
from torch.utils.data import DataLoader
from src.config import DEVICE, BATCH_SIZE
from src.dataset import SyntheticLaneDataset
from src.model import HybridLaneSegNet
from src.metrics import compute_metrics

def evaluate():
    if not os.path.exists("lane_model_weights.pth"):
        print("Model weights not found. Run 'python train.py' first.")
        return

    val_dataset = SyntheticLaneDataset(num_samples=100)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = HybridLaneSegNet(in_channels=4, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("lane_model_weights.pth", map_location=DEVICE))
    model.eval()

    all_targets, all_preds = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            preds = torch.sigmoid(model(imgs))
            preds_flat = (preds > 0.5).cpu().numpy().astype(int).flatten()
            targets_flat = masks.numpy().astype(int).flatten()
            all_preds.extend(preds_flat)
            all_targets.extend(targets_flat)

    results = compute_metrics(all_targets, all_preds)
    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    for k, v in results.items():
        print(f"{k:10s} : {v:.4f}")

if __name__ == "__main__":
    evaluate()
