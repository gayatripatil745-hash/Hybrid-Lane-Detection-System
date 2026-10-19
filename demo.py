import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import DEVICE
from src.dataset import SyntheticLaneDataset
from src.model import HybridLaneSegNet
from src.pipeline import infer_frame

def run_demo():
    if not os.path.exists("lane_model_weights.pth"):
        print("Model weights not found. Run 'python train.py' first.")
        return

    model = HybridLaneSegNet(in_channels=4, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("lane_model_weights.pth", map_location=DEVICE))

    val_dataset = SyntheticLaneDataset(num_samples=1)
    sample_x, _ = val_dataset[0]
    raw_rgb = (sample_x[:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)

    resized, cv_prior, prob_map, binary_mask = infer_frame(model, raw_bgr)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Input Frame (RGB)")
    axes[1].imshow(cv_prior, cmap='gray')
    axes[1].set_title("2. CV Prior (Canny ROI)")
    axes[2].imshow(prob_map, cmap='inferno')
    axes[2].set_title("3. ML Probability Map")

    overlay = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).copy()
    overlay[binary_mask == 1] = [0, 255, 0]
    axes[3].imshow(overlay)
    axes[3].set_title("4. Lane Segmented Overlay")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("demo_output.png")
    print("--> Demo visualization saved as 'demo_output.png'.")

if __name__ == "__main__":
    run_demo()
