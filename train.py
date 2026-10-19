import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, POS_WEIGHT
from src.dataset import SyntheticLaneDataset
from src.model import HybridLaneSegNet

def train():
    print(f"--> Initializing Training on compute device: {DEVICE}")
    train_dataset = SyntheticLaneDataset(num_samples=500)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridLaneSegNet(in_channels=4, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "lane_model_weights.pth")
    print("--> Model training complete. Weights saved to 'lane_model_weights.pth'.")

if __name__ == "__main__":
    train()
