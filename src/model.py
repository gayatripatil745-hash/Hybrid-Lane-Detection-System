import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.cv_preprocessor import ClassicalCVPreprocessor

class SyntheticLaneDataset(Dataset):
    def __init__(self, num_samples: int = 500, img_size: tuple = (160, 320)):
        self.num_samples = num_samples
        self.h, self.w = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = np.full((self.h, self.w, 3), 45, dtype=np.uint8)
        noise = np.random.normal(0, 8, (self.h, self.w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        img[:int(self.h * 0.45), :] = [140, 110, 80]

        left_x_top = int(self.w * 0.42 + np.random.randint(-15, 15))
        left_x_bottom = int(self.w * 0.15 + np.random.randint(-20, 20))
        right_x_top = int(self.w * 0.58 + np.random.randint(-15, 15))
        right_x_bottom = int(self.w * 0.85 + np.random.randint(-20, 20))
        top_y = int(self.h * 0.50)

        pts_left = np.array([[left_x_bottom, self.h], [left_x_top, top_y]], np.int32)
        cv2.polylines(img, [pts_left], isClosed=False, color=(240, 240, 240), thickness=5)
        cv2.polylines(mask, [pts_left], isClosed=False, color=1, thickness=5)

        pts_right = np.array([[right_x_bottom, self.h], [right_x_top, top_y]], np.int32)
        cv2.polylines(img, [pts_right], isClosed=False, color=(50, 230, 240), thickness=5)
        cv2.polylines(mask, [pts_right], isClosed=False, color=1, thickness=5)

        cv_prior = ClassicalCVPreprocessor.extract_prior(img)
        combined = np.dstack((img.astype(np.float32) / 255.0, cv_prior[:, :, None]))
        tensor_x = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1)
        tensor_y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return tensor_x, tensor_y
