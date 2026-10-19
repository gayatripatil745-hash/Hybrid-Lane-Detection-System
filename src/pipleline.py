import cv2
import numpy as np
import torch
from src.config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.cv_preprocessor import ClassicalCVPreprocessor

def infer_frame(model, frame_bgr: np.ndarray):
    resized = cv2.resize(frame_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT))
    cv_prior = ClassicalCVPreprocessor.extract_prior(resized)
    rgb_norm = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    combined = np.dstack((rgb_norm, cv_prior[:, :, None]))
    tensor_in = torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_in)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)

    return resized, cv_prior, prob_map, binary_mask
