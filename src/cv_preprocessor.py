import cv2
import numpy as np
from src.config import (
    CANNY_LOW_THRESH,
    CANNY_HIGH_THRESH,
    BILATERAL_DIAMETER,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE
)

class ClassicalCVPreprocessor:
    @staticmethod
    def extract_prior(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(
            gray, 
            d=BILATERAL_DIAMETER, 
            sigmaColor=BILATERAL_SIGMA_COLOR, 
            sigmaSpace=BILATERAL_SIGMA_SPACE
        )
        edges = cv2.Canny(blurred, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)

        roi_mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.40), int(h * 0.48)),
            (int(w * 0.60), int(h * 0.48)),
            (int(w * 0.95), h)
        ]], np.int32)
        cv2.fillPoly(roi_mask, polygon, 255)

        masked_edges = cv2.bitwise_and(edges, roi_mask)
        return (masked_edges.astype(np.float32) / 255.0)
