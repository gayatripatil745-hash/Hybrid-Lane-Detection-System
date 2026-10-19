import torch

# Compute hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dimensions
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320
NUM_CHANNELS = 4

# Training Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 12.0

# OpenCV Parameters
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50
