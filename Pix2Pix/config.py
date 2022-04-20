import os
import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS=10
DATASET_ROOT = '../dataset/pixTopixIT/maps'
CHECKPOINT_GEN = os.path.join(os.path.dirname(__file__), 'gen.pth.tar')
CHECKPOINT_DISC = os.path.join(os.path.dirname(__file__), 'disc.pth.tar')
LEARNING_RATE = 2e-4

L1_LAMBDA = 100

LOAD_MODEL = os.path.exists(CHECKPOINT_GEN) and os.path.exists(CHECKPOINT_DISC)
SAVE_MODEL = True
