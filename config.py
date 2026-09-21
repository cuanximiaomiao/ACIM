"""Fixed configuration for the released reference implementation."""
import os
import torch
from dataclasses import dataclass

# Training
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 40
WEIGHT_DECAY = 1e-5
SEED = 42

# Architecture
REP_UNITS = 200
REP_LAYERS = 3
HIDDEN_UNITS = 100
HIDDEN_LAYERS = 2

# Interaction mask: M_0 is initialized from W_prior; alpha is the trainable scaling
# coefficient, initialized to DYNAMIC_SCALE_INIT.
DYNAMIC_SCALE_INIT = 0.1

# Loss weights
LAMBDA_RANK = 0.3
LAMBDA_RLEARNER = 1.0
LAMBDA_BAL = 1.0


@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    lr: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    patience: int = PATIENCE
    seed: int = SEED
    weight_decay: float = WEIGHT_DECAY
    backbone_type: str = "dragonnet"
    rep_units: int = REP_UNITS
    rep_layers: int = REP_LAYERS
    hyp_units: int = HIDDEN_UNITS
    hyp_layers: int = HIDDEN_LAYERS
    dynamic_scale_init: float = DYNAMIC_SCALE_INIT
    lambda_rank: float = LAMBDA_RANK
    lambda_rlearner: float = LAMBDA_RLEARNER
    lambda_bal: float = LAMBDA_BAL


def set_seed(seed=SEED):
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_results_dir(base_dir="./results"):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    return base_dir


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
