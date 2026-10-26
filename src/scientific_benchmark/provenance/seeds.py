"""Seed CPU and optional torch generators; determinism is hardware dependent."""
import random
import numpy as np

def set_seed(seed, *, torch_enabled=False, deterministic=False):
    random.seed(seed); np.random.seed(seed)
    if torch_enabled:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark=False
