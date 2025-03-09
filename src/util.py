import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Check for Apple Silicon GPU
        return "mps"
    else:
        return "cpu"

device = get_device()
