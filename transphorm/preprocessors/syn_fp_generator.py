import numpy as np
import torch

from transphorm.preprocessors import create_dataset

if __name__ == "__main__":
    NUM_SAMPLES = 1000
    PATH_TO_SAVE = "/Users/mds8301/Desktop/temp/synthetic_dataset.pt"
    d = create_dataset(NUM_SAMPLES)
    d_tens = torch.tensor(d)
    torch.save(d_tens, PATH_TO_SAVE)
