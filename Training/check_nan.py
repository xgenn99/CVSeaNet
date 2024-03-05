import torch

def check_nan(x):
     if torch.isnan(x).any():
          raise ValueError("There is a nan")


