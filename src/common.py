import os

import torch

device = torch.device("cuda")

data_folder = '../data/'
temp_folder = '../temp/'

eps = torch.tensor([1e-10], device=device)
zero = torch.tensor([1e-10], device=device)

if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)
