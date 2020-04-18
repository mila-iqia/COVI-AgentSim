import torch
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class PandemicDataset(Dataset):
    def __init__(self, filepath="output/output.pkl"):
        self.filepath = filepath
        self.data = pickle.load(open(filepath, 'rb'))
        self.num_days = len(self.data)
        self.num_people = len(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.data[int(idx/self.num_people)][idx%self.num_people]
        return example['observed'], example['unobserved']

dataset = PandemicDataset()
import pdb; pdb.set_trace()
x, y = dataset.__getitem__(101)
print(x)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
import pdb; pdb.set_trace()
print(dataloader.dataset[1100])