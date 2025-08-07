import torch
from torch.utils.data import DataLoader, Dataset

class DiabetesDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

dataset = DiabetesDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)