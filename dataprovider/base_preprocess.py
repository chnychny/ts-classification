from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class BaseSTFTDataset(Dataset):
    """기본 STFT 데이터셋 클래스"""
    def __init__(self, X, y, normalize='minmax'):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
        
        if normalize == 'std':
            self.X = (self.X - self.X.mean()) / (self.X.std() + 1e-8)
        elif normalize == 'minmax':
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min() + 1e-8)
        elif normalize == 'no':
            pass            
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class BaseTimeDataset(Dataset):
    def __init__(self, X, y, normalize='minmax'):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # [batch, channel, length]
        self.y = torch.LongTensor(y)
        
        if normalize == 'std':
            self.X = (self.X - self.X.mean()) / (self.X.std() + 1e-8)
        elif normalize == 'minmax':
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min() + 1e-8)
        elif normalize == 'no':
            pass
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

