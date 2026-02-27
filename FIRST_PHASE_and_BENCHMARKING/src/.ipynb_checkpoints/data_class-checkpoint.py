import torch
from torch.utils.data import Dataset
import numpy as np


class Nuc_Dataset(Dataset):
    def __init__(self, data, dim_embedding, drop_last=False):
        self.data = data
        self.dim_embedding = dim_embedding
        self.drop_last = drop_last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.drop_last:
            try:
            
                return {
                'sequence':sample['sequence'],
                'length': torch.tensor(144, dtype=torch.int64),
                'embedding': sample['embedding'][0:-3].clone().detach(),
                'embedding_rev': sample['embedding_rev'][0:-3].clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
            except:
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                'embedding': sample['embedding'][0:-3].clone().detach(),
                'embedding_rev': sample['embedding'][0:-3].clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
        else:
                
            try:
            
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                'embedding': sample['embedding'],
                'embedding_rev': sample['embedding_rev'],
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
            except:
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                'embedding': sample['embedding'],
                'embedding_rev': sample['embedding'],
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
