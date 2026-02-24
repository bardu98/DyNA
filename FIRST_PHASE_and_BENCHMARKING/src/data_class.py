import torch
from torch.utils.data import Dataset
import numpy as np

class Nuc_Dataset(Dataset):
    def __init__(self, data, dim_embedding, truncate_last=False):
        """
        Args:
            data (list): Lista di dizionari dati.
            dim_embedding (int): Dimensione feature (es. 2560).
            truncate_last (bool): Se True, rimuove gli ultimi 3 token dagli embedding.
        """
        self.data = data
        self.dim_embedding = dim_embedding
        self.truncate_last = truncate_last

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Recupera gli embedding grezzi
        emb = sample['embedding']
        # Fallback sicuro se manca la chiave _rev
        emb_rev = sample.get('embedding_rev', emb) 

        # 2. CONVERSIONE NUMPY -> TORCH (FIX CRITICO)
        # Se sono numpy array, li convertiamo in tensori
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        if isinstance(emb_rev, np.ndarray):
            emb_rev = torch.from_numpy(emb_rev)

        # 3. Logica di taglio (truncate_last)
        if self.truncate_last:
            emb = emb[:-3]
            emb_rev = emb_rev[:-3]

        # 4. Clonazione sicura (Ora funziona perché sono Tensor)
        emb_tensor = emb.float().clone().detach()
        emb_rev_tensor = emb_rev.float().clone().detach()

        # Calcolo lunghezza dinamica
        seq_len = emb_tensor.shape[0]

        # 5. Gestione Label
        label = sample['label']
        # Gestione caso label scalare o numpy
        if not torch.is_tensor(label):
            # Se è numpy, convertilo prima in valore python o array
            if isinstance(label, np.ndarray):
                 label = torch.from_numpy(label).long()
            else:
                 label = torch.tensor(label, dtype=torch.int64)
        else:
            label = label.clone().detach().to(torch.int64)

        return {
            'sequence': sample['sequence'],
            'length': torch.tensor(seq_len, dtype=torch.int64),
            'embedding': emb_tensor,
            'embedding_rev': emb_rev_tensor,
            'label': label
        }