import torch
from torch.utils.data import Dataset

# class Nuc_Dataset(Dataset):
#     def __init__(self, data, dim_embedding, drop_last=False):
#         self.data = data
#         self.dim_embedding = dim_embedding
#         self.drop_last = drop_last

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]

#         if self.drop_last:
#             try:
            
#                 return {
#                 'sequence':sample['sequence'],
#                 'length': torch.tensor(144, dtype=torch.int64),
#                 # 'embedding_CLS': sample['embedding'][0].clone().detach(),
#                 # 'embedding': sample['embedding'][0:-3].clone().detach(),
#                 # 'embedding_CLS_rev': sample['embedding_rev'][0].clone().detach(),
#                 # 'embedding_rev': sample['embedding_rev'][0:-3].clone().detach(),
#                 'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
#             except:
#                 return {
#                 'sequence':sample['sequence'],

#                 'length': torch.tensor(144, dtype=torch.int64),
#                 # 'embedding_CLS': sample['embedding'][0].clone().detach(),
#                 # 'embedding': sample['embedding'][0:-3].clone().detach(),
#                 # 'embedding_CLS_rev': sample['embedding'][0].clone().detach(),
#                 # 'embedding_rev': sample['embedding'][0:-3].clone().detach(),
#                 'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
#         else:
                
#             try:
            
#                 return {
#                 'sequence':sample['sequence'],

#                 'length': torch.tensor(144, dtype=torch.int64),
#                 # 'embedding_CLS': sample['embedding'][0].clone().detach(),
#                 # 'embedding': sample['embedding'].clone().detach(),
#                 # 'embedding_CLS_rev': sample['embedding_rev'][0].clone().detach(),
#                 # 'embedding_rev': sample['embedding_rev'].clone().detach(),
#                 'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
#             except:
#                 return {
#                 'sequence':sample['sequence'],

#                 'length': torch.tensor(144, dtype=torch.int64),
#                 # 'embedding_CLS': sample['embedding'][0].clone().detach(),
#                 # 'embedding': sample['embedding'].clone().detach(),
#                 # 'embedding_CLS_rev': sample['embedding'][0].clone().detach(),
#                 # 'embedding_rev': sample['embedding'].clone().detach(),
#                 'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from Bio.Seq import Seq # Opzionale, se serve calcolare RC al volo

class Nuc_Dataset(Dataset):
    def __init__(self, data, max_length=37, rc_augmentation=True):
        """
        Args:
            data: Lista di dizionari contenenti 'sequence', 'label' e opzionalmente 'sequence_rev'
            max_length: Lunghezza massima per il padding/truncation del tokenizer
            rc_augmentation: Se True, include anche i token per la sequenza reverse complement
        """
        self.data = data
        self.max_length = max_length
        self.rc_augmentation = rc_augmentation
        
        # Inizializza il tokenizer qui
        self.tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
        )

    def __len__(self):
        return len(self.data)

    def tokenization(self, seq):
        return self.tokenizer(
            seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Forward Sequence
        tokens = self.tokenization(item['sequence'])
        
        result = {
            'input_ids': tokens['input_ids'].squeeze(0),      # [L]
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.float32)
        }

        # 2. Reverse Complement (Se richiesto)
        if self.rc_augmentation:
            # Controlla se 'sequence_rev' è già pre-calcolato nel dataset
            if 'sequence_rev' in item:
                seq_rc = item['sequence_rev']
            else:
                # Fallback: calcola al volo se manca (richiede Bio.Seq)
                seq_rc = str(Seq(item['sequence']).reverse_complement())

            tokens_rc = self.tokenization(seq_rc)
            result['input_ids_rc'] = tokens_rc['input_ids'].squeeze(0)
            result['attention_mask_rc'] = tokens_rc['attention_mask'].squeeze(0)
            
        return result