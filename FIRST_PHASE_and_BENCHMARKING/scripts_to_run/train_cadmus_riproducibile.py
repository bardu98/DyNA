#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import sys
# Aggiunge la cartella src al path
sys.path.append(os.path.abspath('../src'))
# Ora puoi importare normalmente
from data_class import Nuc_Dataset
from utils import test_classification,training_validation_and_test_loop_classification


#from model import CadmusDNA,TransformerNuc_Cadmus


from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio.Seq import Seq
import random



import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc


# In[2]:


def training_validation_and_test_loop_classification(
    model, dataloader_train, dataloader_validation, dataloader_test,
    epochs=20, lr=0.001, patience=10, weight_decay=0, weigth_dict=None
):
    # Assicurati che il modello sia sul device desiderato
    device = next(model.parameters()).device

    criterion = nn.BCEWithLogitsLoss(weight=weigth_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mcc_list, val_mcc_list, test_mcc_list = [], [], []
    loss_train, loss_val, loss_test = [], [], []

    best_state_cpu = None  # salviamo lo state_dict su CPU per evitare copie GPU
    best_val_loss, best_epoch = float('inf'), 0

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        total_loss, batch_count = 0.0, 0
        all_probs, all_labels = [], []

        for batch in dataloader_train:
            optimizer.zero_grad()

            # output_model_from_batch_final deve restituire tensori sul device corretto
            output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)

            # assicurati che labels siano sul device
            labels = labels.float().to(device)

            # calcolo loss (training)
            loss = criterion(output, labels) + criterion(output_rc, labels)

            # if torch.isnan(loss) or loss.item() < 1e-8:
            #     # pulizia sicura prima di continuare
            #     del loss, output, output_rc, importance, importance_rc, labels
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Prendi probabilità e spostale su CPU come numpy (detach per rimuovere grafo)
            probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            # --- Pulizia per evitare leak ---
            # elimina riferimenti a tensori GPU pesanti
            # del output, output_rc, importance, importance_rc, labels, loss, probs
            # # libera memoria Python/GC
            # gc.collect()
            # # informa l'allocatore CUDA che può liberare memoria inutilizzata
            # torch.cuda.empty_cache()
            # opzionale: sincronizza per sicurezza in debug
            # torch.cuda.synchronize()

        # calcoli training
        train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        loss_train.append(train_loss)

        if len(all_probs) > 0:
            train_preds = (np.array(all_probs) > 0.5).astype(int)
            train_mcc = matthews_corrcoef(all_labels, train_preds)
        else:
            train_mcc = 0.0
        train_mcc_list.append(train_mcc)

        # === VALIDATION ===
        model.eval()
        val_total_loss, val_batches = 0.0, 0
        val_probs, val_labels = [], []

        with torch.no_grad():
            for batch in dataloader_validation:
                output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
                labels = labels.float().to(device)

                loss = criterion(output, labels) + criterion(output_rc, labels)

                val_total_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

                # # pulizia temporanei della validazione
                # del output, output_rc, importance, importance_rc, labels, loss, probs
                # gc.collect()
                # torch.cuda.empty_cache()

        val_loss = val_total_loss / val_batches if val_batches > 0 else 0.0
        loss_val.append(val_loss)

        if len(val_probs) > 0:
            val_preds = (np.array(val_probs) > 0.5).astype(int)
            val_mcc = matthews_corrcoef(val_labels, val_preds)
        else:
            val_mcc = 0.0
        val_mcc_list.append(val_mcc)

        # === TEST ===
        test_total_loss, test_batches = 0.0, 0
        test_probs, test_labels = [], []

        with torch.no_grad():
            for batch in dataloader_test:
                output, output_rc, importance, importance_rc, labels = output_model_from_batch_final(batch, model, device)
                labels = labels.float().to(device)

                loss = criterion(output, labels) + criterion(output_rc, labels)

                test_total_loss += loss.item()
                test_batches += 1

                probs = torch.sigmoid((output + output_rc) / 2).detach().cpu().numpy()
                test_probs.extend(probs.tolist())
                test_labels.extend(labels.detach().cpu().numpy().tolist())

                # # pulizia temporanei del test
                # del output, output_rc, importance, importance_rc, labels, loss, probs
                # gc.collect()
                # torch.cuda.empty_cache()

        test_loss = test_total_loss / test_batches if test_batches > 0 else 0.0
        loss_test.append(test_loss)

        if len(test_probs) > 0:
            test_preds = (np.array(test_probs) > 0.5).astype(int)
            test_mcc = matthews_corrcoef(test_labels, test_preds)
        else:
            test_mcc = 0.0
        test_mcc_list.append(test_mcc)

        # === LOGGING ===
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, MCC: {train_mcc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MCC: {val_mcc:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, MCC: {test_mcc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # === EARLY STOPPING e salvataggio del best model (solo state_dict su CPU) ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # salva lo state_dict con i tensori trasferiti su CPU => evita di tenere copie GPU in memoria
            best_state_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            final_test_probs = test_probs.copy() if len(test_probs) > 0 else []
            best_val_probs = val_probs.copy()
            best_true_val = val_labels.copy()

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # # Piccola pulizia di fine-epoca
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    # Se vuoi restituire un "modello", restituisci lo state_dict su CPU (più leggero)
    # se vuoi ricaricarlo in seguito:
    # model.load_state_dict(best_state_cpu)
    epoch_best = best_epoch + 1
    return (
        train_mcc_list, val_mcc_list, loss_train, loss_val,
        best_val_loss, best_state_cpu, epoch_best,
        {'label': val_probs}, {'label': val_labels},
        val_labels, val_probs, test_labels, final_test_probs, best_val_probs, best_true_val
    )



# In[3]:


class TransformerNuc_Cadmus(nn.Module):
    def __init__(self, input_dim=2560, num_heads=8, dropout_rate=0.0, 
                 f_activation=nn.ReLU()):
        super(TransformerNuc_Cadmus, self).__init__()

        self.transoformer = MyTransformer(embedding_dim=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, activation=f_activation)
        self.act = f_activation
        
        self.final_ffn = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
        )

        self.final_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 1),
        )

    def forward(self, seq):
        # # seq: (batch_size, seq_len, feature_dim) → no need to transpose
        out, attention_matrix = self.transoformer(seq)
        out = out[:, 0,:]
        out = self.final_linear(out)

        return torch.squeeze(out), attention_matrix


class CadmusDNA(nn.Module):
    
    def __init__(self, att_module, att_parameters, device=None):
        super().__init__()

        # Imposta il dispositivo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
        # # Carica modello e tokenizer una sola volta
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
        # )
        # self.model_LLM = AutoModel.from_pretrained(
        #     "InstaDeepAI/nucleotide-transformer-2.5B-multi-species",
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        # ).to(self.device)
        # self.model_LLM.eval()
    
        # Inizializza i moduli
        self.attention_model = att_module(**att_parameters).to(self.device)
        self.linear_output = nn.Linear(2, 1).to(self.device)
        

    # def batch_embeddings(self, sequences, batch_size=32):
    #     """Calcola gli embedding completi per una lista di sequenze in batch."""
    #     all_embeddings = []
    #     for i in tqdm(range(0, len(sequences), batch_size), disable=True):
    #         batch = sequences[i:i+batch_size]
    #         tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
    #         with torch.no_grad():
    #             outputs = self.model_LLM(**tokens)
    #             # Mantiene tutti gli hidden states (nessun pooling)
    #             batch_emb = outputs.last_hidden_state.cpu()
    #         all_embeddings.extend(batch_emb)
    #     return all_embeddings
        

    def forward(self, seqs):
        #print(seqs)
        #embedding_tot = torch.stack(self.batch_embeddings(seqs)).to(self.device)
        output_att, importance = self.attention_model(seqs)
        return output_att, importance



class MyTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate, activation=nn.ReLU()):
        super(MyTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.act = activation
        self.positional_encoding = SinusoidalPositionalEncoding(self.embedding_dim, max_len=73)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)

        self.pw_ffnn = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.embedding_dim)
        )

    def forward(self, seq):
        # seq: (batch_size, seq_len, embedding_dim)
        seq = self.positional_encoding(seq)
        attn_output, attention_matrix = self.multihead_attention(seq, seq, seq)
        attn_output = self.norm1(attn_output + seq)
        ffn_out = self.pw_ffnn(attn_output)
        out = self.norm2(attn_output + ffn_out)
        return out, attention_matrix


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=28):    

        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# In[4]:


def output_model_from_batch_final(batch, model, device,rc=True):

    '''Dato un modello pytorch e batch restituisce: output_modello, True labels'''

    embedding_tot = batch['embedding'].float().to(device)
    #embedding_tot = batch['sequence']

    if rc:
        embedding_tot_rc = batch['embedding_rev'].float().to(device) 
        #embedding_tot_rc = batch['sequence_rev']
    
    else:
        embedding_tot_rc = batch['embedding'].float().to(device) 
        #embedding_tot_rc = batch['sequence']
        

    labels = batch['label'].to(device)

    output, importance = model(embedding_tot)
    output_rc, importance_rc = model(embedding_tot_rc)

    return output, output_rc, importance, importance_rc, labels


# In[5]:


import torch
from torch.utils.data import Dataset

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
                # 'embedding_CLS': sample['embedding'][0].clone().detach(),
                'embedding': sample['embedding'][0:-3].clone().detach(),
                # 'embedding_CLS_rev': sample['embedding_rev'][0].clone().detach(),
                'embedding_rev': sample['embedding_rev'][0:-3].clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
            except:
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                # 'embedding_CLS': sample['embedding'][0].clone().detach(),
                'embedding': sample['embedding'][0:-3].clone().detach(),
                # 'embedding_CLS_rev': sample['embedding'][0].clone().detach(),
                'embedding_rev': sample['embedding'][0:-3].clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
        else:
                
            try:
            
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                # 'embedding_CLS': sample['embedding'][0].clone().detach(),
                'embedding': sample['embedding'],#.clone().detach(),
                # 'embedding_CLS_rev': sample['embedding_rev'][0].clone().detach(),
                'embedding_rev': sample['embedding_rev'],#.clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}
                
            except:
                return {
                'sequence':sample['sequence'],

                'length': torch.tensor(144, dtype=torch.int64),
                # 'embedding_CLS': sample['embedding'][0].clone().detach(),
                'embedding': sample['embedding'],#.clone().detach(),
                # 'embedding_CLS_rev': sample['embedding'][0].clone().detach(),
                'embedding_rev': sample['embedding'],#.clone().detach(),
                'label': torch.tensor(sample['label'], dtype=torch.int64) if not isinstance(sample['label'], torch.Tensor) else sample['label'].clone().detach().to(torch.int64),}



# In[ ]:





# In[6]:


#CARICAMENTO DATI


# In[7]:


# #First Dataset

# with open('../data/dataframe_nup_1/dataset_nup1_sapiens.pkl', 'rb') as f:  
#     dataset_sapiens = pickle.load(f)
# with open('../data/dataframe_nup_1/dataset_nup1_sapiens_reverse.pkl', 'rb') as f:
#     dataset_sapiens_rev = pickle.load(f)

# RC_augmentation=True

# if RC_augmentation:
#     for d1, d2 in zip(dataset_sapiens, dataset_sapiens_rev):
#         d1['embedding_rev'] = d2['embedding']  
#         #d1['sequence_rev'] = d2['sequence']  
        
# data = dataset_sapiens

# #conto le occorrenze per ogni classe
# labels = [entry['label'] for entry in data]
# label_counts = Counter(labels)
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")


# In[8]:


#First Dataset

with open('../data/dataset_nup1_sapiens.pkl', 'rb') as f:  
    dataset_sapiens = pickle.load(f)
with open('../data/dataset_nup1_sapiens_RC.pkl', 'rb') as f:
    dataset_sapiens_rev = pickle.load(f)

RC_augmentation=True

if RC_augmentation:
    for d1, d2 in zip(dataset_sapiens, dataset_sapiens_rev):
        d1['embedding_rev'] = d2['embedding']  
        #d1['sequence_rev'] = d2['sequence']  
        
data = dataset_sapiens

#conto le occorrenze per ogni classe
labels = [entry['label'] for entry in data]
label_counts = Counter(labels)
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")


# In[9]:


# #Second Dataset PR

# dataset_HS_PR = []

# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_0_10k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_10k_20k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_20k_30k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_30k_40k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_40k_50k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_50k_60k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_60k_70k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_70k_80k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_80k_90k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_promoter_90k_100k.pkl', 'rb') as f:
#     dataset_HS_PR += pickle.load(f)

# # # #quando avrò il RC
# # if RC_augmentation:
# #     for d1, d2 in zip(dataset_HS_PR, dataset_HS_PR_rev):
# #         d1['embedding_rev'] = d2['embedding']


# data = dataset_HS_PR 

# #conto le occorrenze per ogni classe
# labels = [entry['label'] for entry in data]
# label_counts = Counter(labels)
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")


# In[10]:


# #Second Dataset 5U
    
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_5u.pkl', 'rb') as f:
#     dataset_sapiens_5u = pickle.load(f)
# with open('../data/dataframe_nup_2/dataset_nup2_sapiens_5u_reverse.pkl', 'rb') as f:
#     dataset_sapiens_5u_rev = pickle.load(f)

# if RC_augmentation:
#     for d1, d2 in zip(dataset_sapiens_5u, dataset_sapiens_5u_rev):
#         d1['embedding_rev'] = d2['embedding'] 


# data = dataset_sapiens_5u

# #conto le occorrenze per ogni classe
# labels = [entry['label'] for entry in data]
# label_counts = Counter(labels)
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")


# In[11]:


#Second Dataset Largest C.

#..... Da fare 


# In[12]:


# #Third dataset

# #dataset_nup1_sapiens   dataset_nup1_elegans
# with open('../data/dataframe_nup_3/dataset_nup3_h38.pkl', 'rb') as f:
#     dataset_sapiens_h38 = pickle.load(f)
# with open('../data/dataframe_nup_3/dataset_nup3_h38_reverse.pkl', 'rb') as f:
#     dataset_sapiens_h38_rev = pickle.load(f)

# if RC_augmentation:
#     for d1, d2 in zip(dataset_sapiens_h38, dataset_sapiens_h38_rev):
#         d1['embedding_rev'] = d2['embedding'] 

# data = dataset_sapiens_h38

# #conto le occorrenze per ogni classe
# labels = [entry['label'] for entry in data]
# label_counts = Counter(labels)
# for label, count in label_counts.items():
#     print(f"Label {label}: {count} samples")


# In[13]:


def test_classification(model, dataloader_test, threshold=0.5):
    device = next(model.parameters()).device
    model.eval()

    val_labels, val_preds = [], []
    importance_list = []
    importance_rc_list = []
    with torch.no_grad():
        for batch in dataloader_test:
            output, output_rc, importance, importance_rc, labels= output_model_from_batch_final(batch, model, device)
            # labels = batch['label'].float().to(device)

            probs = torch.sigmoid((output+output_rc)/2).cpu().numpy()

            val_preds.extend(probs)
            val_labels.extend(labels.cpu().numpy())
            importance_list.append(importance)
            importance_rc_list.append(importance_rc)

    metrics = classification_metrics(val_labels, val_preds, threshold=threshold)
    return metrics, val_labels, val_preds, importance_list, importance_rc_list,val_preds

import os
import numpy as np
import pandas as pd  # Assicurati di importare pandas
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, recall_score, accuracy_score, 
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

def classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calcola: Sensitivity, Specificity, Accuracy, MCC, AUC, PR AUC,
    salva ROC e PR curve e salva le predizioni in un CSV.
    """

    # Converti in numpy array e appiattisci se necessario (ravel) per evitare errori di forma
    y_true = np.array(y_true).ravel()
    y_pred_probs = np.array(y_pred_probs).ravel()
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('final threshold', threshold)

    # Metriche base
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)

    # ---------- Creazione cartella ../data se non esiste ----------
    save_dir = "../data"
    os.makedirs(save_dir, exist_ok=True)

    # ==================== SALVATAGGIO CSV ====================
    # Creiamo un DataFrame con True, Probabilities e Predicted Class
    df_results = pd.DataFrame({
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,  # Molto utile per analisi post-hoc
        'y_pred': y_pred               # La classe finale basata sulla threshold
    })
    
    csv_path = os.path.join(save_dir, "predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"File CSV con predizioni salvato in {csv_path}")

    # ==================== ROC CURVE ====================
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(save_dir, "roc_AUC.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curve salvata in {roc_path}")

    # ==================== PR CURVE ====================
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {pr_auc:.3f})")
    plt.tight_layout()

    pr_path = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"PR curve salvata in {pr_path}")

    # Output metriche
    output = {
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "MCC": mcc,
        "AUC": auc,
        "PR AUC": pr_auc
    }

    print(output)
    return output


# In[14]:


#FINE CARICAMENTO DATI


# In[15]:


from sklearn.metrics import matthews_corrcoef


# In[16]:


def set_seed(seed=42):
    random.seed(seed)                          # Seed Python (modulo random)
    np.random.seed(seed)                       # Seed NumPy
    torch.manual_seed(seed)                    # Seed PyTorch CPU
    torch.cuda.manual_seed(seed)               # Seed PyTorch CUDA (una GPU)
    torch.cuda.manual_seed_all(seed)           # Seed PyTorch CUDA (tutte le GPU)

    torch.backends.cudnn.deterministic = True  # Usa algoritmi deterministici
    torch.backends.cudnn.benchmark = False     # Disabilita ottimizzazioni non deterministiche

    print(f"Seeds fixed with seed = {seed}")
set_seed()


# In[ ]:


# -----------------------------------------------------------------
# PARTE 1: IMPORTAZIONI E DEFINIZIONE 'OBJECTIVE' PER OPTUNA
# -----------------------------------------------------------------
import optuna
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
import copy # Importato per 'local_data'
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

# --- INIZIO DEFINIZIONE OBJECTIVE PER OPTUNA ---
#
# ASSICURATI CHE LE SEGUENTI VARIABILI/CLASSI SIANO DEFINITE PRIMA DI QUESTA CELLA:
#
# Variabili:
# - data (la lista di dizionari non filtrata)
# - label_counts (il dizionario con i conteggi delle classi)
#
# Classi e Funzioni:
# - Nuc_Dataset
# - CadmusDNA
# - TransformerNuc_Cadmus
# - training_validation_and_test_loop_classification
# - test_classification (usato nella Parte 3)
#
# -----------------------------------------------------------------


g = torch.Generator()
g.manual_seed(42) # Usa lo stesso seed

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def objective(trial):
    """
    Funzione obiettivo che ottimizza sul VALIDATION SET INTERNO (5%).
    Salva anche le metriche del TEST SET esterno su Optuna per analisi post-hoc.
    """
    
    # --- 1. Definizione Iperparametri ---
    hp_dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    hp_num_heads = trial.suggest_categorical('num_heads', [4, 5, 8]) 
    hp_batch_size = trial.suggest_categorical('batch_size', [32, 64])
    hp_lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    hp_weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5, log=True)
    hp_patience = trial.suggest_int('patience', 5, 15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_embedding = 2560
    
    # --- 2. Preparazione Dati ---
    local_data = [x for x in data if x['embedding'].shape[0] == 28]

    if not local_data:
        return -1.0 

    dataset = Nuc_Dataset(local_data, dim_embedding)
    labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
    
    if len(labels) == 0:
        return -1.0 

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- Inizializzazione Dizionari Metriche ---
    # Metriche per il Validation (usate per guidare Optuna)
    fold_metrics = {
        'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 
        'F1': [], 'AUROC': [], 'PR_AUC': []
    }
    # Metriche per il Test (solo per log e analisi)
    test_fold_metrics = {
        'Sn': [], 'Sp': [], 'ACC': [], 'MCC': [], 
        'F1': [], 'AUROC': [], 'PR_AUC': []
    }

    # --- 3. Ciclo K-Fold ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
        train_labels = labels[train_idx]
        
        # Split Interno (95% Train / 5% Internal Val)
        internal_train_idx, internal_val_idx = train_test_split(
            train_idx,
            test_size=0.05,
            stratify=train_labels,
            random_state=fold
        )

        internal_train_subset = Subset(dataset, internal_train_idx)
        internal_val_subset = Subset(dataset, internal_val_idx)
        
        dataloader_internal_train = DataLoader(internal_train_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        dataloader_internal_val = DataLoader(internal_val_subset, batch_size=hp_batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        
        # Test loader del fold
        test_subset_cv = Subset(dataset, val_idx) 
        dataloader_test_cv = DataLoader(test_subset_cv, batch_size=hp_batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
        # Setup Modello
        transf_parameters_att = {'input_dim': dim_embedding, 
                                 'dropout_rate': hp_dropout_rate, 
                                 'num_heads': hp_num_heads}
        
        if 0 in label_counts and 1 in label_counts:
             pos_weight_val = label_counts[0] / label_counts[1]
        else:
             pos_weight_val = 1.0
             
        weights_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        
        model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
        
        # --- Allenamento ---
        # NOTA: Qui usiamo l'unpacking corretto (con variabili temporanee per i dizionari)
        (train_mcc_list, val_mcc_list, loss_train, loss_val,
         best_val_loss, best_state_cpu, epoch_best,
         _dict_probs, _dict_labels, # Variabili dummy per i dizionari restituiti
         val_labels, val_probs, test_labels, final_test_probs, 
         best_val_probs, best_true_val) = training_validation_and_test_loop_classification(
            model_internal,
            dataloader_internal_train,
            dataloader_internal_val, 
            dataloader_test_cv,      
            epochs=200, # Imposta a 200 per l'esecuzione reale             
            lr=hp_lr,
            weight_decay=hp_weight_decay,
            patience=hp_patience
        )

        # --- CALCOLO METRICHE (Loop su Validation e Test) ---
        # Creiamo una lista di tuple per processare sia Validation che Test con la stessa logica
        evaluations = [
            (val_labels, val_probs, fold_metrics),          # Set 1: Validation
            (test_labels, final_test_probs, test_fold_metrics) # Set 2: Test
        ]

        for y_true_raw, y_probs_raw, target_dict in evaluations:
            if len(y_true_raw) > 0:
                y_true = np.array(y_true_raw)
                y_probs = np.array(y_probs_raw)
                y_pred = (y_probs > 0.5).astype(int)

                # Confusion Matrix
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                except ValueError:
                    tn, fp, fn, tp = 0, 0, 0, 0 

                # Calcoli
                sn = recall_score(y_true, y_pred, zero_division=0) 
                sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0      
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                try:
                    auroc = roc_auc_score(y_true, y_probs)
                    pr_auc = average_precision_score(y_true, y_probs)
                except ValueError:
                    auroc, pr_auc = 0.5, 0.0 

                # Accumulo
                target_dict['Sn'].append(sn)
                target_dict['Sp'].append(sp)
                target_dict['ACC'].append(acc)
                target_dict['MCC'].append(mcc)
                target_dict['F1'].append(f1)
                target_dict['AUROC'].append(auroc)
                target_dict['PR_AUC'].append(pr_auc)
            else:
                for k in target_dict: target_dict[k].append(-1.0)

        # --- Pruning su MCC (Solo Validation!) ---
        trial.report(np.mean(fold_metrics['MCC']), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # --- 4. Salvataggio User Attrs (Validation e Test) ---
    mean_val_mcc = np.mean(fold_metrics['MCC'])
    
    # Salva metriche Validation
    for metric_name, values in fold_metrics.items():
        trial.set_user_attr(f"internal_mean_{metric_name}", np.mean(values))
        trial.set_user_attr(f"internal_std_{metric_name}", np.std(values))

    # Salva metriche Test (NUOVO)
    for metric_name, values in test_fold_metrics.items():
        trial.set_user_attr(f"test_mean_{metric_name}", np.mean(values))
        trial.set_user_attr(f"test_std_{metric_name}", np.std(values))

    # Optuna ottimizza sempre sulla metrica di validazione interna
    return mean_val_mcc

# -----------------------------------------------------------------
# PARTE 2: ESECUZIONE DELLO STUDIO OPTUNA
# -----------------------------------------------------------------

print("--- AVVIO STUDIO OPTUNA ---")
# Devi creare un sampler "seminato"
seeded_sampler = TPESampler(seed=42) # Usa lo stesso seed


study = optuna.create_study(
    study_name='my_cv_optimization_cadmus_data1_RIPRODUCIBILE_03_02_26',  # Un nome per il tuo studio
    storage='sqlite:///my_study_cadmus_data1_RIPRODUCIBILE_03_02_26.db',  # Nome del file dove salvare
    load_if_exists=True,              # LA MAGIA: carica i risultati se il file esiste
    direction='maximize',
    sampler=seeded_sampler,  # <-- DI' A OPTUNA DI USARLO
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)



# n_trials=50 significa 50 * 5 = 250 training (costoso!)
# Imposta n_trials a un valore ragionevole (es. 20-50)
study.optimize(objective, n_trials=100) 

print("\n--- STUDIO OPTUNA COMPLETATO ---")
print(f"Miglior trial (Mean Internal Val MCC): {study.best_value}")
print("Iperparametri migliori:")
print(study.best_params)

# Salva i migliori parametri per la Parte 3
best_hyperparameters = study.best_params


# # -----------------------------------------------------------------
# # PARTE 3: ESECUZIONE FINALE CON I MIGLIORI IPERPARAMETRI
# # (Questo è il tuo codice originale, modificato per usare i best_params)
# # -----------------------------------------------------------------

# print("\n--- AVVIO CROSS-VALIDATION FINALE CON HP OTTIMIZZATI ---")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dim_embedding = 2560
# # Rieseguiamo il filtraggio sui dati originali
# data_filtered = [x for x in data if x['embedding'].shape[0] == 28] 

# #dataset = Nuc_Dataset(data_filtered, dim_embedding, drop_last=False)

# labels = np.array([data_filtered[i]['label'].item() for i in range(len(data_filtered))])

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# fold_results = []
# for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#     print(f"\n--- Fold {fold+1} ---")

#     train_labels = labels[train_idx]
    
#     internal_train_idx, internal_val_idx = train_test_split(
#         train_idx,
#         test_size=0.05,
#         stratify=train_labels,
#         random_state=fold
#     )

#     internal_train_subset = Subset(data_filtered, internal_train_idx)
#     internal_val_subset = Subset(data_filtered, internal_val_idx)
    
#     # --- USA HP OTTIMIZZATI ---
#     dataloader_internal_train = DataLoader(internal_train_subset, batch_size=best_hyperparameters['batch_size'], shuffle=True)
#     dataloader_internal_val = DataLoader(internal_val_subset, batch_size=best_hyperparameters['batch_size'], shuffle=True)

#     test_subset_cv = Subset(data_filtered, val_idx) # Corretto
#     dataloader_test_cv = DataLoader(test_subset_cv, batch_size=best_hyperparameters['batch_size'], shuffle=True)
    
#     # --- USA HP OTTIMIZZATI ---
#     #transf_parameters_cls = {'input_dim': dim_embedding, 'dropout_rate': 0.}
#     transf_parameters_att = {'input_dim': dim_embedding, 
#                              'dropout_rate': best_hyperparameters['dropout_rate'], 
#                              'num_heads': best_hyperparameters['num_heads']}
#     #transf_parameters_ohe = {'input_dim': dim_embedding}
    
#     # Calcolo pesi (codice originale)
#     class_weights = {label: len(data_filtered) / count for label, count in label_counts.items()}
#     max_weight = max(class_weights.values())
#     class_weights = {label: weight / max_weight for label, weight in class_weights.items()}
#     weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
#     labels_list = ['label']
#     weight_dict = {label: weights_tensor for label in labels_list}
#     print(weight_dict)
    
#     model_internal = CadmusDNA(TransformerNuc_Cadmus, transf_parameters_att, device)
    
#     # --- USA HP OTTIMIZZATI ---
#     # Allenamento finale: ora ci interessano le performance su dataloader_test_cv
#     _, _, _, _, best_val_acc, best_model, best_epoch, _, _, val_labels, val_probs, test_labels, test_probs = training_validation_and_test_loop_classification(
#         model_internal,
#         dataloader_internal_train,
#         dataloader_internal_val,
#         dataloader_test_cv, # Questo è il VERO test set del fold CV
#         epochs= 200,
#         lr=best_hyperparameters['lr'],
#         weight_decay=best_hyperparameters['weight_decay'],
#         patience=best_hyperparameters['patience']
#     )

#     # Salva i risultati del fold (valutati sul test set del fold CV)
#     fold_results.append({
#         'fold': fold + 1,
#         'test_metrics': test_classification(best_model, dataloader_test_cv, threshold=0.5)[0],
#         'best_epoch': best_epoch
#     })

# # Stampa i risultati finali medi (sul test set dei 5 fold)
# print("\n--- RISULTATI FINALI CV (su test folds) CON HP OTTIMIZZATI ---")
# for metrics in ['Sensitivity (Recall)',
#   'Specificity',
#   'Accuracy',
#   'MCC',
#   'AUC',
#   'PR AUC',]:
#     # Gestisce il caso in cui la metrica non sia presente
#     metric_values = [i['test_metrics'].get(metrics) for i in fold_results if i['test_metrics'].get(metrics) is not None]
#     if metric_values:
#         print(f"{metrics}: {np.mean(metric_values)}")
#     else:
#         print(f"{metrics}: N/A (Metrica non trovata)")


# In[ ]:


assert False


# In[ ]:




