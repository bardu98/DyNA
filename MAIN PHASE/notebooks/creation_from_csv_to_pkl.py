import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('../src'))
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


df = pd.read_csv('../data/lymphoblasotid_data/Lymphoblastoid_99_8_percentile.csv')
df = df.rename(columns={'sequenza':'sequence', 'nucleosoma':'label'})

df_ones = df[df['label'] == 1]
df_zeros = df[df['label'] == 0]

df_zeros_sampled = df_zeros.sample(n=len(df_ones), random_state=42)

df_balanced = pd.concat([df_ones, df_zeros_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_balanced['label'].value_counts())

seqs = df["sequence"].str.upper().tolist()

df["embedding"] = 0   

df = df[['sequence', 'label', 'embedding']].to_dict(orient='records')


import pickle
with open('../data/Geo_dataset/CD4T_h19_Rest_tot_99_8_percentile.pkl', 'wb') as f:
    pickle.dump(df, f)