import pandas as pd
import os
import sys
import argparse
import torch
import pickle  
from transformers import AutoTokenizer, AutoModel
from Bio.Seq import Seq
from tqdm import tqdm 

"""
DNA Sequence Embedding Generator using Nucleotide Transformer 2.5B.

This script reads a CSV file containing DNA sequences, calculates their embeddings 
using the pre-trained Nucleotide Transformer 2.5B model, 
and saves the results to a pickle file.

Usage Examples:

    1. Standard run (Normal sequences):
       $ python data_processing_embedding.py ../data/dataset_name.csv result_name
       # -> Output saved to: ../data/result_name.pkl

    2. Run with Reverse Complement:
       $ python data_processing_embedding.py ../data/dataset_name.csv result_name --RC
       # -> Output saved to: ../data/result_name_RC.pkl
"""



sys.path.append(os.path.abspath('../src'))

# Setup Args
parser = argparse.ArgumentParser(description="Compute Embeddings")
parser.add_argument("data_df_path", type=str, help="dataframe path")
parser.add_argument("--RC", action="store_true", help="Passa questo flag se vuoi calcolare il Reverse Complement")
parser.add_argument("name_saved", type=str, help="The name of the final pickle")
args = parser.parse_args()

# Setup Device and Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "InstaDeepAI/nucleotide-transformer-2.5B-multi-species"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()


def MetaAI_embedding(dna_sequence, tokenizer, model, device):
    tokens = tokenizer(dna_sequence.upper(), return_tensors="pt")
    
    tokens = tokens.to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state
    
    return embedding.squeeze(0).cpu().numpy()

# Load Data
if not os.path.exists(args.data_df_path):
    raise FileNotFoundError(f"File non trovato: {args.data_df_path}")

df = pd.read_csv(args.data_df_path)

tqdm.pandas(desc="Processing Rows")

if args.RC:
    print("Computing Reverse Complement...")
    df['sequence'] = df['sequence'].apply(lambda x: str(Seq(x).reverse_complement()))

print("Computing Embeddings (via apply)...")

df['embedding'] = df.progress_apply(
    lambda x: MetaAI_embedding(x['sequence'], tokenizer, model, device), 
    axis=1
)

# Save the data
output_dir = '../data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

suffix = "_RC" if args.RC else ""
filename = f"{args.name_saved}{suffix}.pkl"
save_path = os.path.join(output_dir, filename)

df_to_dict = df[['sequence', 'label', 'embedding']].to_dict(orient='records')

print(f"Saving to {save_path}...")
with open(save_path, 'wb') as f:
    pickle.dump(df_to_dict, f)

print("Done.")



