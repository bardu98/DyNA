#!/usr/bin/env python3
import argparse
import pickle
import os
from Bio import SeqIO
from Bio.Seq import Seq

def convert_to_pkl(input_file, output_pkl):
    print(f"[*] Lettura del file di input: {input_file}")
    
    data_list = []
    raw_sequences = []
    
    file_format = "raw"
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                file_format = "fasta"
            break 
            
    print(f"[*] Formato rilevato in automatico: {file_format.upper()}")
    
    if file_format == "fasta":
        try:
            records = list(SeqIO.parse(input_file, "fasta"))
            raw_sequences = [str(record.seq) for record in records]
        except Exception as e:
            print(f"[-] Errore durante la lettura del FASTA: {e}")
            return
    else:
        with open(input_file, 'r') as f:
            for line in f:
                seq = line.strip()
                if seq: # Ignora righe vuote
                    raw_sequences.append(seq)

    for seq_str in raw_sequences:
        seq_str = seq_str.upper() 
        
        seq_rev_str = str(Seq(seq_str).reverse_complement())
        
        entry = {
            'sequence': seq_str,
            'sequence_rev': seq_rev_str,
            'label': args.label
        }
        data_list.append(entry)
        
    print(f"[*] Elaborate {len(data_list)} sequenze con successo.")
    
    print(f"[*] Salvataggio nel file: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_list, f)
        
    print("[+] Conversione completata!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte un FASTA o un file txt riga-per-riga in .pkl (sequence, sequence_rev, label=1)")
    parser.add_argument("-i", "--input", required=True, type=str, help="Percorso del file di input (.txt o .fasta)")
    parser.add_argument("-o", "--output", required=True, type=str, help="Percorso del file Pickle (.pkl) di output")
    parser.add_argument("-l", "--label", required=True, type=int, help="true label")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[-] Errore: Il file di input '{args.input}' non esiste.")
        sys.exit(1)
        
    convert_to_pkl(args.input, args.output)