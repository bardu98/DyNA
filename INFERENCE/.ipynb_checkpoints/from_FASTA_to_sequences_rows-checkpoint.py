#!/usr/bin/env python3
import argparse
import os
import sys
from Bio import SeqIO

def fasta_to_raw_txt(input_fasta, output_txt):
    print(f"[*] Lettura del file FASTA: {input_fasta}")
    
    try:
        # SeqIO.parse legge in automatico gli header '>' e unisce le sequenze multiriga
        records = list(SeqIO.parse(input_fasta, "fasta"))
    except Exception as e:
        print(f"[-] Errore durante la lettura del FASTA: {e}")
        return

    print(f"[*] Trovate {len(records)} sequenze. Scrittura in corso...")
    
    # Scriviamo solo la sequenza pura, riga per riga
    with open(output_txt, 'w') as f_out:
        for record in records:
            f_out.write(str(record.seq).upper() + '\n')
            
    print(f"[+] Conversione completata! File salvato in: {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae solo le sequenze da un FASTA e le salva in un txt (una per riga)")
    parser.add_argument("-i", "--input", required=True, type=str, help="Percorso del file FASTA di input")
    parser.add_argument("-o", "--output", required=True, type=str, help="Percorso del file TXT di output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[-] Errore: Il file di input '{args.input}' non esiste.")
        sys.exit(1)
        
    fasta_to_raw_txt(args.input, args.output)