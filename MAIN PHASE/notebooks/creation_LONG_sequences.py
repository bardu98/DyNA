import pandas as pd
import numpy as np
import requests
import gzip
import os
from Bio import SeqIO
from io import StringIO
import random
import matplotlib.pyplot as plt
import seaborn as sns
from intervaltree import IntervalTree 
import glob 

#################################################################################################
# GLOBAL CONFIG
#################################################################################################

CHROMS_TO_PROCESS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

#################################################################################################
# HELPER FUNCTIONS
#################################################################################################

def calc_gc(seq):
    """Calcola il contenuto GC di una sequenza."""
    if not isinstance(seq, str) or len(seq) == 0:
        return 0
    gc = seq.count('G') + seq.count('C')
    return (gc / len(seq)) * 100

#################################################################################################
# STEP 1: GENOME: HG19
#################################################################################################

def download_hg19_genome(output_dir="hg19_genome"):
    """Scarica il genoma umano hg19 da UCSC (TUTTI I CROMOSOMI STANDARD)"""
    print(f"Scaricando genoma hg19 da UCSC per {len(CHROMS_TO_PROCESS)} cromosomi...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/"
    genome_dict = {}
    
    for chrom in CHROMS_TO_PROCESS:
        file_name = f"{chrom}.fa.gz"
        file_path = os.path.join(output_dir, file_name)
        unzipped_file_path = file_path.replace('.gz', '')

        if os.path.exists(unzipped_file_path):
            print(f"✓ {chrom} (hg19) già scaricato")
            try:
                with open(unzipped_file_path, 'r') as f:
                    records = SeqIO.parse(f, "fasta")
                    for record in records:
                        genome_dict[chrom] = str(record.seq).upper()
                continue
            except Exception as e:
                print(f"✗ Errore nel caricamento di {unzipped_file_path}: {e}. Tento di riscaricarlo.")

        url = base_url + file_name
        print(f"Scaricando {chrom} (hg19)...")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                sequence_content = ""
                with gzip.open(file_path, 'rt') as f:
                    sequence_content = f.read()
                
                with StringIO(sequence_content) as f_in:
                    records = SeqIO.parse(f_in, "fasta")
                    for record in records:
                        seq_str = str(record.seq).upper() 
                        genome_dict[chrom] = seq_str
                        print(f"✓ {chrom} (hg19) scaricato: {len(seq_str):,} bp")
                
                with open(unzipped_file_path, 'w') as f_out:
                    f_out.write(sequence_content)
                
                os.remove(file_path)
                
        except Exception as e:
            print(f"✗ Errore scaricamento {chrom} (hg19): {e}")
            if os.path.exists(file_path):
                os.remove(file_path) 
            continue
    
    print(f"\n✓ Genoma hg19 pronto: {len(genome_dict)} cromosomi")
    return genome_dict

def load_hg19_genome(genome_dir="hg19_genome"):
    """Carica genoma hg19 già scaricato (TUTTI I CROMOSOMI STANDARD)"""
    print("Caricamento genoma hg19...")
    genome_dict = {}
    
    for chrom in CHROMS_TO_PROCESS:
        file_path = os.path.join(genome_dir, f"{chrom}.fa")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    records = SeqIO.parse(f, "fasta")
                    for record in records:
                        seq_str = str(record.seq).upper() 
                        genome_dict[chrom] = seq_str
                        print(f"✓ {chrom} (hg19) caricato: {len(seq_str):,} bp")
            except Exception as e:
                print(f"✗ Errore caricamento {file_path}: {e}")
        else:
            print(f"✗ {chrom} non trovato in {genome_dir}")
    
    if not genome_dict:
        print(f"✗ Nessun file genoma hg19 trovato. Prova a scaricarlo prima.")
    return genome_dict

#################################################################################################
# STEP 2: WIG FILES
#################################################################################################
def parse_wig_file(wig_file_path, max_regions=500000000):
    """Parsifica file WIG con dati di occupancy dei nucleosomi (TUTTI I CROMOSOMI)"""
    print(f"  Parsing file WIG: {os.path.basename(wig_file_path)}")
    
    regions = []
    current_chrom = None
    current_start = None
    current_step = None
    current_span = 1
    position = 0
    lines_processed = 0
    
    try: 
        opener = gzip.open if wig_file_path.endswith('.gz') else open
        
        with opener(wig_file_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('track') or line.startswith('browser'):
                    continue
                
                lines_processed += 1
                if lines_processed % 50000000 == 0:
                    print(f"    ... Processate {lines_processed:,} linee, trovate {len(regions):,} regioni...")
                if len(regions) >= max_regions:
                    break
                
                if line.startswith('fixedStep'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('chrom='): current_chrom = part.split('=')[1]
                        elif part.startswith('start='):
                            current_start = int(part.split('=')[1])
                            position = current_start
                        elif part.startswith('step='): current_step = int(part.split('=')[1])
                        elif part.startswith('span='): current_span = int(part.split('=')[1])
                
                elif line.startswith('variableStep'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('chrom='): current_chrom = part.split('=')[1]
                        elif part.startswith('span='): current_span = int(part.split('=')[1])
                    current_step = None 
                
                else:
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            chrom, start_str, end_str, value_str = parts[0], parts[1], parts[2], parts[3]
                            if chrom not in CHROMS_TO_PROCESS: continue
                            value = float(value_str)
                            regions.append({'chr': chrom, 'start': int(start_str), 'end': int(end_str), 'score': value})
                        
                        elif len(parts) == 2 and current_step is None:
                            if current_chrom not in CHROMS_TO_PROCESS: continue
                            pos, value_str = int(parts[0]), float(parts[1])
                            regions.append({'chr': current_chrom, 'start': pos, 'end': pos + current_span, 'score': float(value_str)})
                        
                        elif len(parts) == 1 and current_step is not None:
                            if current_chrom not in CHROMS_TO_PROCESS: continue
                            regions.append({'chr': current_chrom, 'start': position, 'end': position + current_span, 'score': float(parts[0])})
                            position += current_step
                    except Exception:
                        continue
            
        print(f"    ✓ Trovate {len(regions):,} regioni")
        return pd.DataFrame(regions)
    
    except Exception as e:
        print(f"    ✗ Errore parsing WIG: {e}")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])


#################################################################################################
# STEP 3: CREAZIONE DATASET CONTINUO (CON FILTRO "PERCENTILE E ALTALENANTE")
#################################################################################################

def create_continuous_dataset(wig_files, genome_dict, max_sequences_per_file=5000, sequence_length=2000, 
                              percentile_high=99.0, min_std_score=1.5):
    """
    Crea un dataset estraendo finestre da 2000bp e associando un array di probabilità a ogni bp.
    Calcola dinamicamente la soglia di picco in base al percentile indicato per gestire 
    WIG files con diverse scale di valori.
    """
    print("\n" + "="*60)
    print("CREAZIONE DATASET CONTINUO (Seq 2000bp + Array Scores)")
    print("="*60)
    print(f"Lunghezza sequenza: {sequence_length} bp")
    print(f"Percentile per picco massimo: {percentile_high}th percentile")
    print(f"Filtro di variabilità minima (std score): > {min_std_score}")
    
    all_data = []
    
    for idx, wig_file in enumerate(wig_files, 1):
        print(f"\n[{idx}/{len(wig_files)}] Processando: {os.path.basename(wig_file)}")
        
        df_wig = parse_wig_file(wig_file)
        if df_wig.empty or 'score' not in df_wig.columns: 
            continue
            
        # Calcolo dinamico della soglia di picco tramite percentile
        calculated_min_peak = np.percentile(df_wig['score'], percentile_high)
        print(f"  ★ Soglia di picco calcolata ({percentile_high}° percentile): {calculated_min_peak:.4f}")
        
        # Filtriamo solo le zone che hanno segnale significativo iniziale per formare il pool di ricerca
        df_signal = df_wig[df_wig['score'] > calculated_min_peak].copy()
        if df_signal.empty:
            print(f"  ⚠ Nessun segnale > {calculated_min_peak:.4f} trovato. Skip file.")
            continue
            
        # Creazione IntervalTree per query veloci base-per-base
        print("  Creazione IntervalTree per query veloci...")
        trees = {}
        for chrom, group in df_wig.groupby('chr'):
            tree = IntervalTree()
            for row in group.itertuples(index=False):
                tree.addi(row.start, row.end, row.score)
            trees[chrom] = tree
            
        # Campionamento
        print(f"  Estrazione di {max_sequences_per_file} regioni lunghe {sequence_length}bp...")
        # Sovracampioniamo massicciamente (*20) perché scarteremo molte sequenze "piatte"
        sample_size = min(len(df_signal), max_sequences_per_file * 20)
        df_sampled = df_signal.sample(n=sample_size)
        
        extracted_count = 0
        rejected_by_stats = 0
        
        for row in df_sampled.itertuples(index=False):
            if extracted_count >= max_sequences_per_file:
                break
                
            chrom = row.chr
            if chrom not in genome_dict or chrom not in trees:
                continue
                
            # Definiamo il centro e la finestra di 2000bp
            center = (row.start + row.end) // 2
            start = center - (sequence_length // 2)
            end = start + sequence_length
            
            # Controlli di validità della sequenza
            if start < 0 or end > len(genome_dict[chrom]):
                continue
                
            seq = genome_dict[chrom][start:end]
            if len(seq) != sequence_length or 'N' in seq:
                continue
                
            # Creazione array degli score
            # Inizializziamo a 1.0 (sfondo di default se non ci sono picchi)
            scores_array = np.ones(sequence_length, dtype=np.float32)
            
            # Troviamo tutti gli intervalli del WIG che cadono in questi 2000bp
            overlapping_intervals = trees[chrom].overlap(start, end)
            for interval in overlapping_intervals:
                rel_start = max(0, interval.begin - start)
                rel_end = min(sequence_length, interval.end - start)
                scores_array[rel_start:rel_end] = interval.data
                
            # === FILTRO STATISTICO DINAMICO ===
            current_max = np.max(scores_array)
            current_std = np.std(scores_array)
            
            # Lo scartiamo se non tocca il percentile target OPPURE se è troppo piatto (bassa deviazione standard)
            if current_max < calculated_min_peak or current_std < min_std_score:
                rejected_by_stats += 1
                continue
            # ============================================
                
            all_data.append({
                'chr': chrom,
                'start': start,
                'end': end,
                'sequenza': seq,
                'dyad_scores': scores_array.tolist()
            })
            
            extracted_count += 1
            if extracted_count % 1000 == 0:
                print(f"    ... estratte {extracted_count}/{max_sequences_per_file} sequenze")
                
        print(f"  ✓ Completato. Aggiunte {extracted_count} sequenze valide.")
        print(f"  (Scartate {rejected_by_stats} finestre per segnale troppo piatto o senza picchi)")

    if not all_data:
        print("⚠⚠ NESSUNA SEQUENZA TROVATA CHE RISPETTI I CRITERI STATISTICI.")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'sequenza', 'dyad_scores'])

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['sequenza']).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("DATASET COMPLETATO")
    print("="*60)
    print(f"Totale sequenze estratte: {len(df):,}")
    
    return df

#################################################################################################
#                                          MAIN                                             #
#################################################################################################

def main():
    print("="*60)
    print("ELABORAZIONE DATI NUCLEOSOMI (Sequenze 2000bp Dinamiche su Percentile)")
    print("="*60)

    base_dir = os.getcwd()
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "cd4t_data"))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("\n" + "="*60)
    wig_search_path = os.path.join(data_dir, "ActivatedNucleosomes-chr2.hg19.wig")
    wig_files = glob.glob(wig_search_path)
    
    if not wig_files:
        print(f"✗ Nessun file WIG trovato in {data_dir}.")
        return
    else:
        print(f"✓ Trovati {len(wig_files)} file WIG hg19")

    print("\n" + "="*60)
    genome_dir = os.path.join(data_dir, "hg19_genome") 
    check_fa_path = os.path.join(genome_dir, f"{CHROMS_TO_PROCESS[0]}.fa")
    
    if os.path.exists(genome_dir) and os.path.exists(check_fa_path):
        genome_dict = load_hg19_genome(genome_dir) 
    else:
        genome_dict = download_hg19_genome(genome_dir) 

    if not genome_dict:
        return

    # ESTRAZIONE DATASET CONTINUO (2000bp, basato su percentile)
    # Imposta qui il percentile che preferisci (es. 99.0 o 99.8) e la Std. 
    df_final = create_continuous_dataset(
        wig_files,  
        genome_dict,
        max_sequences_per_file=10000, 
        sequence_length=400,
        percentile_high=99.,   # Sostituisce il vecchio parametro assoluto "min_peak_score"
        min_std_score=1.5       # Deviazione standard > 1.5 per garantire fluttuazione
    )

    if df_final is None or df_final.empty:
        return

    # SALVATAGGIO
    csv_file = os.path.join(data_dir, "CD4T_h19_Act_2000bp_Continuous.csv")
    pkl_file = os.path.join(data_dir, "CD4T_h19_Act_2000bp_Continuous.pkl")
    
    try:
        df_final.to_pickle(pkl_file)
        df_final.to_csv(csv_file, index=False)
        print(f"\n✓ Dataset salvato in Pickle: {pkl_file}")
        print(f"✓ Dataset salvato in CSV: {csv_file}")
    except Exception as e:
        print(f"✗ Errore salvataggio dataset: {e}")
        return

    # STATISTICHE
    print("\n" + "="*60)
    print("STATISTICHE FINALI")
    print("="*60)
    
    df_final['gc_content'] = df_final['sequenza'].apply(calc_gc)
    print(f"\nGC Content Medio: {df_final['gc_content'].mean():.2f}%")

    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_final, x='gc_content', bins=50, kde=True, color="blue")
        plt.title('Distribuzione GC Content (Finestre 2000bp)') 
        plot_file = os.path.join(data_dir, "gc_content_2000bp.png")
        plt.savefig(plot_file)
        print(f"✓ Plot GC Content salvato: {plot_file}")
    except Exception as e:
        print(f"✗ Errore generazione plot: {e}")

if __name__ == "__main__":  
    main()