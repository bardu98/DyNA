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
import glob # Per cercare i file

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
    """
    Scarica il genoma umano hg19 da UCSC (TUTTI I CROMOSOMI STANDARD)
    """
    print(f"Scaricando genoma hg19 da UCSC per {len(CHROMS_TO_PROCESS)} cromosomi...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # URL base UCSC per hg19
    base_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/"
    
    ### MODIFICA: Processa tutti i cromosomi ###
    chromosomes = CHROMS_TO_PROCESS
    
    genome_dict = {}
    
    for chrom in chromosomes:
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
    
    print(f"\n✓ Genoma hg19 scaricato: {len(genome_dict)} cromosomi")
    return genome_dict

def load_hg19_genome(genome_dir="hg19_genome"):
    """
    Carica genoma hg19 già scaricato (TUTTI I CROMOSOMI STANDARD)
    """
    print("Caricamento genoma hg19...")
    
    genome_dict = {}
    
    chromosomes = CHROMS_TO_PROCESS
    
    for chrom in chromosomes:
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
        print(f"✗ Nessun file genoma hg19 trovato in {genome_dir}. Prova a scaricarlo prima.")
    else:
        print(f"\n✓ Genoma hg19 caricato: {len(genome_dict)} cromosomi")
    return genome_dict

#################################################################################################
# STEP 2: WIG FILES
#################################################################################################
def parse_wig_file(wig_file_path, max_regions=500000000):
    """
    Parsifica file WIG con dati di occupancy dei nucleosomi (TUTTI I CROMOSOMI)
    VERSIONE v3 (ROBUSTA): Gestisce fixedStep, variableStep, e bedGraph
    """
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
                    print(f"    ⚠ Raggiunto limite di {max_regions:,} regioni, stop parsing")
                    break
                
                if line.startswith('fixedStep'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('chrom='):
                            current_chrom = part.split('=')[1]
                        elif part.startswith('start='):
                            current_start = int(part.split('=')[1])
                            position = current_start
                        elif part.startswith('step='):
                            current_step = int(part.split('=')[1])
                        elif part.startswith('span='):
                            current_span = int(part.split('=')[1])
                
                elif line.startswith('variableStep'):
                    parts = line.split()
                    for part in parts:
                        if part.startswith('chrom='):
                            current_chrom = part.split('=')[1]
                        elif part.startswith('span='):
                            current_span = int(part.split('=')[1])
                    current_step = None 
                
                # --- GESTIONE DATA LINE ---
                else:
                    try:
                        parts = line.split()
                        
                        
                        if len(parts) >= 4:
                            chrom, start_str, end_str, value_str = parts[0], parts[1], parts[2], parts[3]
                            if chrom not in CHROMS_TO_PROCESS:
                                continue
                            value = float(value_str)
                            if value > 0: 
                                regions.append({
                                    'chr': chrom,
                                    'start': int(start_str),
                                    'end': int(end_str),
                                    'score': value
                                })
                        
                        elif len(parts) == 2 and current_step is None:
                            if current_chrom not in CHROMS_TO_PROCESS:
                                continue
                            pos, value_str = parts[0], parts[1]
                            pos = int(pos)
                            value = float(value_str)
                            if value > 0:
                                regions.append({
                                    'chr': current_chrom,
                                    'start': pos,
                                    'end': pos + current_span,
                                    'score': value
                                })
                        
                        elif len(parts) == 1 and current_step is not None:
                            if current_chrom not in CHROMS_TO_PROCESS:
                                continue
                            value = float(parts[0])
                            if value > 0:
                                regions.append({
                                    'chr': current_chrom,
                                    'start': position,
                                    'end': position + current_span,
                                    'score': value
                                })
                            position += current_step
                        

                    except Exception:
                        continue
            
        print(f"    ✓ Trovate {len(regions):,} regioni con segnale (Tutti i cromosomi)")
        return pd.DataFrame(regions)
    
    except Exception as e:
        print(f"    ✗ Errore parsing WIG: {e}")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])

#################################################################################################
# STEP 3: FUNCTIONS FOR OTHER ANNOTETIONS
#################################################################################################

def filter_regions_by_feature(df_regions, df_features):
    """
    Mantiene solo le regioni che si sovrappongono con i feature forniti.
    Se df_features è None, restituisce TUTTE le regioni.
    """
    print("  Filtraggio regioni per sovrapposizione con feature...")

    if df_features is None or df_features.empty:
        print("  ⚠ Nessuna annotazione feature fornita, ritorno tutte le regioni (nessun filtro).")
        return df_regions 

    try:
        df_features_by_chr = {
            chrom: df_features[df_features['chr'] == chrom][['start', 'end']].values
            for chrom in df_features['chr'].unique()
        }
    except KeyError:
        print("  Errore: df_features deve contenere le colonne 'chr', 'start', 'end'.")
        return pd.DataFrame(columns=df_regions.columns)
        
    keep = np.zeros(len(df_regions), dtype=bool)

    for i, row in enumerate(df_regions.itertuples(index=False)):
        chrom = getattr(row, 'chr', None)
        if chrom not in df_features_by_chr:
            continue
            
        s = getattr(row, 'start', None)
        e = getattr(row, 'end', None)
        
        if s is None or e is None:
            continue

        for f_start, f_end in df_features_by_chr[chrom]:
            if not (e < f_start or s > f_end):  
                keep[i] = True
                break 

    filtered_df = df_regions[keep]
    print(f"  ✓ {len(filtered_df):,}/{len(df_regions):,} regioni trovate nei feature forniti.")
    return filtered_df.reset_index(drop=True)

def identify_nucleosome_positions(df_wig, threshold_value=1.0, sequence_length=201):
    """
    Identifica le posizioni delle diadi dei nucleosomi (picchi)
    usando la logica 'Non-Maximum Suppression'.
    """
    print("  Identificazione posizioni nucleosomi (Peak Calling)...")
    
    if df_wig.empty or 'score' not in df_wig.columns:
        print("    ⚠ DataFrame WIG vuoto o 'score' mancante.")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])

    df_positive_candidates = df_wig[df_wig['score'] > threshold_value].copy()
    print(f"    Trovati {len(df_positive_candidates):,} candidati positivi (> {threshold_value:.2f})")

    if df_positive_candidates.empty:
        print("    ⚠ Nessun candidato positivo trovato.")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])
        
    df_sorted = df_positive_candidates.sort_values('score', ascending=False)
    
    true_dyad_rows = []
    claimed_tree = {} 
    
    min_separation = sequence_length  

    print(f"    Avvio Non-Maximum Suppression (min_sep = {min_separation} bp)...")
    
    for chrom in df_sorted['chr'].unique():
        if chrom in CHROMS_TO_PROCESS: 
            claimed_tree[chrom] = IntervalTree()

    for row in df_sorted.itertuples(index=False):
        
        if row.chr not in claimed_tree:
            continue
            
        if claimed_tree[row.chr].overlaps(row.start, row.end):
            continue 
            
        true_dyad_rows.append(row)
        
        buffer_start = row.start - (min_separation // 2)
        buffer_end = row.start + (min_separation // 2) + 1 
        claimed_tree[row.chr].addi(buffer_start, buffer_end)

    if not true_dyad_rows:
        print("    ⚠ Nessun picco positivo trovato dopo il peak calling.")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])
        
    print(f"    ✓ Identificati {len(true_dyad_rows):,} picchi unici (diadi)")

    df_true_dyads = pd.DataFrame(true_dyad_rows, columns=df_sorted.columns)

    df_true_dyads['center'] = (df_true_dyads['start'] + df_true_dyads['end']) // 2
    df_true_dyads['start'] = df_true_dyads['center'] - (sequence_length // 2)
    df_true_dyads['end'] = df_true_dyads['start'] + sequence_length
    
    return df_true_dyads[['chr', 'start', 'end', 'score']]

def identify_non_nucleosome_positions(df_wig, threshold_value=1.0, nucleosome_length=201):
    """
    Identifica posizioni NON-nucleosomi (BASSA occupazione)
    usando una SOGLIA ASSOLUTA (score == threshold)
    """
    print("  Identificazione posizioni NON-nucleosomi (BASSA occupazione)...")
    
    if df_wig.empty or 'score' not in df_wig.columns:
        print("    ⚠ DataFrame WIG vuoto o 'score' mancante.")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'score'])

    print(f"    Soglia segnale (Negativi): == {threshold_value:.2f}")
    
    df_nucleosomes = df_wig[df_wig['score'] == threshold_value].copy()
    
    df_nucleosomes['center'] = (df_nucleosomes['start'] + df_nucleosomes['end']) // 2
    df_nucleosomes['start'] = df_nucleosomes['center'] - nucleosome_length // 2
    df_nucleosomes['end'] = df_nucleosomes['start'] + nucleosome_length
    
    print(f"    ✓ Identificati {len(df_nucleosomes)} NON-nucleosomi (Negativi)")
    
    return df_nucleosomes[['chr', 'start', 'end', 'score']]

def extract_sequences_from_genome(df_regions, genome_dict, max_sequences=2000):
    """
    Estrae sequenze DNA dalle regioni specificate
    """
    print(f"    Estrazione sequenze da {len(df_regions)} regioni...")
    
    sequences = []
    chromosomes = [] 
    
    count = 0
    processed = 0
    df_shuffled = df_regions.sample(frac=1)  
    
    for idx, row in df_shuffled.iterrows():
        processed += 1
        if count >= max_sequences:
            print(f"      Raggiunto limite di {max_sequences} sequenze.")
            break
        
        chrom = row['chr']
        start = int(row['start'])
        end = int(row['end'])
        
        if chrom not in genome_dict:
            continue
        
        if start < 0 or end > len(genome_dict[chrom]):
            continue
        
        seq = genome_dict[chrom][start:end]
        
        if len(seq) != (end - start):
            continue
        
        if 'N' in seq:
            continue

        sequences.append(seq)
        chromosomes.append(chrom) 
        count += 1
        
        if count % 5000 == 0 and count > 0:
            print(f"      Estraendole... {count}/{max_sequences} sequenze valide trovate (processate {processed} regioni)")
    
    print(f"    ✓ Estratte {len(sequences)} sequenze")
    return sequences, chromosomes 

def extract_filtered_sequences_efficiently(df_regions, df_features, genome_dict, positive_regions_tree, max_sequences=2000, buffer_zone=201):
    """
    Combina filtro e estrazione in un unico passaggio efficiente.
    Mescola i candidati e si ferma non appena 'max_sequences' sono state trovate.
    ESCLUDE i negativi che si sovrappongono ai positivi (buffer_zone).
    """
    print(f"  Avvio estrazione efficiente per {max_sequences} sequenze da {len(df_regions):,} candidati...")
    
    sequences = []
    chromosomes = [] 
    
    df_features_by_chr = None
    if df_features is not None and not df_features.empty:
        print("  (Filtro feature ATTIVO)")
        try:
            df_features_by_chr = {
                chrom: df_features[df_features['chr'] == chrom][['start', 'end']].values
                for chrom in df_features['chr'].unique()
            }
        except KeyError:
            print("    Errore: df_features deve contenere le colonne 'chr', 'start', 'end'.")
            return [], []
    else:
        print("  (Filtro feature DISABILITATO)")

    df_regions_shuffled = df_regions.sample(frac=1)

    processed_count = 0
    excluded_by_buffer = 0
    
    for row in df_regions_shuffled.itertuples(index=False):
        processed_count += 1
        
        s = getattr(row, 'start', None)
        e = getattr(row, 'end', None)
        chrom = getattr(row, 'chr', None)
        
        if s is None or e is None or chrom is None or chrom not in CHROMS_TO_PROCESS:
            continue
            
        buffer_start = s - buffer_zone
        buffer_end = e + buffer_zone
        if chrom in positive_regions_tree and positive_regions_tree[chrom].overlaps(buffer_start, buffer_end):
            excluded_by_buffer += 1
            continue 
        
        if df_features_by_chr is not None:
            if chrom not in df_features_by_chr:
                continue 
                
            has_overlap = False
            for f_start, f_end in df_features_by_chr[chrom]:
                if not (e < f_start or s > f_end):
                    has_overlap = True
                    break
            
            if not has_overlap:
                continue 

        
        start = int(s)
        end = int(e)
        
        if chrom not in genome_dict:
            continue
            
        if start < 0 or end > len(genome_dict[chrom]):
            continue
        
        seq = genome_dict[chrom][start:end]
        
        if len(seq) != (end - start):
            continue
        
        if 'N' in seq:
            continue

        sequences.append(seq)
        chromosomes.append(chrom) 
        
        if len(sequences) >= max_sequences:
            print(f"    ✓ Raggiunto limite di {max_sequences} sequenze valide (processati {processed_count:,} candidati).")
            print(f"    (Scartati {excluded_by_buffer:,} candidati per vicinanza ai positivi)")
            break
            
    if len(sequences) < max_sequences:
        print(f"    ✓ Estrazione completata. Trovate {len(sequences)} sequenze valide (processati tutti i {len(df_regions):,} candidati).")
        print(f"    (Scartati {excluded_by_buffer:,} candidati per vicinanza ai positivi)")

    return sequences, chromosomes 



def create_nucleosome_dataset(wig_files, genome_dict,  
                              max_sequences_per_file=1000,
                              percentile_high=99.8, 
                              threshold_low=1.0,
                              max_regions_per_wig=100000,
                              df_features=None,
                              sequence_length=201):
    """
    Crea dataset finale combinando dati WIG e genoma.
    Positivi: score > percentile_high (calcolato sui dati)
    Negativi: score == threshold_low
    """
    print("\n" + "="*60)
    print("CREAZIONE DATASET NUCLEOSOMI (Positivi vs Negativi in-loco)")
    print("="*60)
    print(f"Lunghezza sequenza: {sequence_length} bp")
    print(f"Soglia ALTA (percentile): {percentile_high}%") # <--- Log aggiornato
    print(f"Soglia BASSA (valore fisso): == {threshold_low:.2f}")
    if df_features is not None:
        print(f"Filtro regioni: ATTIVO (usando {len(df_features)} feature)")
    else:
        print("Filtro regioni: DISABILITATO (intero genoma)")
    
    all_sequences = []
    all_labels = []
    all_chromosomes = [] 
    
    for idx, wig_file in enumerate(wig_files, 1):
        print(f"\n[{idx}/{len(wig_files)}] Processando: {os.path.basename(wig_file)}")
        
        df_wig = parse_wig_file(wig_file, max_regions=max_regions_per_wig)
        
        if df_wig is None or len(df_wig) == 0:
            print("  ⚠ Nessun dato trovato, skip")
            continue
            
        # -----------------------------------------------------------------
        # 1. STREAM POSITIVO (Label 1)
        # -----------------------------------------------------------------
        print("\n  --- Stream Positivo (Label 1) ---")
        
        # Calcolo della soglia effettiva basata sul percentile
        calculated_threshold_high = 0
        if not df_wig.empty and 'score' in df_wig.columns:
            calculated_threshold_high = np.percentile(df_wig['score'], percentile_high)
            print(f"  ★ SOGLIA CALCOLATA ({percentile_high}%): {calculated_threshold_high:.4f}")
        else:
            print(f"  ⚠ Impossibile calcolare percentile.")
            continue

        df_nucleosomes_high = identify_nucleosome_positions(
            df_wig,  
            threshold_value=calculated_threshold_high, # <-- Passa il valore calcolato
            sequence_length=sequence_length
        )
        
        df_positive_filtered = filter_regions_by_feature(df_nucleosomes_high, df_features)

        if df_positive_filtered.empty:
            print("  ⚠ Nessuna regione positiva trovata [dopo il filtro], skip file")
            continue
            
        sequences_pos, chroms_pos = extract_sequences_from_genome(
            df_positive_filtered,  
            genome_dict,  
            max_sequences=max_sequences_per_file  
        )
        
        print("    Creazione albero intervalli positivi per zona cuscinetto...")
        positive_tree = {}
        for chrom in df_nucleosomes_high['chr'].unique():
            if chrom in CHROMS_TO_PROCESS: 
                positive_tree[chrom] = IntervalTree()
                for row in df_nucleosomes_high[df_nucleosomes_high['chr'] == chrom].itertuples():
                    positive_tree[chrom].addi(row.start, row.end)

        # -----------------------------------------------------------------
        # 2. STREAM NEGATIVO (Label 0)
        # -----------------------------------------------------------------
        print("\n  --- Stream Negativo (Label 0) ---")
        
        n_to_find = len(sequences_pos) 
        if n_to_find == 0:
             print("  ⚠ Nessuna sequenza positiva trovata, skip negativi")
             continue
        
        # 2a. Trova TUTTI i candidati negativi (==1.0)
        df_nucleosomes_low = identify_non_nucleosome_positions(
            df_wig,  
            threshold_value=threshold_low,
            nucleosome_length=sequence_length
        )
        
        if df_nucleosomes_low.empty:
            print("  ⚠ Nessuna regione negativa (==1.0) trovata, skip negativi")
            continue
        
        # 2b. Estrazione negativi con filtro e albero
        sequences_neg, chroms_neg = extract_filtered_sequences_efficiently(
            df_nucleosomes_low,     
            df_features,            
            genome_dict,
            positive_tree,          
            max_sequences=n_to_find,
            buffer_zone=sequence_length 
        )
        
        if not sequences_neg:
             print("  ⚠ Nessuna sequenza negativa valida trovata (o tutte troppo vicine ai positivi).")
             continue
        
        # -----------------------------------------------------------------
        # 3. BILANCIAMENTO e AGGIUNTA AL DATASET
        # -----------------------------------------------------------------
        
        n_pos = len(sequences_pos)
        n_neg = len(sequences_neg)

        n_to_keep = min(n_pos, n_neg)
        
        if n_pos > n_to_keep:
            print(f"  Bilanciamento: Sottocampiono {n_pos} positivi a {n_to_keep}")
            data_pos = list(zip(sequences_pos, chroms_pos))
            sampled_data_pos = random.sample(data_pos, n_to_keep)
            sequences_pos_sampled, chroms_pos_sampled = zip(*sampled_data_pos)
            sequences_pos = list(sequences_pos_sampled)
            chroms_pos = list(chroms_pos_sampled)
        
        if n_neg > n_to_keep:
            print(f"  Bilanciamento: Sottocampiono {n_neg} negativi a {n_to_keep}")
            data_neg = list(zip(sequences_neg, chroms_neg))
            sampled_data_neg = random.sample(data_neg, n_to_keep)
            sequences_neg_sampled, chroms_neg_sampled = zip(*sampled_data_neg)
            sequences_neg = list(sequences_neg_sampled)
            chroms_neg = list(chroms_neg_sampled)

        all_sequences.extend(sequences_pos)
        all_labels.extend([1] * len(sequences_pos))
        all_chromosomes.extend(chroms_pos) 
        
        all_sequences.extend(sequences_neg)
        all_labels.extend([0] * len(sequences_neg))
        all_chromosomes.extend(chroms_neg) 
        
        print(f"\n  RISULTATO FILE: Aggiunti {len(sequences_pos)} positivi e {len(sequences_neg)} negativi (Bilanciati).")
        print(f"  ✓ Totale sequenze accumulate: {len(all_sequences):,}")

    
    if not all_sequences:
        print("⚠⚠ NESSUNA SEQUENZA TROVATA. Controlla i percorsi e i filtri.")
        return pd.DataFrame(columns=['sequenza', 'nucleosoma', 'chr'])

    # Crea DataFrame
    df = pd.DataFrame({
        'sequenza': all_sequences,
        'nucleosoma': all_labels,
        'chr': all_chromosomes 
    })
    
    df = df.drop_duplicates(subset=['sequenza']).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("DATASET COMPLETATO (Bilanciato per regione)")
    print("="*60)
    print(f"Totale sequenze: {len(df):,}")
    print(f"Nucleosomi (positivi): {df['nucleosoma'].sum():,}")
    print(f"Non-nucleosomi (negativi): {(1 - df['nucleosoma']).sum():,}")
    
    if len(df) > 0:
        print(f"Bilanciamento: {df['nucleosoma'].sum() / len(df) * 100:.1f}% positivi")
    
    return df

#################################################################################################
#                                               MAIN                                            #
#################################################################################################

def main():
    print("="*60)
    print("ELABORAZIONE DATI NUCLEOSOMI GSE36979 (HG19 - INTERO GENOMA)")
    print("="*60)

    base_dir = os.getcwd()

    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "cd4t_data"))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Creata directory dati: {data_dir}")

    print("\n" + "="*60)
    print(f"Ricerca file *_147.hg19.wig in {data_dir}")
    
    wig_search_path = os.path.join(data_dir, "ActivatedNucleosomes-chr*.hg19.wig")
    wig_files = glob.glob(wig_search_path)
    
    if not wig_files:
        print(f"✗ Nessun file WIG 147.hg19.wig trovato in {data_dir}.")
        print("  Assicurati che i file .wig convertiti siano in questa cartella.")
        return
    else:
        print(f"✓ Trovati {len(wig_files)} file WIG hg19:")
        for f in wig_files:
            print(f"  - {os.path.basename(f)}")

    print("\n" + "="*60)
    genome_dir = os.path.join(data_dir, "hg19_genome") 

    check_fa_path = os.path.join(genome_dir, f"{CHROMS_TO_PROCESS[0]}.fa")
    
    if os.path.exists(genome_dir) and os.path.exists(check_fa_path):
        print(f"Genoma hg19 (cartella {genome_dir}) già presente, caricamento...")
        genome_dict = load_hg19_genome(genome_dir) 
    else:
        print("Scaricamento genoma hg19 (tutti i cromosomi standard)...")
        genome_dict = download_hg19_genome(genome_dir) 

    if not genome_dict:
        print("✗ Impossibile caricare genoma hg19")
        return

    print("\n" + "="*60)
    df_features_to_use = None 
        
    # STEP 3: Crea dataset
    df_final = create_nucleosome_dataset(
        wig_files,  
        genome_dict,
        max_sequences_per_file=1000000, 
        percentile_high=99.8, # <--- PERCENTILE
        threshold_low=1.0,            
        max_regions_per_wig=5000000000,
        df_features=df_features_to_use, 
        sequence_length=201
    )

    if df_final is None or df_final.empty:
        print("✗ Creazione dataset fallita o dataset vuoto.")
        return

    # STEP 4: Salva risultati
    #output_file = os.path.join(data_dir, "Lymphoblastoid_99_8_percentile.csv") #
    output_file = os.path.join(data_dir, "CD4T_h19_Act_tot_99_8_percentile.csv")
    try:
        df_final.to_csv(output_file, index=False)
        print(f"\n✓ Dataset (hg19, Intero Genoma) salvato: {output_file}")
    except Exception as e:
        print(f"✗ Errore salvataggio dataset: {e}")
        return

    # STEP 5: Statistiche finali
    print("\n" + "="*60)
    print("STATISTICHE FINALI (HG19 - INTERO GENOMA)")
    print("="*60)
    print(df_final.head(10))
    
    print(f"\nDistribuzione lunghezze:")
    print(df_final['sequenza'].str.len().describe())

    # GC content
    df_final['gc_content'] = df_final['sequenza'].apply(calc_gc)
    print(f"\nGC content per classe:")
    print(df_final.groupby('nucleosoma')['gc_content'].describe())

    # Plot
    print("\nGenerazione plot GC content...")
    try:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_final, x='gc_content', hue='nucleosoma', fill=True, common_norm=False, palette="Set1")
        plt.title('Distribuzione GC Content (0=Bassa, 1=Alta Occupazione) - HG19 / Intero Genoma') 
        plot_file = os.path.join(data_dir, "gc_content_distribution_hg19_whole_genome.png")
        plt.savefig(plot_file)
        print(f"✓ Plot salvato: {plot_file}")
    except Exception as e:
        print(f"✗ Errore generazione plot: {e}")

    print("\n✓ ELABORAZIONE (HG19 - INTERO GENOMA) COMPLETATA!") 

if __name__ == "__main__":  
    main()