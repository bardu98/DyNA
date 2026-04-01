import pickle
import random
from Bio import Align
from tqdm import tqdm

def check_strict_leakage_and_print(train_pkl_path, test_pkl_path, threshold=0.80, sample_size=300):
    print("="*80)
    print(f"ANALISI DATA LEAKAGE - RIGORE BIOINFORMATICO (Soglia {threshold*100}%)")
    print("Allineamento: Semi-Globale (Penalità interne severe, End-Gaps liberi per shift)")
    print("="*80)
    
    # 1. Caricamento Dataset .pkl
    with open(train_pkl_path, 'rb') as f:
        data_train = pickle.load(f)
    with open(test_pkl_path, 'rb') as f:
        data_test = pickle.load(f)
    
    # 2. Estrazione liste tramite List Comprehension
    train_pos = [item['sequence'] for item in data_train if item['label'] == 1]
    train_neg = [item['sequence'] for item in data_train if item['label'] == 0]
    
    test_pos = [item['sequence'] for item in data_test if item['label'] == 1]
    test_neg = [item['sequence'] for item in data_test if item['label'] == 0]
    
    # 3. Configurazione rigorosa dell'Allineatore (Parametri tipo BLAST/CD-HIT per DNA)
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0               # +1 per ogni base identica
    aligner.mismatch_score = -2.0           # Penalità forte per mismatch
    aligner.internal_open_gap_score = -3.0  # Penalità forte per aprire un buco interno
    aligner.internal_extend_gap_score = -1.0# Penalità per estendere il buco
    aligner.end_open_gap_score = 0.0        # ZERO penalità ai bordi (gestisce gli shift del nucleosoma)
    aligner.end_extend_gap_score = 0.0
    
    # 4. Motore di calcolo e ispezione
    def esegui_controllo(train_list, test_list, nome_classe):
        actual_sample_size = min(sample_size, len(test_list))
        if actual_sample_size == 0:
            print(f"\nNessuna sequenza trovata per {nome_classe}.")
            return
            
        test_sample = random.sample(test_list, actual_sample_size)
        
        # Lista per salvare i match trovati e ispezionarli
        coppie_simili = []
        
        print(f"\nAvvio allineamento rigoroso per {nome_classe} (Campione: {actual_sample_size} seqs)")
        
        for test_seq in tqdm(test_sample, desc=f"Progresso {nome_classe}", unit="seq"):
            seq_length = len(test_seq)
            # La soglia minima di score per considerare l'80% di identità netta
            min_score_required = seq_length * threshold 
            
            for train_seq in train_list:
                # Calcolo del punteggio di allineamento
                score = aligner.score(test_seq, train_seq)
                
                if score >= min_score_required:
                    # Calcolo la percentuale di identità effettiva
                    identita_pct = (score / seq_length) * 100
                    coppie_simili.append((test_seq, train_seq, identita_pct))
                    break # Passiamo alla prossima sequenza di test
                    
        percentuale_totale = (len(coppie_simili) / actual_sample_size) * 100
        
        # --- STAMPA DEI RISULTATI E DELLE SEQUENZE ---
        print(f"\n[ RISULTATI {nome_classe} ]")
        print(f"Overlap rilevato: {percentuale_totale:.1f}% ({len(coppie_simili)} su {actual_sample_size})")
        
        # if coppie_simili:
        #     print(f"\n--- Esempi di sequenze in Leakage ({nome_classe}) ---")
        #     # Stampiamo fino a un massimo di 5 esempi per non inondare il terminale
        #     for idx, (t_seq, tr_seq, identita) in enumerate(coppie_simili[:5]):
        #         print(f"\nMatch #{idx+1} (Identità netta stimata: {identita:.1f}%)")
        #         print(f"Test  : {t_seq}")
        #         print(f"Train : {tr_seq}")
        #     if len(coppie_simili) > 5:
        #         print(f"... e altre {len(coppie_simili) - 5} sequenze omesse per brevità.")
        # else:
        #     print("Nessun leakage rilevato in questo campione.")
            
        print("-" * 60)

    # 5. Esecuzione per entrambe le classi
    esegui_controllo(train_pos, test_pos, "POSITIVI (Nucleosomi)")
    esegui_controllo(train_neg, test_neg, "NEGATIVI (Background/Linker)")

# --- ESECUZIONE ---
train_file = "../data/data_pkl/Lymphoblastoid_99_8_percentile.pkl"
test_file = "../data/data_pkl/CD4T_h19_Rest_tot_99_8_percentile.pkl"

# Ho settato sample_size a 100 per darti un risultato in pochi minuti. 
# Se vuoi testare TUTTO il dataset, metti sample_size=1000000 (ci vorrà molto tempo).
check_strict_leakage_and_print(train_file, test_file, threshold=0.80, sample_size=300)