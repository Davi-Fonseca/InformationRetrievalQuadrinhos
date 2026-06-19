import os
import random
from collections import defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent / "datasets"
QUERIES_PATH = BASE_DIR / "queries.tsv"
QRELS_PATH = BASE_DIR / "qrels.txt"

# Output paths
TREINO_QUERIES = BASE_DIR / "queries_treino.tsv"
TREINO_QRELS = BASE_DIR / "qrels_treino.txt"

VAL_QUERIES = BASE_DIR / "queries_val.tsv"
VAL_QRELS = BASE_DIR / "qrels_val.txt"

TESTE_QUERIES = BASE_DIR / "queries_teste.tsv"
TESTE_QRELS = BASE_DIR / "qrels_teste.txt"

SEED = 42

def load_data():
    queries = {}
    if os.path.exists(QUERIES_PATH):
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
    
    qrels = defaultdict(list)
    if os.path.exists(QRELS_PATH):
        with open(QRELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    qid = parts[0]
                    qrels[qid].append(line.strip())
                    
    return queries, qrels

def write_splits(split_qids, queries, qrels, queries_out, qrels_out):
    # Write queries
    with open(queries_out, "w", encoding="utf-8") as f:
        for qid in split_qids:
            if qid in queries:
                f.write(f"{qid}\t{queries[qid]}\n")
                
    # Write qrels
    with open(qrels_out, "w", encoding="utf-8") as f:
        for qid in split_qids:
            if qid in qrels:
                for line in qrels[qid]:
                    f.write(line + "\n")

def main():
    print("Lendo datasets originais...")
    queries, qrels = load_data()
    
    # Precisamos apenas das queries que existem em AMBOS (têm qrels e texto)
    valid_qids = [qid for qid in queries.keys() if qid in qrels]
    print(f"Total de queries válidas (com texto e julgamento): {len(valid_qids)}")
    
    # Ordenar antes de embaralhar garante a reprodutibilidade em qualquer SO
    valid_qids.sort()
    
    # Embaralhar
    random.seed(SEED)
    random.shuffle(valid_qids)
    
    # Divisão 70 / 10 / 20
    n_total = len(valid_qids)
    n_treino = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    treino_qids = valid_qids[:n_treino]
    val_qids = valid_qids[n_treino : n_treino + n_val]
    teste_qids = valid_qids[n_treino + n_val:]
    
    print(f"Split criado com sucesso:")
    print(f" - Treino: {len(treino_qids)} queries")
    print(f" - Validação: {len(val_qids)} queries")
    print(f" - Teste: {len(teste_qids)} queries")
    
    # Escrever arquivos
    write_splits(treino_qids, queries, qrels, TREINO_QUERIES, TREINO_QRELS)
    write_splits(val_qids, queries, qrels, VAL_QUERIES, VAL_QRELS)
    write_splits(teste_qids, queries, qrels, TESTE_QUERIES, TESTE_QRELS)
    
    print("Arquivos salvos fisicamente na pasta 'datasets/'.")

if __name__ == "__main__":
    main()
