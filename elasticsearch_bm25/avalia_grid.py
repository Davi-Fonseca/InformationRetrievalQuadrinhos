"""
Gera run files para todas as combinações do Grid Search (incluindo Expansão de Query) e avalia com trec_eval.

Uso:
    uv run python avalia_grid.py
"""

import itertools
import os
import subprocess
from pathlib import Path

# Importações do NLTK para Expansão Global de Query
import nltk
from elasticsearch import Elasticsearch
from nltk.corpus import wordnet

# Garante o download do WordNet silenciosamente se não estiver disponível
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

from src.indexa import connect_elasticsearch, get_embedding_model
from src.busca import get_ltr_model
import numpy as np
import xgboost as xgb

# ── Configurações ────────────────────────────────────────────────────────────
QUERIES_PATH  = "datasets/queries_teste.tsv"
QRELS_PATH    = "datasets/qrels_teste.txt"
RUNS_DIR      = "datasets/runs"       # Pasta para organizar os 36 run files
RESULTS_PER_QUERY = 100               # Padrão top-100 para avaliação de RI
# ─────────────────────────────────────────────────────────────────────────────


def expandir_query(query: str) -> str:
    """Aplica Expansão Global de Query usando sinônimos do WordNet (NLTK)."""
    palavras = query.strip().split()
    termos_expandidos = []
    vistos = set()

    for palavra in palavras:
        palavra_lower = palavra.lower()
        if palavra_lower not in vistos:
            vistos.add(palavra_lower)
            termos_expandidos.append(palavra)

        # Busca sinônimos no WordNet
        syns = wordnet.synsets(palavra_lower)
        contador_sinonimos = 0
        for syn in syns:
            for lemma in syn.lemmas():
                sinonimo = lemma.name().lower().replace("_", " ")
                # Filtros para evitar "Query Drift" (desvio do assunto):
                # Apenas palavras únicas, sem duplicatas e que não sejam a própria palavra original
                if " " not in sinonimo and sinonimo not in vistos:
                    vistos.add(sinonimo)
                    termos_expandidos.append(sinonimo)
                    contador_sinonimos += 1
                    if contador_sinonimos >= 2:  # Limita a 2 sinônimos por termo original
                        break
            if contador_sinonimos >= 2:
                break

    return " ".join(termos_expandidos)


def buscar_docs(es: Elasticsearch, query: str, index_name: str, sim: str, qe: bool, size: int = 100) -> list[tuple[str, float]]:
    """Busca documentos aplicando a expansão de query na entrada se ativado."""
    if qe:
        query = expandir_query(query)

    if sim == "semantic":
        query_vector = get_embedding_model().encode(query).tolist()
        res = es.search(
            index="hqs_semantic",
            body={
                "knn": {
                    "query_vector": query_vector,
                    "field": "embedding",
                    "k": size,
                    "num_candidates": size * 2,
                }
            },
            size=size,
        )
        return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]

    elif sim == "hybrid":
        query_vector = get_embedding_model().encode(query).tolist()
        k_rrf = 60
        bm25_res = es.search(
            index="hqs_semantic",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["issue_title^2", "issue_description", "comic_name^3"],
                        "type": "best_fields",
                    }
                }
            },
            size=size,
        )
        knn_res = es.search(
            index="hqs_semantic",
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": size,
                    "num_candidates": size * 2,
                }
            },
            size=size,
        )
        scores: dict[str, float] = {}
        for rank, hit in enumerate(bm25_res["hits"]["hits"]):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
        for rank, hit in enumerate(knn_res["hits"]["hits"]):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:size]

    elif sim == "bm25+ltr":
        # 1. Recupera etapa 1 (O modelo clássico atual do grid)
        bm25_res = es.search(
            index=index_name,
            body={"query": {"multi_match": {"query": query, "fields": ["issue_title^4", "issue_description^2", "comic_name^5"], "type": "best_fields"}}},
            size=size,
        )
        bm25_results = [(h["_id"], h["_score"]) for h in bm25_res["hits"]["hits"]]
        
        # 2. Recupera Semântico
        query_vector = get_embedding_model().encode(query).tolist()
        sem_res = es.search(
            index="hqs_semantic",
            body={"knn": {"field": "embedding", "query_vector": query_vector, "k": size, "num_candidates": size * 2}},
            size=size,
        )
        sem_results = [(h["_id"], h["_score"]) for h in sem_res["hits"]["hits"]]
        
        # 3. Recupera scores isolados do campo
        def field_scores(field):
            r = es.search(index=index_name, body={"query": {"match": {field: query}}}, size=size)
            return {h["_id"]: h["_score"] for h in r["hits"]["hits"]}
            
        title_scores  = field_scores("issue_title")
        comic_scores  = field_scores("comic_name")
        desc_scores   = field_scores("issue_description")

        # 4. Constrói features
        bm25_map = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(bm25_results)}
        sem_map  = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(sem_results)}
        candidates = list(set(bm25_map) | set(sem_map))

        rows = []
        for doc_id in candidates:
            bs, br = bm25_map.get(doc_id, (0.0, 0))
            ss, sr = sem_map.get(doc_id,  (0.0, 0))
            rows.append([bs, br, ss, sr, title_scores.get(doc_id, 0.0), comic_scores.get(doc_id, 0.0), desc_scores.get(doc_id, 0.0), abs(br - sr)])

        if not rows:
            return []

        # 5. Predição LTR
        X = np.array(rows, dtype=np.float32)
        features_names = ["bm25_score", "bm25_rank", "semantic_score", "semantic_rank", "title_bm25_score", "comic_name_bm25_score", "description_bm25_score", "rank_diff"]
        ltr_scores = get_ltr_model().predict(xgb.DMatrix(X, feature_names=features_names))

        ranked = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)
        return ranked[:size]

    else:
        res = es.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["issue_title^2", "issue_description", "comic_name^3"],
                        "type": "best_fields",
                    }
                }
            },
            size=size,
        )
        return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]


def carregar_queries() -> list[dict]:
    """Lê e armazena as queries do arquivo TSV."""
    queries = []
    if not os.path.exists(QUERIES_PATH):
        raise FileNotFoundError(f"Arquivo de queries não encontrado em {QUERIES_PATH}")
        
    with open(QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split("\t")
            if len(partes) >= 2:
                queries.append({"qid": partes[0], "query": partes[1]})
    return queries


def gerar_run_file(es, queries: list[dict], sw: bool, proc: str, sim: str, qe: bool) -> Path:
    """Gera um run file TREC específico para uma configuração do grid."""
    sw_str = "yes" if sw else "no"
    qe_str = "expanded" if qe else "original"
    
    # Nome do índice do ES existente (GQE não altera o nome do índice físico)
    base_sim = "bm25" if sim == "bm25+ltr" else sim
    index_name = f"hqs_sw_{sw_str}_proc_{proc}_sim_{base_sim}"
    
    # Identificador único do sistema no arquivo TREC
    system_name = f"grid_{sw_str}_{proc}_{sim}_{qe_str}"
    
    run_file_name = f"run_sw_{sw_str}_proc_{proc}_sim_{sim}_qe_{qe_str}.txt"
    output_path = Path(RUNS_DIR) / run_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for q in queries:
            qid = q["qid"]
            query = q["query"]

            try:
                resultados = buscar_docs(es, query, index_name, sim, qe, RESULTS_PER_QUERY)
            except Exception as e:
                print(f"Erro ao buscar no índice {index_name}: {e}")
                return None

            for rank, (doc_id, score) in enumerate(resultados, start=1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{system_name}\n")

    return output_path


def parse_trec_eval_output(stdout_text: str) -> dict:
    """Extrai as métricas de interesse do output de texto do trec_eval."""
    metrics = {"map": 0.0, "P_10": 0.0, "ndcg_cut_10": 0.0, "recall_100": 0.0, "recip_rank": 0.0}
    for line in stdout_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            metric_name = parts[0]
            metric_value = parts[2]
            if metric_name in metrics:
                metrics[metric_name] = float(metric_value)
    return metrics


def avaliar_config(qrels: str, run_path: Path) -> dict:
    """Executa o trec_eval para o run file gerado e retorna as métricas."""
    cmd = ["trec_eval", "-m", "map", "-m", "P.10", "-m", "ndcg_cut.10", "-m", "recall.100", "-m", "recip_rank", qrels, str(run_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_trec_eval_output(result.stdout)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


def main():
    es = connect_elasticsearch()

    # print(expandir_query("morbius the living vampire blade battle"))  # Teste rápido da função de expansão de query
    # exit(0)

    print("1. Carregando queries...")
    try:
        queries = carregar_queries()
        print(f"✓ {len(queries)} queries carregadas com sucesso.")
    except Exception as e:
        print(f"ERRO: {e}")
        return

    # Hiperparâmetros do Grid Search expandidos
    sw_options = [True, False]
    proc_options = ["none", "stemming", "lemmatization"]
    sim_options = ["bm25", "jelinek_mercer", "dirichlet", "vsm", "semantic", "hybrid", "bm25+ltr"]
    qe_options = [True, False]  # Nova dimensão: Query Expansion ligado/desligado

    combinations_raw = list(itertools.product(sw_options, proc_options, sim_options, qe_options))
    combinations = []
    
    for sw, proc, sim, qe in combinations_raw:
        if sim in ["semantic", "hybrid"]:
            # Modelos densos usam seu próprio tokenizador. Variar SW e Proc no índice não afeta a busca semântica.
            # Adicionamos apenas uma configuração base para evitar testes redundantes.
            if sw is False and proc == "none":
                combinations.append((sw, proc, sim, qe))
        else:
            combinations.append((sw, proc, sim, qe))

    print(f"\n2. Iniciando avaliação de {len(combinations)} combinações do Grid Search...")

    results_grid = []
    trec_eval_available = True

    for idx, (sw, proc, sim, qe) in enumerate(combinations, start=1):
        sw_label = "Removed" if sw else "Unchanged"
        qe_label = "Expanded" if qe else "Original"
        
        print(f"[{idx}/{len(combinations)}] Stopwords: {sw_label} | Proc: {proc} | Sim: {sim} | QE: {qe_label}")
        
        run_path = gerar_run_file(es, queries, sw, proc, sim, qe)
        
        if run_path and trec_eval_available:
            metrics = avaliar_config(QRELS_PATH, run_path)
            if metrics is not None:
                results_grid.append({
                    "sw": sw_label,
                    "proc": proc,
                    "sim": sim,
                    "qe": qe_label,
                    "map": metrics["map"],
                    "p10": metrics["P_10"],
                    "ndcg10": metrics["ndcg_cut_10"],
                    "recall100": metrics.get("recall_100", 0.0),
                    "mrr": metrics.get("recip_rank", 0.0)
                })
            else:
                if idx == 1:
                    print("\n[Aviso] trec_eval não encontrado ou qrels ausente. Apenas os arquivos de run serão gerados.")
                    trec_eval_available = False

    # 3. Exibe o relatório final consolidado com a nova coluna
    if results_grid:
        results_grid.sort(key=lambda x: x["ndcg10"], reverse=True)

        print("\n" + "═" * 122)
        print(f"{'RANK':<5} | {'STOPWORDS':<11} | {'PROCESSAMENTO':<15} | {'SIMILARIDADE':<15} | {'EXPANSÃO':<12} | {'MAP':<8} | {'P@10':<8} | {'NDCG@10':<8} | {'REC@100':<8} | {'MRR':<8}")
        print("─" * 122)
        for rank, res in enumerate(results_grid, start=1):
            print(f"{rank:<5} | {res['sw']:<11} | {res['proc']:<15} | {res['sim']:<15} | {res['qe']:<12} | {res['map']:.4f} | {res['p10']:.4f} | {res['ndcg10']:.4f} | {res['recall100']:.4f} | {res['mrr']:.4f}")
        print("═" * 122)
        print(f"✓ Todos os logs detalhados de run salvos na pasta: {RUNS_DIR}\n")
    else:
        print(f"\n✓ Execução concluída. Os 36 arquivos de run foram gerados e salvos em '{RUNS_DIR}'.")


if __name__ == "__main__":
    main()
