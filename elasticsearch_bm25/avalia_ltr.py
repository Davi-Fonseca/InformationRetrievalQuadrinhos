import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

QUERIES_PATH   = "datasets/queries.tsv"
QRELS_PATH     = "datasets/qrels.txt"
DATASET_PATH   = "datasets/dataset_ltr.csv"
RUN_LTR_PATH   = "datasets/run_ltr_test.txt"
RUN_BM25_PATH  = "datasets/run_bm25_test.txt"
MODEL_PATH     = "datasets/ltr_model.json"
ES_URL         = "http://localhost:9200"
BM25_INDEX     = "hqs"
SEMANTIC_INDEX = "hqs_semantic"
FEATURES       = [
    "bm25_score", "bm25_rank",
    "semantic_score", "semantic_rank",
    "title_bm25_score", "comic_name_bm25_score",
    "rank_diff",
]
TOP_K          = 100
TRAIN_RATIO    = 0.8
SEED           = 42


def load_qrels(path: str):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, doc_id, rel = parts
                qrels[qid][doc_id] = int(rel)
    return qrels


def get_test_query_ids(qrels: dict) -> set[str]:
    df = pd.read_csv(DATASET_PATH)
    query_ids = np.array([qid for qid in df["query_id"].unique() if qid in qrels])
    rng = np.random.default_rng(SEED)
    rng.shuffle(query_ids)
    n_train = int(len(query_ids) * TRAIN_RATIO)
    return set(query_ids[n_train:])


def load_queries(path: str):
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries.append((parts[0], parts[1]))
    return queries


def fetch_bm25(es: Elasticsearch, query: str):
    try:
        res = es.search(
            index=BM25_INDEX,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["issue_title^2", "issue_description", "comic_name^3"],
                        "type": "best_fields",
                    }
                }
            },
            size=TOP_K,
        )
        return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]
    except Exception:
        return []


def fetch_semantic(es: Elasticsearch, model: SentenceTransformer, query: str):
    try:
        vector = model.encode(query).tolist()
        res = es.search(
            index=SEMANTIC_INDEX,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": vector,
                    "k": TOP_K,
                    "num_candidates": TOP_K * 2,
                }
            },
            size=TOP_K,
        )
        return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]
    except Exception:
        return []


def fetch_field_bm25(es: Elasticsearch, query: str, field: str):
    try:
        res = es.search(
            index=BM25_INDEX,
            body={"query": {"match": {field: query}}},
            size=TOP_K,
        )
        return {hit["_id"]: hit["_score"] for hit in res["hits"]["hits"]}
    except Exception:
        return {}


def build_features(
    bm25_results: list,
    sem_results: list,
    title_scores: dict[str, float],
    comic_name_scores: dict[str, float],
):
    bm25_map = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(bm25_results)}
    sem_map  = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(sem_results)}
    candidates = list(set(bm25_map.keys()) | set(sem_map.keys()))
    rows = []
    for doc_id in candidates:
        bm25_score, bm25_rank = bm25_map.get(doc_id, (0.0, 0))
        sem_score,  sem_rank  = sem_map.get(doc_id,  (0.0, 0))
        title_score           = title_scores.get(doc_id, 0.0)
        comic_name_score      = comic_name_scores.get(doc_id, 0.0)
        rank_diff             = abs(bm25_rank - sem_rank)
        rows.append([bm25_score, bm25_rank, sem_score, sem_rank, title_score, comic_name_score, rank_diff])
    return candidates, np.array(rows, dtype=np.float32)


def ndcg_at_k(ranked_docs: list[str], qrel: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        qrel.get(doc, 0) / math.log2(i + 2)
        for i, doc in enumerate(ranked_docs[:k])
    )
    ideal = sorted(qrel.values(), reverse=True)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal[:k])
        if rel > 0
    )
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_docs: list[str], qrel: dict[str, int], k: int = 10) -> float:
    return sum(1 for doc in ranked_docs[:k] if qrel.get(doc, 0) > 0) / k


def average_precision(ranked_docs: list[str], qrel: dict[str, int]) -> float:
    relevant, ap = 0, 0.0
    for i, doc in enumerate(ranked_docs):
        if qrel.get(doc, 0) > 0:
            relevant += 1
            ap += relevant / (i + 1)
    total = sum(1 for v in qrel.values() if v > 0)
    return ap / total if total > 0 else 0.0


def reciprocal_rank(ranked_docs: list[str], qrel: dict[str, int]) -> float:
    for i, doc in enumerate(ranked_docs):
        if qrel.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_docs: list[str], qrel: dict[str, int], k: int = 10) -> float:
    total = sum(1 for v in qrel.values() if v > 0)
    if total == 0:
        return 0.0
    retrieved = sum(1 for doc in ranked_docs[:k] if qrel.get(doc, 0) > 0)
    return retrieved / total


def r_precision(ranked_docs: list[str], qrel: dict[str, int]) -> float:
    r = sum(1 for v in qrel.values() if v > 0)
    if r == 0:
        return 0.0
    return sum(1 for doc in ranked_docs[:r] if qrel.get(doc, 0) > 0) / r


def calcular_metricas(label: str, resultados: dict[str, list[str]], qrels: dict):
    ndcgs5, ndcgs10, maps, ps5, ps10, mrrs, recalls, rprecs = [], [], [], [], [], [], [], []
    for qid, docs in resultados.items():
        if qid not in qrels:
            continue
        qrel = qrels[qid]
        ndcgs5.append(ndcg_at_k(docs, qrel, k=5))
        ndcgs10.append(ndcg_at_k(docs, qrel, k=10))
        maps.append(average_precision(docs, qrel))
        ps5.append(precision_at_k(docs, qrel, k=5))
        ps10.append(precision_at_k(docs, qrel, k=10))
        mrrs.append(reciprocal_rank(docs, qrel))
        recalls.append(recall_at_k(docs, qrel, k=10))
        rprecs.append(r_precision(docs, qrel))

    if not maps:
        print(f"\n{label}: nenhuma query de teste encontrada no qrels.")
        return

    n = len(maps)
    print(f"\n── {label} ({n} queries) ─────────────────────")
    print(f"  MAP:        {sum(maps)/n:.4f}")
    print(f"  P@5:        {sum(ps5)/n:.4f}")
    print(f"  P@10:       {sum(ps10)/n:.4f}")
    print(f"  NDCG@5:     {sum(ndcgs5)/n:.4f}")
    print(f"  NDCG@10:    {sum(ndcgs10)/n:.4f}")
    print(f"  MRR:        {sum(mrrs)/n:.4f}")
    print(f"  Recall@10:  {sum(recalls)/n:.4f}")
    print(f"  R-Prec:     {sum(rprecs)/n:.4f}")


def main():
    print("Conectando ao Elasticsearch...")
    es = Elasticsearch([ES_URL])
    if not es.ping():
        raise SystemExit("Elasticsearch não está rodando em localhost:9200")

    print("Carregando modelo semântico...")
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Carregando modelo LTR: {MODEL_PATH}")
    ltr_model = xgb.Booster()
    ltr_model.load_model(MODEL_PATH)

    qrels    = load_qrels(QRELS_PATH)
    test_ids = get_test_query_ids(qrels)
    queries  = [(qid, text) for qid, text in load_queries(QUERIES_PATH) if qid in test_ids]

    print(f"\n{len(queries)} queries de teste\n")

    bm25_resultados: dict[str, list[str]] = {}
    ltr_resultados:  dict[str, list[str]] = {}

    Path(RUN_LTR_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(RUN_LTR_PATH,  "w", encoding="utf-8", newline="\n") as ltr_out, \
         open(RUN_BM25_PATH, "w", encoding="utf-8", newline="\n") as bm25_out:

        for i, (qid, text) in enumerate(queries, start=1):
            print(f"[{i:>3}/{len(queries)}] {qid}: {text[:60]}")

            bm25_results = fetch_bm25(es, text)
            sem_results  = fetch_semantic(es, sem_model, text)

            if not bm25_results and not sem_results:
                continue

            title_scores      = fetch_field_bm25(es, text, "issue_title")
            comic_name_scores = fetch_field_bm25(es, text, "comic_name")

            bm25_docs = [doc_id for doc_id, _ in bm25_results]
            bm25_resultados[qid] = bm25_docs

            for rank, (doc_id, score) in enumerate(bm25_results, start=1):
                bm25_out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\tbm25\n")

            doc_ids, X = build_features(bm25_results, sem_results, title_scores, comic_name_scores)
            scores  = ltr_model.predict(xgb.DMatrix(X, feature_names=FEATURES))

            ranked  = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
            ltr_docs = [doc_id for doc_id, _ in ranked]
            ltr_resultados[qid] = ltr_docs

            for rank, (doc_id, score) in enumerate(ranked, start=1):
                ltr_out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\tltr_xgboost\n")

    calcular_metricas("BM25", bm25_resultados, qrels)
    calcular_metricas("LTR", ltr_resultados, qrels)

if __name__ == "__main__":
    main()
