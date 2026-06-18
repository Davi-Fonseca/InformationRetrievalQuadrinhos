import csv
from collections import defaultdict
from pathlib import Path

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

QUERIES_PATH   = Path("datasets/queries.tsv")
QRELS_PATH     = Path("datasets/qrels.txt")
OUTPUT_PATH    = Path("datasets/dataset_ltr.csv")
ES_URL         = "http://localhost:9200"
BM25_INDEX     = "hqs"
SEMANTIC_INDEX = "hqs_semantic"
TOP_K          = 100


def load_queries(path: Path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries


def load_qrels(path: Path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, doc_id, rel = parts
                qrels[qid][doc_id] = int(rel)
    return qrels


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
    except Exception as e:
        print(e)
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
    except Exception as e:
        print(e)
        return []


def fetch_field_bm25(es: Elasticsearch, query: str, field: str):
    try:
        res = es.search(
            index=BM25_INDEX,
            body={"query": {"match": {field: query}}},
            size=TOP_K,
        )
        return {hit["_id"]: hit["_score"] for hit in res["hits"]["hits"]}
    except Exception as e:
        print(f"  [ERRO BM25 campo {field}] {e}")
        return {}


def main():
    print("Conectando ao Elasticsearch...")
    es = Elasticsearch([ES_URL])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Modelo carregado!\n")

    queries = load_queries(QUERIES_PATH)
    qrels   = load_qrels(QRELS_PATH)

    total_qrels = sum(len(v) for v in qrels.values())
    print(f"{len(queries)} queries carregadas")
    print(f"{total_qrels} pares (query, doc) nos qrels\n")

    rows  = []
    total = len(queries)

    for i, (qid, text) in enumerate(queries.items(), start=1):
        print(f"[{i:>3}/{total}] {qid}: {text[:60]}")

        bm25_results = fetch_bm25(es, text)
        bm25_map = {
            doc_id: (score, rank + 1)
            for rank, (doc_id, score) in enumerate(bm25_results)
        }

        sem_results = fetch_semantic(es, model, text)
        sem_map = {
            doc_id: (score, rank + 1)
            for rank, (doc_id, score) in enumerate(sem_results)
        }

        title_scores      = fetch_field_bm25(es, text, "issue_title")
        comic_name_scores = fetch_field_bm25(es, text, "comic_name")

        candidates = (
            set(qrels.get(qid, {}).keys())
            | set(bm25_map.keys())
            | set(sem_map.keys())
        )

        for doc_id in candidates:
            relevance              = qrels.get(qid, {}).get(doc_id, 0)
            bm25_score, bm25_rank  = bm25_map.get(doc_id, (0.0, 0))
            sem_score,  sem_rank   = sem_map.get(doc_id,  (0.0, 0))
            title_score            = title_scores.get(doc_id, 0.0)
            comic_name_score       = comic_name_scores.get(doc_id, 0.0)
            rank_diff              = abs(bm25_rank - sem_rank)

            rows.append({
                "query_id":              qid,
                "doc_id":                doc_id,
                "relevance":             relevance,
                "bm25_score":            round(bm25_score, 6),
                "bm25_rank":             bm25_rank,
                "semantic_score":        round(sem_score, 6),
                "semantic_rank":         sem_rank,
                "title_bm25_score":      round(title_score, 6),
                "comic_name_bm25_score": round(comic_name_score, 6),
                "rank_diff":             rank_diff,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id", "doc_id", "relevance",
        "bm25_score", "bm25_rank",
        "semantic_score", "semantic_rank",
        "title_bm25_score", "comic_name_bm25_score",
        "rank_diff",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    relevant     = sum(1 for row in rows if row["relevance"] > 0)
    not_relevant = len(rows) - relevant

    print(f"\nDataset salvo em {OUTPUT_PATH}")
    print(f"  Total de linhas (pares query-doc): {len(rows):,}")
    print(f"  Relevantes:     {relevant:,}")
    print(f"  Não relevantes: {not_relevant:,}")


if __name__ == "__main__":
    main()
