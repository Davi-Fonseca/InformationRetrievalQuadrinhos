import itertools
from collections import defaultdict

from elasticsearch import Elasticsearch

from avalia_ltr import load_qrels, ndcg_at_k

ES_URL = "http://localhost:9200"
INDEX = "hqs"
VAL_QUERIES_PATH = "datasets/queries_val.tsv"
VAL_QRELS_PATH = "datasets/qrels_val.txt"
TREINO_QUERIES_PATH = "datasets/queries_treino.tsv"
TREINO_QRELS_PATH = "datasets/qrels_treino.txt"


def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries.append((parts[0], parts[1]))
    return queries


def search_with_weights(es, query, weights, size=10):
    t_w, c_w, d_w = weights
    fields = []
    if t_w > 0:
        fields.append(f"issue_title^{t_w}")
    if c_w > 0:
        fields.append(f"comic_name^{c_w}")
    if d_w > 0:
        fields.append(f"issue_description^{d_w}")

    res = es.search(
        index=INDEX,
        body={
            "query": {
                "multi_match": {"query": query, "fields": fields, "type": "best_fields"}
            }
        },
        size=size,
    )
    return [hit["_id"] for hit in res["hits"]["hits"]]


def main():
    print("Iniciando Descoberta de Pesos do BM25 (Grid Search)")
    es = Elasticsearch([ES_URL])

    qrels_val = load_qrels(VAL_QRELS_PATH)
    queries_val = load_queries(VAL_QUERIES_PATH)

    qrels_treino = load_qrels(TREINO_QRELS_PATH)
    queries_treino = load_queries(TREINO_QUERIES_PATH)

    # Juntar Treino e Validação
    qrels_combo = {**qrels_val, **qrels_treino}
    queries_combo = queries_val + queries_treino

    # Reduzindo para as queries válidas
    queries_combo = [(qid, text) for qid, text in queries_combo if qid in qrels_combo]

    print(f"Testando em {len(queries_combo)} queries combinadas (Treino + Validação).")

    # Definir grid de pesos (pesos variando de 1 a 4)
    # Como a descrição geralmente é longa, mantemos pesos menores para ela.
    title_weights = [1, 2, 3, 4, 5, 6, 7, 8]
    comic_weights = [1, 2, 3, 4, 5, 6, 7, 8]
    desc_weights = [1, 2, 3, 4, 5]

    best_ndcg = 0.0
    best_weights = None

    combinations = list(itertools.product(title_weights, comic_weights, desc_weights))
    print(f"Total de combinações a testar: {len(combinations)}")

    results = []

    for i, weights in enumerate(combinations, 1):
        ndcgs = []
        for qid, text in queries_combo:
            docs = search_with_weights(es, text, weights, size=10)
            score = ndcg_at_k(docs, qrels_combo[qid], k=10)
            ndcgs.append(score)

        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        results.append((avg_ndcg, weights))

        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_weights = weights

        if i % 10 == 0 or i == len(combinations):
            print(
                f"Progresso: {i}/{len(combinations)} | Melhor NDCG atual: {best_ndcg:.4f} (Pesos: {best_weights})"
            )

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n==============================================")
    print("TOP 3 MELHORES COMBINAÇÕES (NDCG@10 no Treino + Validação):")
    for i in range(3):
        print(
            f"{i+1}º Lugar: Título^{results[i][1][0]} | Revista^{results[i][1][1]} | Desc^{results[i][1][2]} --> NDCG@10 = {results[i][0]:.4f}"
        )

    # O default antigo era Título^2, Revista^3, Desc^1
    print("\nCombinação antiga (Título^2, Revista^3, Desc^1):")
    for r in results:
        if r[1] == (2, 3, 1):
            print(f"NDCG@10 = {r[0]:.4f}")
            break


if __name__ == "__main__":
    main()
