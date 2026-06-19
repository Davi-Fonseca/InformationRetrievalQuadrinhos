import json
import itertools
from pathlib import Path

import numpy as np
import xgboost as xgb
from elasticsearch import Elasticsearch
from .indexa import connect_elasticsearch, get_embedding_model

import nltk
from nltk.corpus import wordnet
# Garante os dados do WordNet carregados
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


_LTR_FEATURES = [
    "bm25_score", "bm25_rank",
    "semantic_score", "semantic_rank",
    "title_bm25_score", "comic_name_bm25_score", "description_bm25_score",
    "rank_diff",
]

_ltr_model_instance = None

def get_ltr_model():
    global _ltr_model_instance
    if _ltr_model_instance is None:
        _ltr_model_instance = xgb.Booster()
        _ltr_model_instance.load_model(Path(__file__).parent.parent / "datasets" / "ltr_model.json")
    return _ltr_model_instance
def expandir_query(query: str) -> str:
    """Aplica Expansão Global de Query usando sinônimos do WordNet."""
    palavras = query.strip().split()
    termos_expandidos = []
    vistos = set()

    for palavra in palavras:
        palavra_lower = palavra.lower()
        if palavra_lower not in vistos:
            vistos.add(palavra_lower)
            termos_expandidos.append(palavra)

        syns = wordnet.synsets(palavra_lower)
        contador = 0
        for syn in syns:
            for lemma in syn.lemmas():
                sinonimo = lemma.name().lower().replace("_", " ")
                if " " not in sinonimo and sinonimo not in vistos:
                    vistos.add(sinonimo)
                    termos_expandidos.append(sinonimo)
                    contador += 1
                    if contador >= 2:
                        break
            if contador >= 2:
                break
    return " ".join(termos_expandidos)


def multi_match_grid_search(
    es: Elasticsearch, query, sw: bool, proc: str, sim: str, qe: bool, skip=0, size=5
):
    """Searches a specific index configuration, conditionally applying global query expansion."""
    # Modifica a query se a expansão estiver ativa
    if qe:
        query = expandir_query(query)

    index_name = f"hqs_sw_{'yes' if sw else 'no'}_proc_{proc}_sim_{sim}"

    res = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["issue_title^4", "issue_description^2", "comic_name^5"],
                    "type": "best_fields",
                }
            },
        },
        from_=skip,
        size=size,
    )
    return res


def run_full_grid_evaluation(es: Elasticsearch, search_query: str):
    """Executes the query across all 36 configurations to cross-examine top score profiles."""
    sw_options = [True, False]
    proc_options = ["none", "stemming", "lemmatization"]
    sim_options = ["bm25", "jelinek_mercer", "dirichlet", "vsm"]
    qe_options = [True, False]  # Inclusão da dimensão de Query Expansion

    print(f"\nAVALIANDO QUERY ORIGINAL: '{search_query}' ATRAVÉS DO GRID SEARCH EXPANDIDO")
    print(f"Query se expandida: '{expandir_query(search_query)}'")
    print("-" * 95)
    print(f"{'CONFIGURAÇÃO':<63} | {'MAX SCORE':<10} | {'TOP HIT ID':<10}")
    print("-" * 95)

    for sw, proc, sim, qe in itertools.product(
        sw_options, proc_options, sim_options, qe_options
    ):
        sw_label = "Removed" if sw else "Unchanged"
        qe_label = "Expanded" if qe else "Original"
        cfg_desc = f"SW:{sw_label:<9} | Proc:{proc:<13} | Sim:{sim:<14} | QE:{qe_label:<8}"

        try:
            res = multi_match_grid_search(
                es, search_query, sw, proc, sim, qe, size=1
            )
            hits = res["hits"]["hits"]

            if hits:
                max_score = round(res["hits"]["max_score"], 4)
                top_hit_id = hits[0]["_id"]
                print(f"{cfg_desc} | {max_score:<10} | {top_hit_id:<10}")
            else:
                print(f"{cfg_desc} | Sem resultados")
        except Exception as e:
            print(f"{cfg_desc} | Erro ao buscar: {e}")

def match_search(es: Elasticsearch, query: str, skip: int = 0, size: int = 10):
    res = es.search(
        index="hqs",
        body={
            "query": {
                "match": {
                    "issue_title": query,
                }
            },
        },
        from_=skip,
        size=size,
    )
    return res


def multi_match_search(es: Elasticsearch, query: str, skip: int = 0, size: int = 10):
    res = es.search(
        index="hqs",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["issue_title^4", "issue_description^2", "comic_name^5"],
                    "type": "best_fields",
                }
            },
        },
        from_=skip,
        size=size,
    )
    return res


def get_hq_by_id(es: Elasticsearch, id: str):
    return es.get(index="hqs", id=id)


def semantic_search(es: Elasticsearch, query: str, skip: int = 0, size: int = 10):
    query_vector = get_embedding_model().encode(query).tolist()

    res  = es.search(
        index="hqs_semantic",
        body={
            "knn": {
                "query_vector": query_vector,
                "field": "embedding",
                "k": 5,
                "num_candidates": 100,
            }
        },
        from_=skip,
        size=size,
    )
    return res

def hybrid_search(es: Elasticsearch, query: str, skip: int = 0, size: int = 10):
    """RRF manual: combina BM25 e kNN sem precisar de licença Enterprise."""
    query_vector = get_embedding_model().encode(query).tolist()
    k = 60 

    bm25_res = es.search(
        index="hqs_semantic",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["issue_title^4", "issue_description^2", "comic_name^5"],
                }
            }
        },
        size=50,
    )

    knn_res = es.search(
        index="hqs_semantic",
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": 50,
                "num_candidates": 100,
            }
        },
        size=50,
    )

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_res["hits"]["hits"]):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = hit["_source"]

    for rank, hit in enumerate(knn_res["hits"]["hits"]):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id] = hit["_source"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    paginated = ranked[skip: skip + size]

    return {"hits": {"hits": [{"_source": docs[doc_id]} for doc_id, _ in paginated]}}

def ltr_search(es: Elasticsearch, query: str, skip: int = 0, size: int = 10):
    TOP_K = 100

    bm25_res = es.search(
        index="hqs",
        body={"query": {"multi_match": {"query": query, "fields": ["issue_title^4", "issue_description^2", "comic_name^5"], "type": "best_fields"}}},
        size=TOP_K,
    )
    bm25_results = [(h["_id"], h["_score"]) for h in bm25_res["hits"]["hits"]]

    vector = get_embedding_model().encode(query).tolist()
    sem_res = es.search(
        index="hqs_semantic",
        body={"knn": {"field": "embedding", "query_vector": vector, "k": TOP_K, "num_candidates": TOP_K * 2}},
        size=TOP_K,
    )
    sem_results = [(h["_id"], h["_score"]) for h in sem_res["hits"]["hits"]]

    def field_scores(field):
        r = es.search(index="hqs", body={"query": {"match": {field: query}}}, size=TOP_K)
        return {h["_id"]: h["_score"] for h in r["hits"]["hits"]}

    title_scores  = field_scores("issue_title")
    comic_scores  = field_scores("comic_name")
    desc_scores   = field_scores("issue_description")

    bm25_map = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(bm25_results)}
    sem_map  = {doc_id: (score, rank + 1) for rank, (doc_id, score) in enumerate(sem_results)}
    candidates = list(set(bm25_map) | set(sem_map))

    rows = []
    for doc_id in candidates:
        bs, br = bm25_map.get(doc_id, (0.0, 0))
        ss, sr = sem_map.get(doc_id,  (0.0, 0))
        rows.append([bs, br, ss, sr,
                     title_scores.get(doc_id, 0.0),
                     comic_scores.get(doc_id, 0.0),
                     desc_scores.get(doc_id, 0.0),
                     abs(br - sr)])

    X = np.array(rows, dtype=np.float32)
    ltr_scores = get_ltr_model().predict(xgb.DMatrix(X, feature_names=_LTR_FEATURES))

    ranked = sorted(zip(candidates, ltr_scores), key=lambda x: x[1], reverse=True)
    page = ranked[skip: skip + size]

    doc_ids_page = [doc_id for doc_id, _ in page]
    if not doc_ids_page:
        return {"hits": {"hits": []}}

    mget_res = es.mget(index="hqs", body={"ids": doc_ids_page})
    sources = {d["_id"]: d["_source"] for d in mget_res["docs"] if d.get("found")}

    return {"hits": {"hits": [{"_source": sources[doc_id]} for doc_id, _ in page if doc_id in sources]}}


def main():
    es = connect_elasticsearch()

    print("\n--- BUSCA EM CONFIGURAÇÃO ÚNICA ESPECÍFICA (COM EXPANSÃO) ---")
    resultado_unico = multi_match_grid_search(
        es, query="thor", sw=True, proc="stemming", sim="bm25", qe=True
    )
    print(f"Total de hits na config escolhida: {resultado_unico['hits']['total']['value']}")

    # Caso queira rodar o exame macro de 36 combinações para uma query no terminal:
    # print("\n--- EXECUÇÃO COMPARATIVA EM TODO O GRID (36 CONFIGS) ---")
    # run_full_grid_evaluation(es, "green lantern")


if __name__ == "__main__":
    main()