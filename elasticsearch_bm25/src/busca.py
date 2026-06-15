import json
import itertools
from elasticsearch import Elasticsearch
from indexa import connect_elasticsearch

import nltk
from nltk.corpus import wordnet

# Garante os dados do WordNet carregados
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


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
                    "fields": ["issue_title^2", "issue_description", "comic_name^3"],
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
    sim_options = ["bm25", "jelinek_mercer", "dirichlet"]
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