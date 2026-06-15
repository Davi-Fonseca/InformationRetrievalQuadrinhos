import itertools
import json
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def connect_elasticsearch():
    es = Elasticsearch(["http://localhost:9200"])
    if es.ping():
        print("✓ Conectado ao ElasticSearch com sucesso!")
    else:
        raise Exception("Erro ao conectar ao Elasticsearch.")
    return es


def create_grid_index(es: Elasticsearch, sw: bool, proc: str, sim: str):
    # Generates a unique index name based on the configuration matrix
    index_name = f"hqs_sw_{'yes' if sw else 'no'}_proc_{proc}_sim_{sim}"

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # 1. Configure Filters dynamically based on Text Processing choice
    filters = ["lowercase"]
    if sw:
        filters.append("english_stop")

    if proc == "stemming":
        filters.append("porter_stem")
    elif proc == "lemmatization":
        filters.append("kstem")  # Krovetz stemmer works as a linguistic lemmatizer proxy

    # 2. Configure Similarity Model dynamically
    similarity_config = {}
    if sim == "bm25":
        similarity_config = {"type": "BM25"}
    elif sim == "jelinek_mercer":
        similarity_config = {"type": "LMJelinekMercer", "lambda": 0.1}
    elif sim == "dirichlet":
        similarity_config = {"type": "LMDirichlet", "mu": 2000}

    # 3. Assemble Complete Index Settings
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"}
                },
                "analyzer": {
                    "grid_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": filters,
                    }
                },
            },
            "index": {"similarity": {"grid_similarity": similarity_config}},
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "comic_name": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "issue_title": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                },
                "issue_description": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                },
                "writer": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "penciler": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "cover_artist": {
                    "type": "text",
                    "analyzer": "grid_analyzer",
                    "similarity": "grid_similarity",
                    "fields": {"keyword": {"type": "keyword"}},
                },
            }
        },
    }

    es.indices.create(index=index_name, body=index_settings)
    return index_name


def load_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado em {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def bulk_indexing_action(docs, index_name):
    for doc in docs:
        yield {
            "_index": index_name,
            "_id": doc.get("id"),
            "_source": {
                "id": doc.get("id"),
                "comic_name": doc.get("comic_name"),
                "issue_title": doc.get("issue_title"),
                "issue_description": doc.get("issue_description"),
                "penciler": doc.get("penciler"),
                "writer": doc.get("writer"),
                "cover_artist": doc.get("cover_artist"),
            },
        }


def main():
    FILE_PATH = "datasets/dataset.json"

    try:
        es = connect_elasticsearch()
        docs = load_json_file(FILE_PATH)

        # Grid Search Hyperparameters
        sw_options = [True, False]
        proc_options = ["none", "stemming", "lemmatization"]
        sim_options = ["bm25", "jelinek_mercer", "dirichlet"]

        # Compute the Cartesian product of all configurations
        combinations = list(
            itertools.product(sw_options, proc_options, sim_options)
        )
        print(f"Iniciando criação e indexação de {len(combinations)} índices...")

        for sw, proc, sim in combinations:
            # Create configured index
            idx_name = create_grid_index(es, sw, proc, sim)

            # Bulk index documents into this configuration
            success, _ = bulk(es, bulk_indexing_action(docs, idx_name), stats_only=True)
            es.indices.refresh(index=idx_name)

            print(f" ✓ Índice [{idx_name}] configurado e com {success} HQs.")

        print("\n" + "=" * 60)
        print("Grid Search Indexing Completo!")
        print("=" * 60)

    except Exception as e:
        print(f"ERRO: {e}")


if __name__ == "__main__":
    main()