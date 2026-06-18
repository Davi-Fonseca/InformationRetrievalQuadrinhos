import itertools
import json
import os

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

embedding_model = None


def get_embedding_model():
    """Carrega o modelo de embeddings apenas na primeira chamada (lazy singleton)."""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return embedding_model


def connect_elasticsearch():
    es = Elasticsearch(["http://localhost:9200"])

    if es.ping():
        print("✓ Conectado ao ElasticSearch com sucesso!")
        info = es.info()
        print(f"Versão: {info['version']['number']}")
    else:
        raise Exception("Erro ao conecatar ao Elasticsearch.")

    return es


def create_index(es: Elasticsearch, index_name="hqs"):
    if es.indices.exists(index=index_name):
        print("Indice existente, removendo.")
        es.indices.delete(index=index_name)
        print("Indice removido.")

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "english_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_",
                    }
                },
            },
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "comic_name": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "issue_title": {"type": "text", "analyzer": "english_analyzer"},
                "issue_description": {"type": "text", "analyzer": "english_analyzer"},
                "writer": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "penciler": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "cover_artist": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
            }
        },
    }

    es.indices.create(index=index_name, body=index_settings)
    print(f"Índice '{index_name}' criado com sucesso!")


def create_grid_index(es: Elasticsearch, sw: bool, proc: str, sim: str):
    index_name = f"hqs_sw_{'yes' if sw else 'no'}_proc_{proc}_sim_{sim}"

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    filters = ["lowercase"]
    if sw:
        filters.append("english_stop")

    if proc == "stemming":
        filters.append("porter_stem")
    elif proc == "lemmatization":
        filters.append("kstem")

    similarity_config = {}
    if sim == "bm25":
        similarity_config = {"type": "BM25"}
    elif sim == "jelinek_mercer":
        similarity_config = {"type": "LMJelinekMercer", "lambda": 0.1}
    elif sim == "dirichlet":
        similarity_config = {"type": "LMDirichlet", "mu": 2000}
    elif sim == "vsm":
        similarity_config = {
            "type": "scripted",
            "weight_script": {
                "source": (
                    "double idf = Math.log((field.docCount + 1.0)"
                    " / (term.docFreq + 1.0)) + 1.0;"
                    " return query.boost * idf;"
                )
            },
            "script": {
                "source": (
                    "double tf = Math.sqrt(doc.freq);"
                    " double norm = 1.0 / Math.sqrt(doc.length);"
                    " return weight * tf * norm;"
                )
            },
        }

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {"english_stop": {"type": "stop", "stopwords": "_english_"}},
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
        data = json.load(file)

    print("Arquivo carregado com sucesso!")
    print(f"Total de HQs lidas: {len(data)}")

    return data


def bulk_indexing_action(docs, index_name="hqs"):
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


def indexing(es: Elasticsearch, docs, index_name="hqs"):
    print("Indexando documentos...")
    success, failure = bulk(es, bulk_indexing_action(docs, index_name), stats_only=True)
    print("Indexação concluída!")
    print(f"HQs indexadas: {success}")
    if failure:
        print(f"Erros: {failure}")
    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)
    print(f"Total de HQs no índice: {count['count']}")


def create_semantic_index(es: Elasticsearch):
    index_name = "hqs_semantic"

    if es.indices.exists(index=index_name):
        return index_name

    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "comic_name": {"type": "text"},
                "issue_title": {"type": "text"},
                "issue_description": {"type": "text"},
                "writer": {"type": "text"},
                "penciler": {"type": "text"},
                "cover_artist": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }

    es.indices.create(index=index_name, body=mapping)
    return index_name


def index_with_embeddings(es: Elasticsearch, dataset_path: str):
    index_name = "hqs_semantic"

    print("Carregando dataset...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Verifica se o índice já está populado
    if es.indices.exists(index=index_name):
        count = es.count(index=index_name).get("count", 0)
        if count == len(docs):
            print(
                f"✅ O índice semântico já existe com {count} documentos! Pulando recomputação pesada."
            )
            return

    # Se não existir, cria o mapping
    create_semantic_index(es)

    embeddings_file = "datasets/embeddings.npy"
    if os.path.exists(embeddings_file):
        print(
            "⚡ Arquivo de embeddings pré-calculados encontrado! Lendo direto do disco..."
        )
        embeddings = np.load(embeddings_file)
    else:
        # Prepara os textos de todos os documentos
        texts = [
            f"{doc.get('comic_name', '')} {doc.get('issue_title', '')} {doc.get('issue_description', '')}"
            for doc in docs
        ]

        print(
            f"Gerando embeddings para {len(docs)} documentos (isso pode demorar um pouco)..."
        )
        model = get_embedding_model()
        # O modelo processa a lista inteira internamente em batches otimizados e exibe barra de progresso
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

        # Salva para as próximas vezes
        np.save(embeddings_file, embeddings)
        print(
            "✅ Embeddings salvos em 'datasets/embeddings.npy' para acelerar futuras execuções!"
        )

    print("Enviando para o Elasticsearch...")
    actions = []
    for doc, vector in zip(docs, embeddings):
        actions.append({"index": {"_index": index_name, "_id": str(doc["id"])}})
        actions.append({**doc, "embedding": vector.tolist()})

        if len(actions) >= 400:  # 200 docs * 2 (index_action + data)
            es.bulk(body=actions)
            actions = []

    if actions:
        es.bulk(body=actions)

    print(f"Indexação semântica concluída: {len(docs)} documentos")


def main():
    INDEX_NAME = "hqs"
    FILE_PATH = "datasets/dataset.json"

    try:
        print("1. Tentando conexão com o Elasticsearch")
        es = connect_elasticsearch()
        print()

        print("2. Criando o Índice")
        create_index(es, INDEX_NAME)
        print()

        print("3. Lendo as HQs do JSON")
        docs = load_json_file(FILE_PATH)
        print()

        print("4. Indexando HQs")
        indexing(es, docs, INDEX_NAME)
        print()

        print("=" * 60)
        print("Indexação concluída")
        print("=" * 60)
        print()

        print("5. Iniciando Grid Search de Hiperparâmetros")
        sw_options = [True, False]
        proc_options = ["none", "stemming", "lemmatization"]
        sim_options = ["bm25", "jelinek_mercer", "dirichlet", "vsm"]

        combinations = list(itertools.product(sw_options, proc_options, sim_options))
        print(f"Criando e indexando {len(combinations)} índices...")
        print()

        for sw, proc, sim in combinations:
            idx_name = create_grid_index(es, sw, proc, sim)
            success, _ = bulk(es, bulk_indexing_action(docs, idx_name), stats_only=True)
            es.indices.refresh(index=idx_name)
            print(f" ✓ Índice [{idx_name}] configurado e com {success} HQs.")

        print()
        print("=" * 60)
        print("Grid Search Indexing Completo!")
        print("=" * 60)

    except Exception as e:
        print(f"ERRO: {e}")


if __name__ == "__main__":
    main()
