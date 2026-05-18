import json
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


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
                "id": {
                    "type": "keyword",
                },
                "comic_name": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                        }
                    },
                },
                "issue_title": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                },
                "issue_description": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                },
                "writer": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                        }
                    },
                },
                "penciler": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                        }
                    },
                },
                "cover_artist": {
                    "type": "text",
                    "analyzer": "english_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                        }
                    },
                },
            }
        },
    }

    es.indices.create(index=index_name, body=index_settings)

    print(f"Índice '{index_name}' criado com sucesso!")
    print(
        "Campos: id, comic_name, issue_title, issue_description, penciler, writer, cover_artist"
    )
    print("Analyzer: english_analyzer (com stopwords em inglês)")


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
        print(
            f"Acesse http://localhost:9200/{INDEX_NAME}/_count para ver o total de HQs"
        )

    except Exception as e:
        print(f"ERRO: {e}")


if __name__ == "__main__":
    main()
