import json

from elasticsearch import Elasticsearch

from .indexa import connect_elasticsearch


def match_search(es: Elasticsearch, query, skip=0, size=10):
    print(f"\n{'='*80}")
    print(f"BUSCA SIMPLES: '{query}'")
    print(f"Tipo: Match Query | Campo: issue_title")
    print("=" * 80)

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


def multi_match_search(es: Elasticsearch, query, skip=0, size=10):
    print(f"\n{'='*80}")
    print(f"BUSCA MULTI-CAMPO: '{query}'")
    print(f"Campos: title (peso 2x), highlight, judging_organ")
    print("=" * 80)

    res = es.search(
        index="hqs",
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["issue_title^2", "issue_description", "comic_name^3"],
                    "type": "best_fields",
                }
            },
        },
        from_ = skip,
        size = size,
    )

    return res

def get_hq_by_id(es: Elasticsearch, id: str):
    try:
        res = es.get(index="hqs", id=id)
        return res
    except Exception as e:
        return None

def main():
    es = connect_elasticsearch()
    simple_match = match_search(es, "thor #1", 5)
    multi_match = multi_match_search(es, "thor #1", 5)

    print("RESULTADO SIMPLE MATCH\n")
    print()
    print(json.dumps(simple_match.body, indent=4))
    print()
    print()
    print()
    print("RESULTADO MULTI MATCH")
    print()
    print(json.dumps(multi_match.body, indent=4))


if __name__ == "__main__":
    main()
