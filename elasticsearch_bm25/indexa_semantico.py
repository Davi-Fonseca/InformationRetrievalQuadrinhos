from src.indexa import connect_elasticsearch, index_with_embeddings

es = connect_elasticsearch()
index_with_embeddings(es, "datasets/dataset.json")
