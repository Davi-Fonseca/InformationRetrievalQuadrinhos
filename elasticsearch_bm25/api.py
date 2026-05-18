from src.schemas import SearchResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.busca import multi_match_search
from src.indexa import connect_elasticsearch

app = FastAPI()

es_client = connect_elasticsearch()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API do ElasticSearch rodando!"}

@app.get("/search", response_model=SearchResponse)
def search_hqs(q: str, skip: int = 0, size: int = 10):
    res = multi_match_search(es_client, q, skip, size)

    results = [hit["_source"] for hit in res.body["hits"]["hits"]]

    return {"data": results}