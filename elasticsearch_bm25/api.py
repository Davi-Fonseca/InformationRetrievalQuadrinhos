from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.busca import get_hq_by_id, match_search, multi_match_search
from src.indexa import connect_elasticsearch
from src.schemas import HQ, SearchResponse

app = FastAPI()

es_client = connect_elasticsearch()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8888"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API do ElasticSearch rodando!"}


@app.get("/search")
def search(q: str, skip: int = 0, size: int = 10):
    res = match_search(es_client, q, skip, size)

    results = [hit["_source"] for hit in res.body["hits"]["hits"]]

    return {"data": results}


@app.get("/search/simple", response_model=SearchResponse)
def search_hqs_simple(q: str, skip: int = 0, size: int = 10):
    res = match_search(es_client, q, skip, size)

    results = [hit["_source"] for hit in res.body["hits"]["hits"]]

    return {"data": results}


@app.get("/search/multi", response_model=SearchResponse)
def search_hqs_multi(q: str, skip: int = 0, size: int = 10):
    res = multi_match_search(es_client, q, skip, size)

    results = [hit["_source"] for hit in res.body["hits"]["hits"]]

    return {"data": results}


@app.get("/hq/{id}", response_model=HQ)
def get_hq(id: str):
    res = get_hq_by_id(es_client, id)

    if res and res.body.get("found"):
        return res.body["_source"]

    raise HTTPException(status_code=404, detail="HQ não encontrada")

