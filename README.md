# Sistema de Recuperação de Informação — HQs

Trabalho da disciplina de Recuperação de Informação (2026.1). Sistema de busca para uma coleção de histórias em quadrinhos que implementa três abordagens progressivamente mais sofisticadas: BM25, busca híbrida (BM25 + semântica via RRF) e Learning to Rank (LambdaMART).

**Equipe:** Lucas Martini · André Cristo · Davi Fonseca · José Lázaro

---

## Visão Geral

```
Consulta do usuário
        │
        ├──► /search/semantic  → Busca vetorial (kNN)
        ├──► /search/hybrid    → BM25 + kNN com Reciprocal Rank Fusion
        └──► /search/ltr       → BM25 + kNN + reranking por LambdaMART
```

A coleção contém ~8.000 documentos de HQs com campos de título, descrição, roteiristas e desenhistas, indexados no Elasticsearch. A API é construída com FastAPI e o frontend com Next.js.

---

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Motor de busca | Elasticsearch |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Learning to Rank | XGBoost (LambdaMART, `rank:ndcg`) |
| API | FastAPI + Uvicorn |
| Frontend | Next.js (TypeScript) |
| Avaliação | `trec_eval` |

---

## Pré-requisitos

- Python 3.10+
- Node.js 18+ e npm
- Elasticsearch 8.x rodando localmente na porta `9200`
- (Opcional) Docker para o frontend

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositório>
cd "Trabalho de RI"

# 2. Crie e ative o ambiente virtual Python
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências Python
pip install elasticsearch sentence-transformers xgboost fastapi uvicorn pandas numpy nltk pydantic python-dotenv

# 4. Instale as dependências do frontend
cd frontend && npm install && cd ..
```

---

## Como Executar

### Pipeline completo (indexação + treinamento + avaliação)

```bash
bash run.sh
```

Flags disponíveis:
- `--skip-eval` — pula a etapa de avaliação
- `--only-eval` — executa apenas a avaliação

### Passo a passo manual

```bash
cd elasticsearch_bm25

# 1. Criar índices BM25 (grid de configurações)
python src/indexa.py

# 2. Criar índice semântico e gerar embeddings
python indexa_semantico.py

# 3. Criar splits treino/validação/teste (70/10/20)
python src/cria_splits.py

# 4. Extrair features para o LTR
python src/prepara_dataset_ltr.py

# 5. Treinar modelo LambdaMART
python treina_ltr.py

# 6. Iniciar a API
uvicorn api:app --reload --port 8888

# 7. (Outro terminal) Iniciar o frontend
cd ../frontend && npm run dev
```

O sistema estará disponível em `http://localhost:8888`.

---

## Estrutura do Projeto

```
.
├── run.sh / stop.sh                   # Scripts de pipeline e limpeza
├── compose.yml                        # Docker Compose (frontend)
├── frontend/                          # Interface Next.js
└── elasticsearch_bm25/
    ├── api.py                         # API REST (FastAPI)
    ├── treina_ltr.py                  # Treinamento LambdaMART
    ├── avalia.py / avalia_ltr.py      # Geração de run files e avaliação
    ├── avalia_grid.py                 # Grid search de configurações BM25
    ├── indexa_semantico.py            # Criação do índice vetorial
    ├── datasets/
    │   ├── dataset.json               # Coleção (~8k HQs)
    │   ├── queries.tsv / qrels.txt    # Consultas e julgamentos de relevância
    │   ├── embeddings.npy             # Cache de embeddings (384-dim)
    │   └── ltr_model.json             # Modelo XGBoost treinado
    └── src/
        ├── busca.py                   # Implementações de busca (semântica, híbrida, LTR)
        ├── indexa.py                  # Indexação e grid BM25
        ├── cria_splits.py             # Divisão treino/val/teste
        ├── prepara_dataset_ltr.py     # Extração de features para o LTR
        └── schemas.py                 # Modelos Pydantic
```

---

## API — Endpoints de Busca

Todos aceitam os parâmetros `q` (consulta), `skip` (offset) e `size` (padrão: 10).

| Endpoint | Descrição |
|---|---|
| `GET /search/semantic` | Busca vetorial kNN com `all-MiniLM-L6-v2` |
| `GET /search/hybrid` | BM25 + kNN combinados por Reciprocal Rank Fusion |
| `GET /search/ltr` | Reranking dos candidatos com LambdaMART (XGBoost) |
| `GET /hq/{id}` | Retorna um documento específico pelo ID |

**Exemplo:**
```bash
curl "http://localhost:8888/search/ltr?q=batman+gotham&size=5"
```

---

## Pipeline de Learning to Rank

### Features extraídas por par (consulta, documento)

| Feature | Descrição |
|---|---|
| `bm25_score` | Score BM25 multi-campo |
| `bm25_rank` | Posição no ranking BM25 |
| `semantic_score` | Similaridade cosseno (kNN) |
| `semantic_rank` | Posição no ranking semântico |
| `title_bm25_score` | BM25 apenas no título |
| `comic_name_bm25_score` | BM25 apenas no nome da série |
| `description_bm25_score` | BM25 apenas na descrição |
| `rank_diff` | Diferença absoluta entre `bm25_rank` e `semantic_rank` |

### Treinamento

- Algoritmo: **LambdaMART** (`rank:ndcg` no XGBoost)
- Divisão por **query** (70% treino / 10% validação / 20% teste), seed=42
- Early stopping com paciência de 10 rounds sem melhora em NDCG@10

