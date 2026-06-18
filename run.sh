#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  run.sh — Pipeline completo do projeto RI-2026.1
#
#  Uso:
#    chmod +x run.sh
#    ./run.sh              # roda tudo (padrão)
#    ./run.sh --skip-eval  # pula a etapa de avaliação (trec_eval)
#    ./run.sh --only-eval  # roda apenas a avaliação (ES já deve estar rodando)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ── Diretórios ──────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ES_DIR="$ROOT_DIR/elasticsearch_bm25"
FRONTEND_DIR="$ROOT_DIR/frontend"

# ── Cores para output ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helpers ─────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERRO]${NC}  $*"; }
header()  { echo -e "\n${BOLD}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"; }

# ── Parse de flags ──────────────────────────────────────────────────────────
SKIP_EVAL=false
ONLY_EVAL=false

for arg in "$@"; do
    case "$arg" in
        --skip-eval) SKIP_EVAL=true ;;
        --only-eval) ONLY_EVAL=true ;;
        -h|--help)
            echo "Uso: $0 [--skip-eval] [--only-eval]"
            echo "  --skip-eval  Pula a etapa de avaliação com trec_eval"
            echo "  --only-eval  Roda apenas a avaliação (ES já deve estar rodando e indexado)"
            exit 0
            ;;
    esac
done

# ── Verificação de dependências ─────────────────────────────────────────────
check_dependency() {
    if ! command -v "$1" &>/dev/null; then
        error "'$1' não encontrado. Instale antes de continuar."
        return 1
    fi
}

# ── PIDs de processos em background (para cleanup) ──────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    header "Encerrando processos..."

    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        info "Parando backend (PID $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
        success "Backend encerrado."
    fi

    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        info "Parando frontend (PID $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
        success "Frontend encerrado."
    fi

    success "Cleanup concluído. Até mais!"
}

trap cleanup EXIT INT TERM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 0 — Verificar dependências
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 0 — Verificando dependências"

DEPS_OK=true
for dep in docker pnpm uv curl lsof; do
    if check_dependency "$dep"; then
        success "$dep encontrado: $(command -v "$dep")"
    else
        DEPS_OK=false
    fi
done

if [[ "$SKIP_EVAL" == false ]]; then
    if command -v trec_eval &>/dev/null; then
        success "trec_eval encontrado: $(command -v trec_eval)"
    else
        warn "trec_eval não encontrado no PATH. A avaliação será pulada."
        warn "Para instalar: git clone https://github.com/usnistgov/trec_eval && cd trec_eval && make && sudo mv trec_eval /usr/local/bin/"
    fi
fi

if [[ "$DEPS_OK" == false ]]; then
    error "Dependências faltando. Corrija os erros acima e tente novamente."
    exit 1
fi

# ── Pular para avaliação se --only-eval ─────────────────────────────────────
if [[ "$ONLY_EVAL" == true ]]; then
    header "ETAPA ÚNICA — Avaliação (--only-eval)"
    info "Rodando avaliação com trec_eval..."
    cd "$ES_DIR"
    uv run python avalia_grid.py
    info "Rodando avaliação do LTR (conjunto de teste)..."
    uv run python avalia_ltr.py
    success "Avaliações concluídas!"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 0.5 — Verificar se já está rodando
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 0.5 — Verificando instâncias existentes"

ALREADY_RUNNING=false

if command -v lsof >/dev/null 2>&1; then
    if lsof -ti:8888 >/dev/null || lsof -ti:8000 >/dev/null; then
        ALREADY_RUNNING=true
    fi
elif fuser 8888/tcp >/dev/null 2>&1 || fuser 8000/tcp >/dev/null 2>&1; then
    ALREADY_RUNNING=true
fi

if cd "$ES_DIR" && docker compose ps | grep -qi "up\|running"; then
    ALREADY_RUNNING=true
fi

if [[ "$ALREADY_RUNNING" == true ]]; then
    warn "Instância do projeto já em execução detectada."
    info "Resetando o ambiente (chamando stop.sh)..."
    "$ROOT_DIR/stop.sh"
    success "Ambiente resetado. Iniciando novo deploy..."
else
    success "Nenhuma instância anterior detectada."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 1 — Subir o Elasticsearch via Docker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 1 — Subindo Elasticsearch (Docker)"

ES_URL="http://localhost:9200"

cd "$ES_DIR"
info "Subindo container via docker compose..."
docker compose up -d

# ── Aguardar Elasticsearch ficar pronto ─────────────────────────────────────
info "Aguardando Elasticsearch ficar pronto em $ES_URL..."

MAX_RETRIES=30
RETRY_INTERVAL=18
ATTEMPT=0

while [[ $ATTEMPT -lt $MAX_RETRIES ]]; do
    ATTEMPT=$((ATTEMPT + 1))

    if curl -s -o /dev/null -w "%{http_code}" "$ES_URL" | grep -q "200"; then
        ES_VERSION=$(curl -s "$ES_URL" | python3 -c "import sys,json; print(json.load(sys.stdin)['version']['number'])" 2>/dev/null || echo "desconhecida")
        success "Elasticsearch pronto! (v$ES_VERSION) — tentativa $ATTEMPT/$MAX_RETRIES"
        break
    fi

    if [[ $ATTEMPT -eq $MAX_RETRIES ]]; then
        error "Elasticsearch não respondeu após $MAX_RETRIES tentativas."
        error "Verifique os logs: docker compose -f $ES_DIR/compose.yaml logs elasticsearch"
        exit 1
    fi

    echo -ne "\r  Tentativa $ATTEMPT/$MAX_RETRIES... aguardando ${RETRY_INTERVAL}s"
    sleep "$RETRY_INTERVAL"
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 2 — Instalar dependências Python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 2 — Instalando dependências Python (uv sync)"

cd "$ES_DIR"
uv sync
success "Dependências Python instaladas."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 3 — Indexação no Elasticsearch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 3 — Indexação (BM25 + LMs + VSM — Grid Search)"

cd "$ES_DIR"

# 3a. Índice padrão + Grid Search (BM25, Jelinek-Mercer, Dirichlet, VSM)
info "Criando índice padrão + 24 índices do Grid Search..."
uv run python -m src.indexa
success "Indexação principal e Grid Search concluídos."

# 3b. Índice semântico (embeddings)
info "Criando índice semântico (embeddings com all-MiniLM-L6-v2)..."
uv run python indexa_semantico.py
success "Indexação semântica concluída."

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 4 — Avaliação com trec_eval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$SKIP_EVAL" == false ]]; then
    header "ETAPA 4 — Avaliação (Grid Search + trec_eval)"

    cd "$ES_DIR"

    if command -v trec_eval &>/dev/null; then
        info "Gerando run files e avaliando 48 configurações..."
        uv run python avalia_grid.py
        info "Rodando avaliação do LTR..."
        uv run python avalia_ltr.py
        success "Avaliação concluída! Resultados em $ES_DIR/datasets/runs/"
    else
        warn "trec_eval não encontrado. Gerando apenas os run files..."
        uv run python avalia_grid.py
        uv run python avalia_ltr.py
        warn "Instale o trec_eval para ver as métricas. Os run files estão em $ES_DIR/datasets/runs/"
    fi
else
    info "Avaliação pulada (--skip-eval)."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 5 — Subir o Backend (FastAPI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 5 — Subindo Backend (FastAPI + Uvicorn)"

cd "$ES_DIR"
info "Iniciando API em http://localhost:8000 ..."
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
success "Backend iniciado em background (PID $BACKEND_PID)"

# Aguardar o backend ficar pronto
sleep 3
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/" | grep -q "200"; then
    success "Backend respondendo em http://localhost:8000"
else
    warn "Backend ainda não respondeu. Pode levar alguns segundos para carregar o modelo de embeddings."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPA 6 — Subir o Frontend (Next.js)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "ETAPA 6 — Subindo Frontend (Next.js)"

cd "$FRONTEND_DIR"
info "Instalando dependências do frontend..."
pnpm install --silent
success "Dependências instaladas."

info "Iniciando frontend em http://localhost:8888 ..."
pnpm run dev &
FRONTEND_PID=$!
success "Frontend iniciado em background (PID $FRONTEND_PID)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESUMO FINAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
header "Pipeline concluído com sucesso!"

echo -e ""
echo -e "  ${BOLD}Serviços rodando:${NC}"
echo -e "  ────────────────────────────────────────────────"
echo -e "  ${CYAN}Elasticsearch${NC}  →  http://localhost:9200"
echo -e "  ${CYAN}Backend API${NC}    →  http://localhost:8000"
echo -e "  ${CYAN}Frontend${NC}       →  http://localhost:8888"
echo -e "  ────────────────────────────────────────────────"
echo -e ""
echo -e "  ${BOLD}Endpoints da API:${NC}"
echo -e "  GET /search/simple?q=...   — Busca simples (match)"
echo -e "  GET /search/multi?q=...    — Busca multi-campo (BM25)"
echo -e "  GET /search/semantic?q=... — Busca semântica (kNN)"
echo -e "  GET /search/hybrid?q=...   — Busca híbrida (RRF)"
echo -e "  GET /hq/{id}               — Detalhes de uma HQ"
echo -e ""
echo -e "  ${YELLOW}Pressione Ctrl+C para encerrar todos os serviços.${NC}"
echo -e ""

# Manter o script rodando até Ctrl+C
wait
