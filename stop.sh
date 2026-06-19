#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  stop.sh — Para completamente o projeto RI-2026.1
#
#  A parada ocorre do menor impacto para o maior:
#  1. Frontend (Next.js - porta 8888)
#  2. Backend (FastAPI - porta 8000)
#  3. Elasticsearch (Docker Compose)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ES_DIR="$ROOT_DIR/elasticsearch_bm25"

# ── Cores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
error()   { echo -e "${RED}[ERRO]${NC}  $*"; }
header()  { echo -e "\n${BOLD}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"; }

header "Parando o projeto RI-2026.1"

# ── 1. Frontend ─────────────────────────────────────────────────────────────
info "Verificando processo do Frontend (porta 8888)..."
if command -v lsof >/dev/null 2>&1; then
    FRONTEND_PIDS=$(lsof -ti:8888 || true)
    if [[ -n "$FRONTEND_PIDS" ]]; then
        info "Parando Frontend (PIDs: $(echo $FRONTEND_PIDS | tr '\n' ' '))..."
        kill -9 $FRONTEND_PIDS 2>/dev/null || true
        success "Frontend parado."
    else
        success "Frontend não estava rodando."
    fi
else
    info "Comando 'lsof' não encontrado, tentando fuser..."
    if fuser -k 8888/tcp >/dev/null 2>&1; then
        success "Frontend parado via fuser."
    else
        success "Frontend não estava rodando ou fuser falhou."
    fi
fi

# ── 2. Backend ──────────────────────────────────────────────────────────────
echo ""
info "Verificando processo do Backend (porta 8000)..."
if command -v lsof >/dev/null 2>&1; then
    BACKEND_PIDS=$(lsof -ti:8000 || true)
    if [[ -n "$BACKEND_PIDS" ]]; then
        info "Parando Backend (PIDs: $(echo $BACKEND_PIDS | tr '\n' ' '))..."
        kill -9 $BACKEND_PIDS 2>/dev/null || true
        success "Backend parado."
    else
        success "Backend não estava rodando."
    fi
else
    info "Comando 'lsof' não encontrado, tentando fuser..."
    if fuser -k 8000/tcp >/dev/null 2>&1; then
        success "Backend parado via fuser."
    else
        success "Backend não estava rodando ou fuser falhou."
    fi
fi

# ── 3. Elasticsearch ────────────────────────────────────────────────────────
echo ""
info "Verificando Elasticsearch (Docker)..."
if cd "$ES_DIR" && docker compose ps | grep -qi "up\|running"; then
    info "Parando e removendo containers do Elasticsearch..."
    docker compose down
    success "Elasticsearch parado."
else
    success "Elasticsearch não estava rodando via Docker Compose."
fi

echo -e "\n${GREEN}${BOLD}Projeto parado com sucesso! Ambiente limpo.${NC}\n"
