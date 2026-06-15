"""
Gera o run file do sistema BM25 e instruções para avaliar com trec_eval.

Uso:
    uv run python avalia.py

Pré-requisitos:
    - Elasticsearch rodando
    - datasets/queries.tsv gerado
    - datasets/qrels.txt gerado
"""

import json
import os
import subprocess
from pathlib import Path

from elasticsearch import Elasticsearch

# ── Configurações ────────────────────────────────────────────────────────────
QUERIES_PATH  = "datasets/queries.tsv"
QRELS_PATH    = "datasets/qrels.txt"
RUN_PATH      = "datasets/run_bm25.txt"
ES_INDEX      = "hqs"
ES_URL        = "http://localhost:9200"
RESULTS_PER_QUERY = 100
SYSTEM_NAME   = "bm25_elasticsearch"
# ─────────────────────────────────────────────────────────────────────────────

es = Elasticsearch([ES_URL])


def buscar_docs(query: str, size: int = 100) -> list[tuple[str, float]]:
    res = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["issue_title^2", "issue_description", "comic_name^3"],
                    "type": "best_fields",
                }
            }
        },
        size=size,
    )
    return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]


def gerar_run_file():
    print("Carregando queries...")
    queries = []
    with open(QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split("\t")
            if len(partes) >= 2:
                queries.append({"qid": partes[0], "query": partes[1]})

    print(f"{len(queries)} queries — gerando run file...")

    output_path = Path(RUN_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, q in enumerate(queries):
            qid   = q["qid"]
            query = q["query"]

            try:
                resultados = buscar_docs(query, RESULTS_PER_QUERY)
            except Exception as e:
                print(f"[{i+1}] {qid}: ERRO — {e}")
                continue

            for rank, (doc_id, score) in enumerate(resultados, start=1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{SYSTEM_NAME}\n")

            print(f"[{i+1}/{len(queries)}] {qid}: {len(resultados)} resultados")

    print(f"\n✓ Run file salvo em: {RUN_PATH}")
    return output_path


def instalar_trec_eval():
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COMO INSTALAR O trec_eval (Linux/WSL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  git clone https://github.com/usnistgov/trec_eval
  cd trec_eval
  make
  
  sudo mv trec_eval /usr/local/bin/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def avaliar(qrels: str, run: str):
    print("\nRodando avaliação com trec_eval...\n")

    cmd_map = [
        "trec_eval",
        "-m", "map",
        "-m", "P.5",
        "-m", "P.10",
        "-m", "ndcg_cut.5",
        "-m", "ndcg_cut.10",
        "-m", "recip_rank",
        "-m", "recall.10",
        "-m", "Rprec",
        qrels, run
    ]

    cmd_full = ["trec_eval", "-q", "-m", "all_trec", qrels, run]

    try:
        result = subprocess.run(cmd_map, capture_output=True, text=True, check=True)
        print("── Métricas principais ──────────────────────────")
        print(result.stdout)

        result2 = subprocess.run(cmd_full, capture_output=True, text=True, check=True)
        full_path = Path("datasets/resultados_completos.txt")
        full_path.write_text(result2.stdout)
        print(f"Métricas completas salvas em: {full_path}")

    except FileNotFoundError:
        print("trec_eval não encontrado no PATH.")
        instalar_trec_eval()
        print("Depois de instalar, rode manualmente pelo Ubuntu:")
        print(f"  trec_eval -m map -m P.5 -m P.10 -m ndcg_cut.5 -m ndcg_cut.10 -m recip_rank -m recall.10 -m Rprec {qrels} {run}")

    except subprocess.CalledProcessError as e:
        print(f"Erro ao rodar trec_eval: {e.stderr}")


def main():
    run_path = gerar_run_file()
    avaliar(QRELS_PATH, str(run_path))

    print("\n── Resumo dos arquivos gerados ──────────────────")
    print(f"  Queries   : {QUERIES_PATH}")
    print(f"  Qrels     : {QRELS_PATH}")
    print(f"  Run file  : {RUN_PATH}")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
