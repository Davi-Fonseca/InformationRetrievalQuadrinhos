"""
Gera run files para todas as combinações do Grid Search e avalia com trec_eval.

Uso:
    uv run python avalia.py
"""

import itertools
import os
import subprocess
from pathlib import Path
from elasticsearch import Elasticsearch

# ── Configurações ────────────────────────────────────────────────────────────
QUERIES_PATH  = "datasets/queries.tsv"
QRELS_PATH    = "datasets/qrels.txt"
RUNS_DIR      = "datasets/runs"       # Pasta para organizar os 18 run files
ES_URL        = "http://localhost:9200"
RESULTS_PER_QUERY = 100               # Padrão top-100 para avaliação de RI
# ─────────────────────────────────────────────────────────────────────────────

es = Elasticsearch([ES_URL])


def buscar_docs(es: Elasticsearch, query: str, index_name: str, size: int = 100) -> list[tuple[str, float]]:
    """Busca documentos no índice específico da configuração atual."""
    res = es.search(
        index=index_name,
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


def carregar_queries() -> list[dict]:
    """Lê e armazena as queries do arquivo TSV."""
    queries = []
    if not os.path.exists(QUERIES_PATH):
        raise FileNotFoundError(f"Arquivo de queries não encontrado em {QUERIES_PATH}")
        
    with open(QUERIES_PATH, encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split("\t")
            if len(partes) >= 2:
                queries.append({"qid": partes[0], "query": partes[1]})
    return queries


def gerar_run_file(queries: list[dict], sw: bool, proc: str, sim: str) -> Path:
    """Gera um run file TREC específico para uma configuração do grid."""
    sw_str = "yes" if sw else "no"
    index_name = f"hqs_sw_{sw_str}_proc_{proc}_sim_{sim}"
    system_name = f"grid_{sw_str}_{proc}_{sim}"
    
    run_file_name = f"run_sw_{sw_str}_proc_{proc}_sim_{sim}.txt"
    output_path = Path(RUNS_DIR) / run_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for q in queries:
            qid = q["qid"]
            query = q["query"]

            try:
                resultados = buscar_docs(es, query, index_name, RESULTS_PER_QUERY)
            except Exception as e:
                # Caso o índice correspondente não tenha sido criado no Elasticsearch
                print(f"Erro ao buscar no índice {index_name}: {e}")
                return None

            # Formato TREC run: qid Q0 doc_id rank score system_name
            for rank, (doc_id, score) in enumerate(resultados, start=1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\t{system_name}\n")

    return output_path


def parse_trec_eval_output(stdout_text: str) -> dict:
    """Extrai as métricas de interesse do output de texto do trec_eval."""
    metrics = {"map": 0.0, "P_10": 0.0, "ndcg_cut_10": 0.0}
    for line in stdout_text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            metric_name = parts[0]
            metric_value = parts[2]
            if metric_name in metrics:
                metrics[metric_name] = float(metric_value)
    return metrics


def avaliar_config(qrels: str, run_path: Path) -> dict:
    """Executa o trec_eval para o run file gerado e retorna as métricas."""
    cmd = ["trec_eval", "-m", "map", "-m", "P.10", "-m", "ndcg_cut.10", qrels, str(run_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return parse_trec_eval_output(result.stdout)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None


def main():
    print("1. Carregando queries...")
    try:
        queries = carregar_queries()
        print(f"✓ {len(queries)} queries carregadas com sucesso.")
    except Exception as e:
        print(f"ERRO: {e}")
        return

    # Definindo os hiperparâmetros do Grid Search (idêntico ao indexa.py)
    sw_options = [True, False]
    proc_options = ["none", "stemming", "lemmatization"]
    sim_options = ["bm25", "jelinek_mercer", "dirichlet"]

    combinations = list(itertools.product(sw_options, proc_options, sim_options))
    print(f"\n2. Iniciando avaliação de {len(combinations)} combinações do Grid Search...")

    results_grid = []
    trec_eval_available = True

    for idx, (sw, proc, sim) in enumerate(combinations, start=1):
        sw_str = "Removed" if sw else "Unchanged"
        print(f"[{idx}/{len(combinations)}] Processando -> Stopwords: {sw_str} | Proc: {proc} | Sim: {sim}")
        
        # Gera o arquivo de run individual para esta configuração
        run_path = gerar_run_file(queries, sw, proc, sim)
        
        if run_path and trec_eval_available:
            # Roda a avaliação usando o binário trec_eval
            metrics = avaliar_config(QRELS_PATH, run_path)
            if metrics is not None:
                results_grid.append({
                    "sw": sw_str,
                    "proc": proc,
                    "sim": sim,
                    "map": metrics["map"],
                    "p10": metrics["P_10"],
                    "ndcg10": metrics["ndcg_cut_10"]
                })
            else:
                if idx == 1:
                    print("\n[Aviso] trec_eval não encontrado no PATH ou qrels ausente. Apenas os arquivos de run serão gerados.")
                    trec_eval_available = False

    # 3. Exibe o relatório final consolidado se o trec_eval foi executado
    if results_grid:
        # Ordena o resultado pelo NDCG@10 de forma decrescente para destacar os melhores setups
        results_grid.sort(key=lambda x: x["ndcg10"], reverse=True)

        print("\n" + "═" * 85)
        print(f"{'RANK':<5} | {'STOPWORDS':<10} | {'PROCESSAMENTO':<15} | {'SIMILARIDADE':<15} | {'MAP':<8} | {'P@10':<8} | {'NDCG@10':<8}")
        print("─" * 85)
        for rank, res in enumerate(results_grid, start=1):
            print(f"{rank:<5} | {res['sw']:<10} | {res['proc']:<15} | {res['sim']:<15} | {res['map']:.4f} | {res['p10']:.4f} | {res['ndcg10']:.4f}")
        print("═" * 85)
        print(f"✓ Todos os logs detalhados de run salvos na pasta: {RUNS_DIR}\n")
    else:
        print(f"\n✓ Execução concluída. Os 18 arquivos de run foram gerados e salvos em '{RUNS_DIR}'.")
        print(f"Para avaliar manualmente, use: trec_eval {QRELS_PATH} {RUNS_DIR}/<nome_do_arquivo>.txt")


if __name__ == "__main__":
    main()