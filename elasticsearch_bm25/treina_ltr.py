import numpy as np
import pandas as pd
import xgboost as xgb

DATASET_PATH = "datasets/dataset_ltr.csv"
MODEL_PATH   = "datasets/ltr_model.json"
FEATURES     = [
    "bm25_score", "bm25_rank",
    "semantic_score", "semantic_rank",
    "title_bm25_score", "comic_name_bm25_score", "description_bm25_score",
    "rank_diff",
]
TRAIN_RATIO  = 0.8
SEED         = 42


def get_qids_from_file(filepath):
    qids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            qids.add(line.strip().split("\t")[0])
    return qids

def split_by_physical_files(df: pd.DataFrame):
    train_qids = get_qids_from_file("datasets/queries_treino.tsv")
    val_qids   = get_qids_from_file("datasets/queries_val.tsv")
    test_qids  = get_qids_from_file("datasets/queries_teste.tsv")
    
    df_train = df[df["query_id"].isin(train_qids)]
    df_val   = df[df["query_id"].isin(val_qids)]
    df_test  = df[df["query_id"].isin(test_qids)]
    
    return df_train, df_val, df_test


def build_dmatrix(df: pd.DataFrame) -> xgb.DMatrix:
    df_sorted = df.sort_values("query_id")
    X = df_sorted[FEATURES].values
    y = df_sorted["relevance"].values

    groups = df_sorted.groupby("query_id", sort=False)["doc_id"].count().values

    dmatrix = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    dmatrix.set_group(groups)
    return dmatrix


def main():
    print(f"Carregando Dataset de Treino: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df):,} linhas | {df['query_id'].nunique()} queries únicas\n")

    df_train, df_val, df_test = split_by_physical_files(df)
    print(f"Treino:    {df_train['query_id'].nunique()} queries ({len(df_train):,} pares)")
    print(f"Validação: {df_val['query_id'].nunique()} queries ({len(df_val):,} pares)")
    print(f"Teste:     {df_test['query_id'].nunique()} queries ({len(df_test):,} pares)\n")

    dtrain = build_dmatrix(df_train)
    dval   = build_dmatrix(df_val)
    dtest  = build_dmatrix(df_test)

    params = {
        "objective":   "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta":         0.1,
        "max_depth":   6,
        "subsample":   0.8,
        "seed":        SEED,
    }

    print("Training LambdaMART...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=20,
        early_stopping_rounds=10
    )

    model.save_model(MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH}")

    print("\nImportância das features:")
    importances = model.get_score(importance_type="gain")
    for feat, value in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<20} {value:.4f}")


if __name__ == "__main__":
    main()
