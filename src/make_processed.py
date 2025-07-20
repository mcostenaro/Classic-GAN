# -*- coding: utf-8 -*-
"""
make_processed.py  —  Pré-processa os datasets H1, H2, H3
e salva versões baseline, clean e balanced em data/processed/.

Uso:
    python -m make_processed            # processa H1 H2 H3
    python -m make_processed --tags H2  # processa apenas H2
"""

import argparse
import pathlib as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from joblib import dump

ROOT = pl.Path(__file__).resolve().parents[1] # pasta do projeto
LAB  = ROOT / "data" / "labeled"

# ---------- Funções utilitárias ----------
def load_labeled(tag: str) -> pd.DataFrame:
    return pd.read_csv(LAB / f"{tag}_labeled.csv")

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    nunique = df.nunique()
    return df.loc[:, nunique.gt(1)]          # remove colunas sem variância

def scale(df: pd.DataFrame, method: str = "standard"):
    X = df.drop(columns=["labels"]).astype(np.float32)
    y = df["labels"].to_numpy()

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    out = pd.DataFrame(X_scaled, columns=X.columns)
    out["labels"] = y
    return out, scaler

def balance(df: pd.DataFrame, target: str = "labels", strategy: str = "random_over"):
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    y = df[target].to_numpy()

    sampler = SMOTE() if strategy == "smote" else RandomOverSampler()
    X_bal, y_bal = sampler.fit_resample(X, y)

    out = pd.DataFrame(X_bal, columns=df.columns.drop(target))
    out[target] = y_bal
    return out, sampler

def process_one(tag: str, out_root: pl.Path):
    df0 = load_labeled(tag)
    dfc = basic_clean(df0)

    out_root.mkdir(parents=True, exist_ok=True)

    # baseline -------------------------------------------------
    df_base, scaler_b = scale(dfc, "standard")
    df_base.to_csv(out_root / "baseline.csv", index=False)
    dump(scaler_b, out_root / "scaler_baseline.joblib")

    # clean ----------------------------------------------------
    df_clean, scaler_c = scale(dfc, "standard")
    df_clean.to_csv(out_root / "clean.csv", index=False)
    dump(scaler_c, out_root / "scaler_clean.joblib")

    # balanced -------------------------------------------------
    df_bal, sampler = balance(df_clean, strategy="random_over")
    df_bal.to_csv(out_root / "balanced.csv", index=False)
    dump(sampler, out_root / "sampler_balanced.joblib")

    print(f"[✓] {tag}: baseline / clean / balanced prontos.")

# ---------- Bloco principal ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pré-processa datasets H1, H2, H3.")
    parser.add_argument(
        "--tags", nargs="+", default=["H1", "H2", "H3"],
        help="Datasets a processar (padrão: H1 H2 H3)"
    )
    args = parser.parse_args()

    for tag in args.tags:
        out_dir = ROOT / "data" / "processed" / tag
        process_one(tag, out_dir)
