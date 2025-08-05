from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
import joblib

# -----------------------------------------------------------------------------
# Caminhos
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]              # pasta do projeto
DATA_LABELED = ROOT / "data" / "labeled"              # CSVs brutos rotulados


def prep_data(df_train, df_test):


    params_train = df_train.iloc[:, :2]
    X_aux = df_train.iloc[:, 2:-1].to_numpy(dtype=np.float32)
    y_aux = df_train["labels"].to_numpy()

    params_test = df_test.iloc[:, :2]
    X_test = df_test.iloc[:, 2:-1].to_numpy(dtype=np.float32)
    y_test = df_test["labels"].to_numpy()

    #balancing train dataset
    X_train, y_train = balance_train(X_aux, y_aux, "random_over")

    #normalize
    scaler = fit_scaler(X_train, method="spatial")
    X_train = transform(X_train, scaler)
    X_test = transform(X_test, scaler)

    return X_train, y_train, X_test, y_test, params_train, params_test


# -----------------------------------------------------------------------------
# Carregamento
# -----------------------------------------------------------------------------

def load_raw(tag: str):

    path = DATA_LABELED / f"{tag}_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Normalização
# -----------------------------------------------------------------------------

ScalerName = Literal["standard", "minmax", "spatial"]

def _get_scaler(method: ScalerName, *, copy: bool = True):
    if method == "standard":
        return StandardScaler(copy=copy)
    if method == "minmax":
        return MinMaxScaler(copy=copy)
    if method == "spatial":
        return Normalizer(norm="l2", copy=copy)
    raise ValueError("Método de escala inválido: " + repr(method))


def fit_scaler(X_train: np.ndarray, method: ScalerName = "standard"):
    scaler = _get_scaler(method)
    scaler.fit(X_train)
    return scaler

def transform(X, scaler):
    return scaler.transform(X)

# -----------------------------------------------------------------------------
# Balanceamento
# -----------------------------------------------------------------------------

BalanceName = Literal["random_over", "smote", None]


def balance_train(X, y, strategy: str = "random_over", random_state: int = 42):
    
    if strategy is None:
        return X, y

    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
    else:
        sampler = RandomOverSampler(random_state=random_state)

    X_bal, y_bal = sampler.fit_resample(X, y)
    return X_bal, y_bal


def main():
    print("[TESTE] Rodando pipeline de pré-processamento...")

    # Carregar datasets de exemplo (alterar nomes conforme os seus arquivos rotulados)
    df1 = load_raw("H1")  # Ex.: "H1_labeled.csv"
    df2 = load_raw("H3")  # Ex.: "H3_labeled.csv"
    df_test = load_raw("H2")  # Ex.: "H2_labeled.csv"


    df_train = pd.concat([df1, df2], ignore_index=True)

    # Rodar pipeline
    X_train, y_train, X_test, y_test, _, _ = prep_data(df_train, df_test)

    # Exibir shapes para conferir se bate com o esperado
    print(f"Treino: {X_train.shape}, {y_train.shape}")
    print(f"Teste:  {X_test.shape}, {y_test.shape}")

    # Exibir contagem de rótulos
    print("Distribuição treino:", np.bincount(y_train))
    print("Distribuição teste:", np.bincount(y_test))

    print("[TESTE] Pré-processamento finalizado com sucesso.")


if __name__ == "__main__":
    main()