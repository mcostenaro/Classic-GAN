#imports
import pandas as pd, numpy as np, pathlib as pl, argparse
from sklearn.preprocessing import normalize, MinMaxScaler
from joblib import dump

#colecting data
ROOT = pl.Path(__file__).resolve().parents[1]
LAB  = ROOT / "data" / "labeled"
OUT  = ROOT / "data" / "processed"


def process(tag, balance=False):
    df  = pd.read_csv(LAB / f"{tag}_labeled.csv")

    # ------------------ oversample per fase -----------------
    if balance:
        n_max = df["labels"].value_counts().max()
        df = pd.concat([g.sample(n_max, replace=True, random_state=42)
                        for _, g in df.groupby("labels")], ignore_index=True)

    X = df.iloc[:, 2:26].to_numpy(np.float32)   # 24 correlações
    y = df["labels"].to_numpy(np.int64)

    X = normalize(X, norm="l2")                # spatial-sign
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X = scaler.transform(X)

    dest = OUT / tag
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "X.npy", X)
    np.save(dest / "y.npy", y)
    dump(scaler, dest / "scaler.joblib")
    print(f"[OK] {tag}: {X.shape}  salvo em {dest}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=["H1","H2","H3"],
                    help="Hamiltoniano a processar")
    ap.add_argument("--balance", action="store_true",
                    help="oversample para igualar fases")
    args = ap.parse_args()
    process(args.tag, args.balance)