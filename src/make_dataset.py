# src/make_dataset.py
import pandas as pd
import pathlib as pl
import Rotulador as R  # já carrega dfXXZ, dfBond, dfBilinear com labels

ROOT = pl.Path(__file__).resolve().parents[1]  # volta para gan_fases/
DATA = ROOT / "data"

def save(df, name):
    out = DATA / f"{name}_labeled.csv"
    df.to_csv(out, index=False)
    print(f"[OK] {name:<2} → {len(df):>5} linhas  →  {out.name}")

def main():
    save(R.dfXXZ,      "H1")   # XXZ uniaxial
    save(R.dfBond,     "H2")   # bond-alternating
    save(R.dfBilinear, "H3")   # bilinear-biquadrático

if __name__ == "__main__":
    main()
