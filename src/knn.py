"""knn_phase_plot.py — Classifica fases quânticas com k‑NN
-------------------------------------------------------
• Mantém parâmetros físicos sem escala.
• Alinha features entre H1/H2/H3.
• **Novo estilo de plot:** marcadores quadrados preenchidos (marker='s', s=20) sem borda → diagramas sólidos como no Mahlow.
"""

import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_raw, prep_data

# ------------------------------------------------------------
# 1) Definições globais
# ------------------------------------------------------------
PHASES = [
    "Haldane", "Trimer", "Ferro", "Dimer", "LD", "XY1", "Neel", "XY2"
]
COLORS = [
    "red", "purple", "blue", "green", "yellow", "cyan", "olive", "black"
]
PHASE_COLOR = {i: COLORS[i] for i in range(len(PHASES))}

ROOT = pl.Path(__file__).resolve().parents[1]
DATA_LABELED = ROOT / "data" / "Labeled"
EXTRA = ROOT / "extra"

CONFIG = dict(
    H_idx= 1,           # 1→H1, 2→H2, 3→H3
    K=50,              # vizinhos
    legend=True,
)

# ------------------------------------------------------------
# 3) Dados
# ------------------------------------------------------------

#XXZ Single-Ion Anisotropy
df_H1= pd.read_csv(DATA_LABELED / "H1_labeled.csv")
df_H1_theory = pd.DataFrame(EXTRA / "pontos_plot_XXZ.csv")

#XXZ Bond-alternating 
df_H2 = pd.read_csv(DATA_LABELED / "H2_labeled.csv")
df_H2_theory = pd.DataFrame(EXTRA / "data_paper_ferro_bond.csv")

#XXZ Bilinear-Biquadratic
df_H3 = pd.read_csv(DATA_LABELED / "H3_labeled.csv")
df_H3_theory = pd.DataFrame([0.25, 0.5, 1.25, 1.75])


###
#_list = [df_H1, df_H2, df_H3]

def set_train_test(H_list, H_test : str):
    H_list = [df_H1, df_H2, df_H3]

    df1 = 

    # Rodar pipeline
    X_train, y_train, X_test, y_test, params_train, params_test = prep_data(df1, df2, df_test)

    return X_train, y_train, X_test, y_test, params_test

# ------------------------------------------------------------
# 4) KNN
# ------------------------------------------------------------

def knn_predict(X_train, y_train, X_test, k):
    model = KNeighborsClassifier(k)
    model.fit(X_train, y_train)
    return model.predict(X_test)

# ------------------------------------------------------------
# 5) Anotações
# ------------------------------------------------------------

def annotate_h1():
    plt.text( 2, -1, "Néel", fontsize=12)
    plt.text(-0.6, 2, "Large D", fontsize=12)
    plt.text(-2.7, -1, "Ferro", color="white", fontsize=12)
    plt.text(-0.8, 0, "XY1", fontsize=10)
    plt.text(-0.4,-1.5,"XY2", fontsize=10)
    plt.text(0.15, 0, "Haldane", fontsize=10)
    plt.xlabel(r"$J_z$", fontsize=14)
    plt.ylabel("D", fontsize=14)

def annotate_h2():
    plt.text(1.7,0.3, "Néel", fontsize=12)
    plt.text(0.2,0.5, "Dimer", fontsize=12)
    plt.text(-1.49,0.5,"Ferro", color="white", fontsize=12)
    plt.text(-0.7,0.25,"XY1", fontsize=12)
    plt.text(0.2,0.1, "Haldane", fontsize=12)
    plt.xlabel("Δ", fontsize=14)
    plt.ylabel("δ", fontsize=14)

def annotate_h3():
    plt.text(0,0.02, "Hald.", fontsize=12)
    plt.text(0.26,0.02,"Trim.", fontsize=12)
    plt.text(0.75,0.02,"Ferro", fontsize=12)
    plt.text(1.35,0.02,"Dimer", fontsize=12)
    plt.text(1.75,0.02,"Hald.", fontsize=12)
    plt.xlabel("θ", fontsize=14)
    plt.ylim(-0.02,0.05)

# ------------------------------------------------------------
# 6) Plot
# ------------------------------------------------------------

def plot_diagram(params_test, y_pred, h_idx):
    df_xxz, df_bond, df_bilin = load_theoretical()
    x, y = params_test.iloc[:,0], params_test.iloc[:,1]

    # Pontos k‑NN — marcadores quadrados preenchidos
    for lbl in range(len(PHASES)):
        mask = y_pred == lbl
        if mask.any():
            plt.scatter(
                x[mask], y[mask],
                s=20, marker='s', edgecolors='none',
                color=PHASE_COLOR[lbl], label=PHASES[lbl]
            )

    # Transições teóricas
    if h_idx == 1:
        plt.plot(df_xxz[0], df_xxz[1], ls='None', marker='o', mec='black', mfc='white', ms=10, label='Transitions')
        annotate_h1()
    elif h_idx == 2:
        plt.plot(df_bond[0], df_bond[1], ls='None', marker='o', mec='black', mfc='white', ms=10, label='Transitions')
        plt.plot(0,0.1, lw=0, marker='|', ms=45, mec='black', c='white')
        annotate_h2()
    else:
        plt.plot(df_bilin[0], df_bilin[1], lw=0, marker='|', ms=12, mec='black', c='white', label='Transitions')
        annotate_h3()

    if CONFIG["legend"]:
        plt.legend(bbox_to_anchor=(1.02,-0.15), ncol=3, fontsize=8)
    plt.tight_layout()

# ------------------------------------------------------------
# 7) Main
# ------------------------------------------------------------

def main():

    df1 = load_raw("H1")  # Ex.: "H1_labeled.csv"
    df2 = load_raw("H3")  # Ex.: "H3_labeled.csv"
    df_test = load_raw("H2")  # Ex.: "H2_labeled.csv"


    X_tr, y_tr, X_te, y_te, p_te = load_experimental()
    y_pred = knn_predict(X_tr, y_tr, X_te, CONFIG["K"])

    print("Acurácia:", accuracy_score(y_te, y_pred))
    print("\nRelatório:\n", classification_report(y_te, y_pred))

    plt.figure(figsize=(6,5))
    plot_diagram(p_te, y_pred, CONFIG["H_idx"])
    plt.title(f"k‑NN (k={CONFIG['K']}) — H{CONFIG['H_idx']}")
    plt.show()

if __name__ == "__main__":
    main()
