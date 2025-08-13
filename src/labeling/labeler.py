import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import pathlib as pl

ROOT  = pl.Path(__file__).resolve().parents[1]   # volta para gan_fases/
DATA  = ROOT / "data"                            # pasta onde estão os .xls


fases = ['Haldane', 'Trimer', 'Ferro', 'Dimer', 'LD', 'XY1', 'Neel', 'XY2'] #Definir a ordem das fases
colors = ['red', 'purple', 'blue', 'green', 'yellow', 'cyan', 'olive', 'black'] #Definir cores de cada fase


dfXXZ = pd.DataFrame(pd.read_excel(DATA / "dfXXZ_final.xls"))
dfBond = pd.DataFrame(pd.read_excel(DATA / "dfBond_final.xls"))
dfBilinear = pd.DataFrame(pd.read_excel(DATA / "dfBilinear_final.xls"))
#dfXXZTeorico = pd.DataFrame(pd.read_csv('pontos_plot_XXZ.csv', header=None))
#dfBondTeorico = pd.DataFrame(pd.read_csv('data_paper_ferro_bond.csv', header=None))
#dfBilinearTeorico = pd.DataFrame([0.25, 0.5, 1.25, 1.75])

# ROTULAR XXZ
rotuladorXXZ = []
for x in range(0, len(dfXXZ)):
    X = dfXXZ.values[x, 0]
    y = dfXXZ.values[x, 1]

#     FASES FERRO/LD
    if X < -1.8375250:
        if y < (-0.1009)*(X**2) - 1.6701*X - 1.329:
            rotuladorXXZ.append(2)
        else:
            rotuladorXXZ.append(4)

#     FASES FERRO/XY
    elif X >= -1.837525 and X < -0.278:
        if y >-0.0741*X**3 - 0.3014*X**2 - 0.872*X + 0.3499:
            rotuladorXXZ.append(4)
        elif y < -5.1127*X**5 - 27.865*X**4 - 57.426*X**3 - 55.858*X**2 - 27.638*X - 6.4824:
            rotuladorXXZ.append(2)
        else:
            rotuladorXXZ.append(5)

    elif X >= -0.278 and X < 0:
        if y >-0.0741*X**3 - 0.3014*X**2 - 0.872*X + 0.3499:
            rotuladorXXZ.append(4)
        elif y < -5.1127*X**5 - 27.865*X**4 - 57.426*X**3 - 55.858*X**2 - 27.638*X - 6.4824:
            rotuladorXXZ.append(2)
        elif y < 4.595*X**2 + 1.393*X - 2.007:
            rotuladorXXZ.append(7)
        else:
            rotuladorXXZ.append(5)

#     FASES LD/ HALDANE
    elif X >= 0 and X < 3.28405:
        if y > 0.0807*X**2 + 0.5418*X + 0.3465:
            rotuladorXXZ.append(4)
        elif y > -0.0462*X**3 + 0.154*X**2 + 1.5213*X - 2.0196:
            rotuladorXXZ.append(0)
        else:
            rotuladorXXZ.append(6)

#     FASES LD/NEEL
    elif X >= 3.28405 and X <= 4:
        if y > 1.0906*X - 0.583:
            rotuladorXXZ.append(4)
        else:
            rotuladorXXZ.append(6)
dfXXZ["labels"] = rotuladorXXZ

######################################################################
# ROTULAR BOND
rotuladorBond = []
for x in range(0, len(dfBond)):
    X = dfBond.values[x, 0]
    y = dfBond.values[x, 1]

#     FASE FERRO
    if X < -1:
        rotuladorBond.append(2)

#     FASE XY/ DIMER
    elif X >= -1 and X < 0:
        if y > 0.1435*X**2 - 0.6447*X + 0.2197:
            rotuladorBond.append(3)
        else:
            rotuladorBond.append(5)

#     FASE HALDANE/DIMER
    elif X >= 0 and X < 1:
        if y > 0.0043*X**3 - 0.0384*X**2 + 0.0631*X + 0.2317:
            rotuladorBond.append(3)
        else:
            rotuladorBond.append(0)   

#     FASE HALDANE/NEEL
    elif X >= 1.0 and X <= 1.2:
        if y > -0.0872*X**2 + 0.6067*X - 0.2264:
            rotuladorBond.append(3)
        elif y > -247.1*X**3 + 826.12*X**2 - 921.85*X + 343.48:
            rotuladorBond.append(6)
        else:
            rotuladorBond.append(0)

#     FASE NEEL/DIMER
    elif X > 1.2 and X <= 2.5:
        if y > -0.0872*X**2 + 0.6067*X - 0.2264:
            rotuladorBond.append(3)
        else:
            rotuladorBond.append(6)

rotuladorBondDF = pd.DataFrame(rotuladorBond)
dfBond["labels"]= rotuladorBondDF

######################################################################
# ROTULAR BILINEAR

rotuladorBilinear = []
zeros = []
yf = list()
for k in range(0, len(dfBilinear)):
    yf.append(0)
for x in range(0, len(dfBilinear)):
#     FASE HALDANE
    if dfBilinear.values[x,0] <= 0.25 and dfBilinear.values[x,0] >= 0 or dfBilinear.values[x,0] > 1.75 and dfBilinear.values[x,0] <= 2:
        rotuladorBilinear.append(0)
#     FASE TRIMER
    elif dfBilinear.values[x,0] > 0.25 and dfBilinear.values[x,0] <= 0.5:
        rotuladorBilinear.append(1)
#     FASE FERRO
    elif dfBilinear.values[x,0] > 0.5 and dfBilinear.values[x,0] <= 1.25:
        rotuladorBilinear.append(2)
#     FASE DIMER
    else:
        rotuladorBilinear.append(3)

dfBilinear["labels"]= pd.DataFrame(rotuladorBilinear)