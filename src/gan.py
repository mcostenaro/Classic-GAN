import torch, torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"   

version = "balanced"        #"baseline", "clean", "balanced"

#function to load data_set of the chosen hamiltonians to get data augmentation
def data_loader(H_list, version):
    frames = []
    for h in H_list:
        frames.append(pd.read_csv(DATA_DIR / h / f"{version}.csv"))
        
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    #numpy array
    X = df.drop(columns="labels").values.astype(np.float32)
    y = df["labels"].values.astype(np.int64)

    return X, y

#function to balance the concatenated dataset
def balance_df(X, y, target_per_class=300):
    df = pd.DataFrame(X)
    df["labels"] = y

    dfs = []
    for label in df["labels"].unique():
        df_label = df[df["labels"] == label]
        count = len(df_label)

        if count < target_per_class:
            sampled = resample(df_label, replace=True, n_samples=target_per_class, random_state=42)
        else:
            sampled = resample(df_label, replace=False, n_samples=target_per_class, random_state=42)

        dfs.append(sampled)

    df_bal = pd.concat(dfs, ignore_index=True)

    X_bal = df_bal.drop(columns="labels").values.astype(np.float32)
    y_bal = df_bal["labels"].values.astype(np.int64)

    return X_bal, y_bal

#split into training and testing data:
def split_data(X, y, test_size=0.1):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


# loading 
X, y = data_loader(["H1", "H2"], version="baseline")

# balancing
X_bal, y_bal = balance_df(X, y, target_per_class=300)

# spliting
X_train, X_val, y_train, y_val = split_data(X_bal, y_bal, test_size=0.1)


class Generator(nn.Module):
    def __init__(self, latent_dim, n_features, n_labels, label_emb_dim=4):
        super().__init__()
        self.label_embed = nn.Embedding(n_labels, label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_features),
            nn.Tanh()  # saída entre -1 e 1 se os dados estiverem normalizados
        )

    def forward(self, z, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([z, label_vec], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_features, n_labels, label_emb_dim=4):
        super().__init__()
        self.label_embed = nn.Embedding(n_labels, label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(n_features + label_emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output: probabilidade de ser real
        )

    def forward(self, x, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([x, label_vec], dim=1)
        return self.model(x)