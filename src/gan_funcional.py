# train_cgan.py
# ============================================================
#  cGAN baseline (tabular) – pronto para rodar só em CPU
#  Altere hiperparâmetros na seção CONFIG abaixo.
# ============================================================

import random, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import json, datetime, os

# ------------- CONFIG --------------------------------------------------------
CONFIG = dict(
    H_LIST           = ["H1", "H2"],   # quais Hamiltonianos usar
    CSV_VERSION      = "baseline",     # sub-pasta (baseline / clean / balanced)
    TARGET_PER_CLASS = 500,            # amostras por classe após balanceamento
    LATENT_DIM       = 64,
    BATCH_SIZE       = 80,            # pode baixar p/ CPU lenta
    EPOCHS           = 100,
    LR               = 2e-4,
    SEED             = 42,
)
# -----------------------------------------------------------------------------

ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("runs/"+ts, exist_ok=True)
json.dump(CONFIG, open(f"runs/{ts}/config.json","w"), indent=2)


# ---------- utilidades -------------------------------------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_concat(H_list, version, data_dir):
    frames = []
    for h in H_list:
        df = pd.read_csv(data_dir / h / f"{version}.csv")
        df = df.drop(df.columns[:2], axis=1)          # remove colunas idx/Unnamed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    X = df.drop(columns="labels").values.astype(np.float32)
    y = df["labels"].values.astype(np.int64)
    return X, y

def balance_df(X, y, target):
    df = pd.DataFrame(X); df["labels"] = y
    parts = []
    for lbl, part in df.groupby("labels"):
        parts.append(
            resample(part,
                     replace=len(part) < target,
                     n_samples=target,
                     random_state=CONFIG["SEED"])
        )
    df_bal = pd.concat(parts, ignore_index=True)
    Xb = df_bal.drop(columns="labels").values.astype(np.float32)
    yb = df_bal["labels"].values.astype(np.int64)
    return Xb, yb

def make_loader(X, y, bs):
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=bs, shuffle=True,
                      drop_last=True, num_workers=0)

# ---------- modelos ----------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, n_feat, n_lbl, emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(n_lbl, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + emb_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_feat)        # saída linear (dados em z-score)
        )
    def forward(self, z, y):
        return self.net(torch.cat([z, self.emb(y)], dim=1))

class Discriminator(nn.Module):
    def __init__(self, n_feat, n_lbl, emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(n_lbl, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_feat + emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, self.emb(y)], dim=1))

# ---------- treino -----------------------------------------------------------
def train():
    seed_all(CONFIG["SEED"])

    device = torch.device("cpu")                 # força CPU

    ROOT = Path(__file__).resolve().parents[1]   # <repo>/src/train_cgan.py
    data_dir = ROOT / "data" / "processed"

    # 1) Dados
    X, y = load_concat(CONFIG["H_LIST"], CONFIG["CSV_VERSION"], data_dir)
    X, y = balance_df(X, y, CONFIG["TARGET_PER_CLASS"])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=CONFIG["SEED"]
    )
    n_labels  = 8       # do paper do mahlow (entrada do gerador para ter 8 possíveis fases)
    labels_present = torch.tensor(np.unique(y), device=device)      #pega a quantidade de labels dos H presentes em H_List
    n_features = X.shape[1]
    loader = make_loader(X_tr, y_tr, CONFIG["BATCH_SIZE"])

    # 2) Modelos/otimizadores
    G = Generator(CONFIG["LATENT_DIM"], n_features, n_labels).to(device)
    D = Discriminator(n_features, n_labels).to(device)
    opt_G = Adam(G.parameters(), lr=CONFIG["LR"], betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=CONFIG["LR"], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # 3) Loop de épocas
    for ep in range(1, CONFIG["EPOCHS"] + 1):
        g_loss_ep, d_loss_ep = 0., 0.
        for real_x, real_y in loader:
            bs = real_x.size(0)
            real_x, real_y = real_x.to(device), real_y.to(device)
            lbl_real = torch.ones(bs, 1, device=device)
            lbl_fake = torch.zeros(bs, 1, device=device)

            # -------- Discriminador --------
            D.zero_grad()
            # reais
            out_real = D(real_x, real_y)
            loss_real = criterion(out_real, lbl_real)
            # fakes
            z = torch.randn(bs, CONFIG["LATENT_DIM"], device=device)
            idx = torch.randint(0, len(labels_present), (bs,), device=device)
            y_g = labels_present[idx]          # agora só rótulos vistos pelo D
            fake_x = G(z, y_g).detach()
            out_fake = D(fake_x, y_g)
            loss_fake = criterion(out_fake, lbl_fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward(); opt_D.step()

            # -------- Gerador --------
            G.zero_grad()
            z = torch.randn(bs, CONFIG["LATENT_DIM"], device=device)
            y_g = torch.randint(0, n_labels, (bs,), device=device)
            fake_x = G(z, y_g)
            out = D(fake_x, y_g)
            loss_G = criterion(out, lbl_real)
            loss_G.backward(); opt_G.step()

            g_loss_ep += loss_G.item(); d_loss_ep += loss_D.item()

        print(f"Epoch {ep:03d} | D_loss {d_loss_ep/len(loader):.4f} "
              f"| G_loss {g_loss_ep/len(loader):.4f}")

    # 4) Salva
    ck = Path("checkpoints"); ck.mkdir(exist_ok=True)
    torch.save(G.state_dict(), ck / "G_final.pt")
    torch.save(D.state_dict(), ck / "D_final.pt")
    print("\n✓ Modelos salvos em checkpoints/")

# ---------- gerar amostras rápidas ------------------------------------------
def quick_sample(num=5):
    """Carrega G_final.pt (já treinado) e gera <num> amostras por label."""
    ck = Path("checkpoints/G_final.pt")
    if not ck.exists():
        raise FileNotFoundError("Treine primeiro! checkpoints/G_final.pt não existe.")
    # meta-dados
    n_feat    = None
    n_labels  = None
    # lemos um CSV só p/ pegar dims
    ROOT = Path(__file__).resolve().parents[1]
    sample_csv = pd.read_csv(ROOT / "data" / "processed" /
                             CONFIG["H_LIST"][0] / f"{CONFIG['CSV_VERSION']}.csv")
    n_feat = sample_csv.shape[1] - 3          # -2 cols dropadas -1 label

    n_labels = len(np.unique(sample_csv["labels"]))

    G = Generator(CONFIG["LATENT_DIM"], n_feat, n_labels)
    G.load_state_dict(torch.load(ck, map_location="cpu"))
    G.eval()

    z = torch.randn(num, CONFIG["LATENT_DIM"])
    y = torch.randint(0, n_labels, (num,))
    with torch.no_grad():
        samples = G(z, y).numpy()
    print("Amostras geradas (primeiras linhas):\n", samples[:3])

# -----------------------------------------------------------------------------    
if __name__ == "__main__":
    train()
    # quick_sample(10)  # descomente p/ verificar geração após treino
