# train_cgan_augmented.py
# ============================================================
#  cGAN tabular – pronto para rodar só em CPU
#  Integração mínima p/ chamar do notebook: run_experiment(config, X, y)
# ============================================================

import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

# (mantém import caso você ainda rode via carregar arquivos)
from preprocessing import load_raw, balance_train

# ------------- CONFIG DEFAULT ------------------------------------------------
CONFIG = {
    "H_LIST": ["H1", "H2"],    # usado só no modo legado (sem X,y)
    "LATENT_DIM": 64,
    "BATCH_SIZE": 80,
    "EPOCHS": 100,
    "LR": 2e-4,
    "SEED": 42,
}

# -----------------------------------------------------------------------------
# 1) Utilitários
# -----------------------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -----------------------------------------------------------------------------
# 2) Modelos da GAN
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, n_feat, n_lbl, emb_dim=4):
        super().__init__()
        self.emb = nn.Embedding(n_lbl, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + emb_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_feat)
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

# -----------------------------------------------------------------------------
# 3) Treino da cGAN
# -----------------------------------------------------------------------------
def _encode_labels(y: np.ndarray):
    classes = np.unique(y)
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_enc = np.vectorize(cls2idx.get)(y).astype(np.int64)
    return y_enc, classes  # classes serve como decodificador

def _prepare_from_arrays(X, y):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    y_enc, classes = _encode_labels(y)
    n_feat = X.shape[1]
    n_lbl = len(classes)
    return X, y_enc, n_feat, n_lbl, classes

def _prepare_from_files(cfg):
    dfs = [load_raw(h) for h in cfg['H_LIST']]
    df_all = pd.concat(dfs, ignore_index=True)
    df_feats = df_all.iloc[:, 2:].copy()
    X = df_feats.drop(columns='labels').to_numpy(dtype=np.float32)
    y = df_feats['labels'].to_numpy()
    # balanceia ANTES de encodar (tanto faz), mas encodamos após obter y_bal
    from preprocessing import balance_train
    Xb, yb = balance_train(X, y, strategy='random_over')
    yb_enc, classes = _encode_labels(yb)
    n_feat = Xb.shape[1]
    n_lbl = len(classes)
    return Xb.astype(np.float32), yb_enc, n_feat, n_lbl, classes


def train_gan(config_override: dict = None, X=None, y=None):
    """
    Treina a cGAN.
    - Se X,y forem fornecidos: usa diretamente (modo notebook).
    - Caso contrário: carrega via H_LIST (modo legado).
    Retorna: dict com caminhos, dims e objetos úteis.
    """
    cfg = CONFIG.copy()
    if config_override:
        cfg.update(config_override)

    seed_all(cfg['SEED'])
    device = torch.device('cpu')

    # >>> NOVO: escolhe fonte dos dados
    if X is not None and y is not None:
        Xb, yb, n_feat, n_lbl, classes = _prepare_from_arrays(X, y)
    else:
        Xb, yb, n_feat, n_lbl, classes = _prepare_from_files(cfg)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xb), torch.from_numpy(yb)),
        batch_size=cfg['BATCH_SIZE'], shuffle=True, drop_last=True
    )

    # modelos/otimizadores
    G = Generator(cfg['LATENT_DIM'], n_feat, n_lbl).to(device)
    D = Discriminator(n_feat, n_lbl).to(device)
    opt_G = Adam(G.parameters(), lr=cfg['LR'], betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=cfg['LR'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for ep in range(1, cfg['EPOCHS'] + 1):
        g_loss, d_loss = 0.0, 0.0
        for real_x, real_y in loader:
            real_x, real_y = real_x.to(device), real_y.to(device)
            bs = real_x.size(0)
            lbl_real = torch.ones(bs, 1, device=device)
            lbl_fake = torch.zeros(bs, 1, device=device)

            # D
            D.zero_grad()
            out_r = D(real_x, real_y)
            loss_r = criterion(out_r, lbl_real)
            z = torch.randn(bs, cfg['LATENT_DIM'], device=device)
            y_g = torch.randint(0, n_lbl, (bs,), device=device)
            fake_x = G(z, y_g).detach()
            out_f = D(fake_x, y_g)
            loss_f = criterion(out_f, lbl_fake)
            loss_D = 0.5 * (loss_r + loss_f)
            loss_D.backward(); opt_D.step()

            # G
            G.zero_grad()
            z = torch.randn(bs, cfg['LATENT_DIM'], device=device)
            y_g = torch.randint(0, n_lbl, (bs,), device=device)
            fake_x = G(z, y_g)
            out_g = D(fake_x, y_g)
            loss_G = criterion(out_g, lbl_real)
            loss_G.backward(); opt_G.step()

            g_loss += loss_G.item()
            d_loss += loss_D.item()

        print(f"Epoch {ep:03d} | D_loss {d_loss/len(loader):.4f} | G_loss {g_loss/len(loader):.4f}")

    # salvar checkpoints
    # >>> FIX: usar mkdir (não existe .makedirs em Path)
    ROOT = Path(__file__).resolve().parents[1]
    ckpt_dir = ROOT / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # >>> FIX
    torch.save(G.state_dict(), ckpt_dir / 'G_final.pt')
    torch.save(D.state_dict(), ckpt_dir / 'D_final.pt')
    print("\n✓ GAN treinada e checkpoints salvos em 'checkpoints/'")

    # >>> retorna infos necessárias p/ geração sintética
    return {
        "G_path": str(ckpt_dir / 'G_final.pt'),
        "D_path": str(ckpt_dir / 'D_final.pt'),
        "n_feat": n_feat,
        "n_lbl": n_lbl,
        "latent_dim": cfg['LATENT_DIM'],
        "classes": classes,         # <<< decodificador salvo
        "config": cfg
    }

# -----------------------------------------------------------------------------
# 4) Geração de dataset sintético (sem ler CSV externo)
# -----------------------------------------------------------------------------
def generate_synthetic_dataset(
    ckpt_path: str,
    *,
    n_feat: int,
    n_lbl: int,
    latent_dim: int,
    n_samples_per_label: int = 500,
    classes: np.ndarray | None = None,  # <<< novos
) -> pd.DataFrame:
    G = Generator(latent_dim, n_feat, n_lbl)
    G.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    G.eval()

    if classes is None:
        classes = np.arange(n_lbl)  # identidade

    feats_list, labels_list = [], []
    for lbl in range(n_lbl):
        z = torch.randn(n_samples_per_label, latent_dim)
        y = torch.full((n_samples_per_label,), lbl, dtype=torch.long)
        with torch.no_grad():
            feats = G(z, y).numpy()
        feats_list.append(feats)
        # escreve o rótulo ORIGINAL correspondente a este índice compacto
        labels_list.append(np.full(n_samples_per_label, classes[lbl], dtype=int))

    X_syn = np.vstack(feats_list)
    y_syn = np.concatenate(labels_list)
    cols = [f'feat_{i}' for i in range(X_syn.shape[1])]
    df = pd.DataFrame(X_syn, columns=cols)
    df['labels'] = y_syn
    return df

# -----------------------------------------------------------------------------
# 5) Quick sample (inspeção rápida)
# -----------------------------------------------------------------------------
def quick_sample(num: int = 5):
    ckpt = Path(__file__).resolve().parents[1] / 'checkpoints' / 'G_final.pt'
    # >>> precisamos das dims: para quick, assumimos defaults do último treino
    raise RuntimeError("quick_sample requer n_feat/n_lbl/latent_dim do treino.")

# -----------------------------------------------------------------------------
# 6) Pipeline simples p/ notebook
# -----------------------------------------------------------------------------
def run_experiment(config, X, y, *, save_csv=True):
    train_out = train_gan(config_override=config, X=X, y=y)

    df_syn = generate_synthetic_dataset(
        train_out["G_path"],
        n_feat=train_out["n_feat"],
        n_lbl=train_out["n_lbl"],
        latent_dim=train_out["latent_dim"],
        n_samples_per_label=config.get("SAMPLES_PER_LABEL", 500),
        classes=train_out["classes"],   # <<< garante rótulo original
    )

    if save_csv:
        ROOT = Path(__file__).resolve().parents[1]
        out_dir = ROOT / "synthetic"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "synthetic_cgan.csv"
        df_syn.to_csv(csv_path, index=False)
        print(f"✓ Sintético salvo em: {csv_path}")
    else:
        csv_path = None

    X_syn = df_syn.drop(columns="labels").to_numpy(dtype=np.float32)
    y_syn = df_syn["labels"].to_numpy(dtype=np.int64)

    # Aqui y permanece nos IDs ORIGINAIS; y_syn já veio nos mesmos IDs
    X_aug = np.vstack([X, X_syn])
    y_aug = np.concatenate([y, y_syn])

    return {
        "X_aug": X_aug,
        "y_aug": y_aug,
        "checkpoints": (train_out["G_path"], train_out["D_path"]),
        "synthetic_csv": (str(csv_path) if csv_path else None),
        "config": train_out["config"]
    }

# -----------------------------------------------------------------------------
# 7) CLI legado (mantido)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    train_gan()
