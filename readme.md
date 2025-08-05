# 📚 Project: "GAN‑Driven Phase Classification in Quantum Spin Chains"

This repository contains the **data‑engineering layer** for three spin‑1 Hamiltonians (H1, H2, H3) taken from Mahlow (2021).  Current scripts convert raw DMRG output into clean tabular files and generate three scaled variants ready for downstream models.

> **Aim**  Prepare reliable datasets so that a Generative Adversarial Network (GAN) can later perform **data augmentation**, enabling better **phase classification** and construction of the **phase diagram**.

---

## 🗂️ Directory layout

```text
.
├── data/
│   ├── raw/               # original DMRG samples
│   ├── labeled/           # CSV with labels (make_dataset.py)
│   └── processed/         # baseline / clean / balanced (make_processed.py)
│       └── H1/, H2/, H3/
├── notebooks/             # EDA & sanity checks
├── src/
│   ├── make_dataset.py    # raw → labeled CSV
│   ├── make_processed.py  # preprocessing pipeline
│   └── ...
├── environment.yml
└── README.md
```
---


---

## 📑 Dataset summary

| Tag    | Spin‑1 Hamiltonian              | Parameter sweep                                  |
| ------ | ------------------------------- | ------------------------------------------------ |
| **H1** | XXZ chain + uniaxial anisotropy | \(J_z,\,D \in [-4,4]\) step 0.1                  |
| **H2** | Bond‑alternating XXZ            | \(\Delta \in [0,-1],\; \delta \in [-1.5,\,2.5]\) |
| **H3** | Bilinear–biquadratic            | \(\theta \in [0,2\pi]\) step \(\pi\,10^{-3}\)    |

### 📝 Hamiltonians

```math
\mathbf{H_1} = \sum_{l} \Big[ J\,(S^x_l S^x_{l+1} + S^y_l S^y_{l+1}) + J_z S^z_l S^z_{l+1} \Big] + D \sum_l (S^z_l)^2
```

```math
\mathbf{H_2} = \sum_{l} \big[ 1 - \delta (-1)^l \big]\big[ S^x_l S^x_{l+1} + S^y_l S^y_{l+1} + \Delta S^z_l S^z_{l+1} \big]
```

```math
\mathbf{H_3} = \sum_{l} \Big[ \cos\theta\,(\mathbf{S}_l \!\cdot\! \mathbf{S}_{l+1}) + \sin\theta\,(\mathbf{S}_l \!\cdot\! \mathbf{S}_{l+1})^2 \Big]
```

Each row in the CSV stores pre‑selected correlation functions plus a **phase label**.

---

## 🔧 Key dependencies

- `python >= 3.11`
- `numpy`, `pandas`
- `scikit‑learn >= 1.5`
- `imbalanced‑learn >= 0.13`
- `joblib`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

