"""Autoencoder convolucional 1D multissensor para detecção de desvio do comportamento normal.

Entrada : janelas de JANELA_MIN minutos dos 15 sensores (dataset limpo, reamostrado a 1 min).
Treino  : só janelas normais (fora das janelas de falha e dos episódios de carga contínua).
Saída   : erro de reconstrução por sensor -> por componente (config.COMPONENTES) -> sistema.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import COMPONENTES, FALHAS, JANELA_ANTES_DIAS, JANELA_DEPOIS_DIAS

SENSORES = [s for c in COMPONENTES.values() for s in c["sensores"]]
JANELA_MIN = 60      # tamanho da janela (minutos); precisa ser múltiplo de 4
PASSO_MIN = 10     # passo entre janelas
EPOCAS = 120
LOTE = 256
LATENTE = 8          # canais do gargalo (8 x 15 = 120 números para 60 x 15 = 900 de entrada)
CLIP = 10.0


# ---------------------------------------------------------------- dados
def serie_minuto(limpo: pd.DataFrame) -> pd.DataFrame:
    """Dataset limpo (grade de 10 s, NaN onde não há dado válido) -> média por minuto."""
    d = limpo.set_index("timestamp")[SENSORES].astype(float)
    return d.resample("1min").mean().ffill(limit=2)


def janelas(m: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Janelas sem nenhum NaN. Devolve (N, JANELA_MIN, C) e o instante do FIM de cada janela
    (o alerta só usa o passado: o escore é carimbado no fim da janela)."""
    x = m.to_numpy(dtype=np.float32)
    ruim = np.concatenate([[0], np.cumsum(np.isnan(x).any(axis=1))])
    ini = np.arange(0, len(x) - JANELA_MIN + 1, PASSO_MIN)
    ini = ini[(ruim[ini + JANELA_MIN] - ruim[ini]) == 0]
    w = np.stack([x[i:i + JANELA_MIN] for i in ini])
    return w, pd.DatetimeIndex(m.index[ini + JANELA_MIN - 1])


def zona_excluida(index: pd.DatetimeIndex, episodios: pd.DataFrame | None) -> np.ndarray:
    """Instantes que NÃO podem entrar no 'normal': em torno das falhas rotuladas
    (JANELA_ANTES/DEPOIS dias) e dos episódios de carga contínua (±1 dia)."""
    m = np.zeros(len(index), bool)
    for a, b in FALHAS:
        m |= (index >= pd.Timestamp(a) - pd.Timedelta(days=JANELA_ANTES_DIAS)) & \
             (index <= pd.Timestamp(b) + pd.Timedelta(days=JANELA_DEPOIS_DIAS))
    if episodios is not None:
        for r in episodios.itertuples():
            m |= (index >= r.ini - pd.Timedelta(days=1)) & (index <= r.fim + pd.Timedelta(days=1))
    return m


# ---------------------------------------------------------------- modelo
class AE(nn.Module):
    def __init__(self, canais: int = len(SENSORES), z: int = LATENTE):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(canais, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv1d(64, z, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(z, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(32, canais, 4, 2, 1),
        )

    def forward(self, x):  # x: (N, C, T)
        return self.dec(self.enc(x))


def _dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(w, mu, sd):
    z = np.clip((w - mu) / sd, -CLIP, CLIP).astype(np.float32)
    return torch.from_numpy(z).permute(0, 2, 1).contiguous()  # (N, C, T)


def treinar(w_treino: np.ndarray, w_val: np.ndarray, verbose: bool = True):
    """Treina só com janelas normais. Devolve (modelo, mu, sd, historico)."""
    torch.manual_seed(0)
    np.random.seed(0)
    mu = w_treino.reshape(-1, w_treino.shape[2]).mean(0)
    sd = np.maximum(w_treino.reshape(-1, w_treino.shape[2]).std(0), 1e-2)
    dev = _dispositivo()
    xt, xv = _tensor(w_treino, mu, sd).to(dev), _tensor(w_val, mu, sd).to(dev)
    modelo = AE().to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-5)
    perda = nn.MSELoss()
    melhor, estado, hist = np.inf, None, []
    for ep in range(EPOCAS):
        modelo.train()
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for i in range(0, len(xt), LOTE):
            b = xt[perm[i:i + LOTE]]
            opt.zero_grad()
            l = perda(modelo(b), b)
            l.backward()
            opt.step()
            tot += l.item() * len(b)
        modelo.eval()
        with torch.no_grad():
            lv = perda(modelo(xv), xv).item()
        hist.append((tot / len(xt), lv))
        if lv < melhor:
            melhor, estado = lv, {k: v.clone() for k, v in modelo.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == EPOCAS - 1):
            print(f"  época {ep + 1:2d}: treino {hist[-1][0]:.4f} | validação {lv:.4f}")
    modelo.load_state_dict(estado)
    return modelo.cpu(), mu, sd, hist


def erro_por_sensor(modelo, w: np.ndarray, mu, sd) -> np.ndarray:
    """Erro quadrático médio no tempo, por janela e por sensor: (N, C)."""
    modelo.eval()
    x = _tensor(w, mu, sd)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            b = x[i:i + 1024]
            out.append(((modelo(b) - b) ** 2).mean(dim=2).numpy())
    return np.concatenate(out)


def escores_componentes(err: np.ndarray, escala: np.ndarray, index) -> pd.DataFrame:
    """Erro de cada sensor dividido pela sua mediana em dados normais de validação
    (1 = típico). Componente = média dos seus sensores; sistema = pior componente."""
    rel = err / escala
    cols = {}
    for c, info in COMPONENTES.items():
        idx = [SENSORES.index(s) for s in info["sensores"]]
        cols[c] = rel[:, idx].mean(axis=1)
    df = pd.DataFrame(cols, index=index)
    df["sistema"] = df.max(axis=1)
    return df
