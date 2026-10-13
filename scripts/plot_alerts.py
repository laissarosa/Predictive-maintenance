import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import COMPONENTES, FALHAS, FIGURAS, RESULTADOS

COR = ["#e6e6e6", "#4c9be8", "#f0a030", "#d62728"]
NOME = ["normal", "informação", "aviso", "crítico"]


def episodios_alerta(niv: pd.Series, minimo: int = 2) -> pd.DataFrame:
    """Subidas do nível do sistema: início, pico, fim e duração de cada trecho com nível >= minimo."""
    ativo = (niv >= minimo).to_numpy()
    linhas, i = [], 0
    while i < len(ativo):
        if ativo[i]:
            j = i
            while j + 1 < len(ativo) and ativo[j + 1]:
                j += 1
            seg = niv.iloc[i:j + 1]
            linhas.append({"inicio": seg.index[0], "fim": seg.index[-1], "duracao_h": j - i + 1,
                           "pico": NOME[int(seg.max())]})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(linhas)


def main():
    niv = pd.read_parquet(RESULTADOS / "alertas.parquet")
    comps = list(COMPONENTES)
    com_dado = niv["com_dado"]
    n = len(FALHAS)
    fig, axes = plt.subplots(2 * n, 1, figsize=(14, 6.2 * n),
                             gridspec_kw={"height_ratios": [3, 2] * n})
    for k, (ini, fim) in enumerate(FALHAS):
        a, b = pd.Timestamp(ini), pd.Timestamp(fim)
        x0, x1 = a - pd.Timedelta(days=3), max(b, a) + pd.Timedelta(days=1.5)
        sub = niv[(niv.index >= x0) & (niv.index <= x1)]
        t = mdates.date2num(sub.index.append(sub.index[-1:] + pd.Timedelta(hours=1)))
        ax, ax2 = axes[2 * k], axes[2 * k + 1]

        m = sub[comps + ["sistema"]].to_numpy().T.astype(float)
        m[:, ~sub["com_dado"].to_numpy()] = np.nan
        ax.pcolormesh(t, np.arange(len(comps) + 3), np.ma.masked_invalid(
            np.vstack([m[:-1], np.full(m.shape[1], np.nan), m[-1:]])),
            cmap=ListedColormap(COR), vmin=0, vmax=3)
        ax.set_yticks(np.arange(len(comps) + 2) + 0.5,
                      [f"{c} (sev {COMPONENTES[c]['severidade']})" for c in comps] + ["", "SISTEMA"],
                      fontsize=10)
        ax.invert_yaxis()
        for axx in (ax, ax2):
            axx.axvspan(a, b, facecolor="none", edgecolor="black", lw=2, hatch="//", alpha=0.5)
            axx.set_xlim(x0, x1)
            axx.xaxis.set_major_locator(mdates.DayLocator())
            axx.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            axx.grid(axis="x", color="white", lw=0.8, alpha=0.8)
        ax.set_title(f"Falha {k + 1}: {ini} → {fim}   (hachura = janela de falha; "
                     f"espaço em branco = sem dado válido)", fontsize=11, loc="left")

        # quantos componentes estão em cada nível, hora a hora (subida e queda de cada tipo)
        cont = {l: (sub[comps] >= l).sum(axis=1).where(sub["com_dado"]) for l in (1, 2, 3)}
        for l, cor in ((1, COR[1]), (2, COR[2]), (3, COR[3])):
            ax2.fill_between(sub.index, cont[l].fillna(0), step="post", color=cor, alpha=0.35)
            ax2.step(sub.index, cont[l].fillna(0), where="post", color=cor, lw=1.8,
                     label=f"componentes em {NOME[l]} ou mais")
            if cont[l].max() > 0:
                pk = cont[l].idxmax()
                ax2.annotate(f"pico {int(cont[l].max())}", (pk, cont[l].max()), fontsize=8,
                             xytext=(0, 6), textcoords="offset points", ha="center", color=cor)
        ax2.set_ylim(0, len(comps) + 0.7)
        ax2.set_ylabel("nº de componentes")
        if k == 0:
            ax2.legend(loc="upper left", fontsize=8, ncol=3)
    handles = [Patch(color=c, label=n_) for c, n_ in zip(COR, NOME)]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "11_alertas_subida_e_queda.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    ep = episodios_alerta(niv["sistema"][com_dado])
    ep.to_csv(RESULTADOS / "episodios_alerta.csv", index=False)
    print(ep.to_string(index=False))


if __name__ == "__main__":
    main()
