import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import FALHAS, FIGURAS, MANUTENCOES, RISCO_ALTO, RISCO_MEDIO


def _salvar(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nome, dpi=120, bbox_inches="tight")
    plt.close(fig)


def curva_precisao_recall(y, oof, nome="04_curva_precisao_recall.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in oof.columns:
        ok = oof[col].notna()
        p, r, _ = precision_recall_curve(y[ok], oof[col][ok])
        ax.plot(r, p, label=col)
    ax.axhline(y.mean(), color="gray", ls="--", label=f"aleatório ({y.mean():.3f})")
    ax.set_xlabel("Recall (falhas detectadas)")
    ax.set_ylabel("Precisão (alertas corretos)")
    ax.set_title("Curva precisão x recall (validação por falha)")
    ax.legend()
    _salvar(fig, nome)


def probabilidade_por_falha(oof, modelo, nome="05_probabilidade_por_falha.png"):
    """Probabilidade prevista ao redor de cada falha (faixa vermelha = falha)."""
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(12, 3 * len(FALHAS)), sharey=True)
    for ax, (ini, fim) in zip(axes, FALHAS):
        a = pd.Timestamp(ini) - pd.Timedelta(days=7)
        b = pd.Timestamp(fim) + pd.Timedelta(days=2)
        s = oof[modelo].loc[a:b]
        ax.plot(s.index, s.values, marker=".", lw=0.8, label=modelo)
        ax.plot(oof["regra_base_duty_cycle"].loc[a:b].index,
                oof["regra_base_duty_cycle"].loc[a:b].values, lw=0.6, alpha=0.6,
                label="regra base (duty cycle)")
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim), color="red", alpha=0.25)
        ax.axhline(RISCO_ALTO, color="red", ls=":", lw=1)
        ax.axhline(RISCO_MEDIO, color="orange", ls=":", lw=1)
        ax.set_title(f"Falha em {ini}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Probabilidade de falha nas próximas 24 h (linhas: alto=vermelho, médio=laranja)")
    fig.tight_layout()
    _salvar(fig, nome)


def matriz_confusao(y, s, nome="06_matriz_confusao.png"):
    from sklearn.metrics import confusion_matrix
    ok = s.notna()
    pred = (s[ok] >= RISCO_ALTO).astype(int)
    cm = confusion_matrix(y[ok], pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["sem falha", "alerta"])
    ax.set_yticks([0, 1], ["sem falha", "falha"])
    ax.set_title(f"Matriz de confusão (limiar alto = {RISCO_ALTO})")
    _salvar(fig, nome)
    return cm


def simulacao(linha: pd.DataFrame, nome_cenario: str):
    nome = f"07_simulacao_{nome_cenario}.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(linha.index, linha["probabilidade"], marker=".", lw=0.8)
    ax.axhline(RISCO_ALTO, color="red", ls=":", label=f"alto ({RISCO_ALTO})")
    ax.axhline(RISCO_MEDIO, color="orange", ls=":", label=f"médio ({RISCO_MEDIO})")
    for ini, fim in FALHAS:
        if pd.Timestamp(fim) >= linha.index.min() and pd.Timestamp(ini) <= linha.index.max():
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim), color="red", alpha=0.2)
    for m in MANUTENCOES:
        if linha.index.min() <= pd.Timestamp(m) <= linha.index.max():
            ax.axvline(pd.Timestamp(m), color="green", ls="--", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probabilidade de falha (24 h)")
    ax.set_title(f"Simulação: {nome_cenario}")
    ax.legend(loc="upper left", fontsize=8)
    _salvar(fig, nome)
