import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from config import (ALERTA_PERCENTIL, CENARIOS, FALHAS, FIGURAS, HORIZONTE_H, PARQUET_LIMPO, RESULTADOS)
from src import anomalia
from src.features import features_horarias
import matplotlib.pyplot as plt


def main():
    ciclos = pd.read_parquet(RESULTADOS / "ciclos.parquet")
    modelo, limiar = anomalia.treinar(ciclos)
    mediana = ciclos[anomalia.ATRIBUTOS].median()
    score = anomalia.escores(modelo, ciclos, mediana)
    alerta = score > limiar

    # rótulo: o ciclo começa nas HORIZONTE_H horas antes do início de uma falha
    y = pd.Series(0, index=ciclos.index)
    avaliado = pd.Series(False, index=ciclos.index)
    dentro_total = pd.Series(False, index=ciclos.index)
    for ini, fim in FALHAS:
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        antes = (ciclos.index >= ini - pd.Timedelta(hours=HORIZONTE_H)) & (ciclos.index < ini)
        dentro = (ciclos.index >= ini) & (ciclos.index <= fim)
        y[antes] = 1
        avaliado |= antes | ((ciclos.index >= ini - pd.Timedelta(days=7)) & (ciclos.index <= fim))
        dentro_total |= dentro
    avaliado &= ~dentro_total   # só depois do loop

    # baseline: fração de tempo sob carga na hora do ciclo (mesma regra das etapas anteriores)
    dados = pd.read_parquet(PARQUET_LIMPO)
    dados = dados[dados["TP3"].notna()].reset_index(drop=True)  # mesmo filtro de episodes.py
    duty = features_horarias(dados)["duty_cycle"]
    base = duty.reindex(ciclos.index.floor("h")).to_numpy()
    base = pd.Series(base, index=ciclos.index).fillna(0)

    ok = avaliado
    print(f"Limiar de alarme (percentil {ALERTA_PERCENTIL} dos ciclos normais): {limiar:.3f}")
    print(f"Ciclos avaliados (perto de falhas): {ok.sum()} (positivos: {int(y[ok].sum())})")
    print(f"  ROC-AUC anomalia : {roc_auc_score(y[ok], score[ok]):.3f}")
    print(f"  PR-AUC anomalia  : {average_precision_score(y[ok], score[ok]):.3f}")
    print(f"  ROC-AUC regra    : {roc_auc_score(y[ok], base[ok]):.3f}")
    print(f"  PR-AUC regra     : {average_precision_score(y[ok], base[ok]):.3f}")
    print(f"  prevalência      : {y[ok].mean():.3f}")

    print("\nAlarmes por falha (alerta nas HORIZONTE_H h antes do início):")
    for ini, fim in FALHAS:
        ini = pd.Timestamp(ini)
        antes = alerta[(alerta.index >= ini - pd.Timedelta(hours=HORIZONTE_H)) & (alerta.index < ini)]
        cic_ini = ciclos.index[(ciclos.index >= ini - pd.Timedelta(days=7)) & (ciclos.index < ini)]
        n_alarm = int(alerta[cic_ini].sum())
        print(f"  falha {ini}: alarmes na janela de 7 dias = {n_alarm} de {len(cic_ini)} ciclos;"
              f" alarme nas {HORIZONTE_H} h antes = {int(antes.sum())}")

    a, b = (pd.Timestamp(x) for x in CENARIOS["normal_controle"])
    ctrl = alerta[(alerta.index >= a) & (alerta.index < b)]
    dias = (b - a).total_seconds() / 86400
    print(f"\nControle normal (sem falha): {int(ctrl.sum())} alarmes em {dias:.0f} dias"
          f" = {ctrl.sum() / dias:.2f} por dia")
    dias_treino = (ciclos.index.max() - ciclos.index.min()).total_seconds() / 86400
    print(f"Todo o período: {int(alerta.sum())} alarmes em {dias_treino:.0f} dias"
          f" = {alerta.sum() / dias_treino:.2f} por dia")

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(12, 3 * len(FALHAS)), sharey=True)
    for ax, (ini, fim) in zip(axes, FALHAS):
        a_ = pd.Timestamp(ini) - pd.Timedelta(days=7)
        b_ = pd.Timestamp(fim) + pd.Timedelta(days=2)
        s = score[(score.index >= a_) & (score.index <= b_)]
        ax.plot(s.index, s.values, marker=".", lw=0.6, label="escore de anomalia")
        ax.axhline(limiar, color="red", ls=":", label="limiar de alarme")
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim), color="red", alpha=0.25)
        ax.set_title(f"Falha em {ini}", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Escore de anomalia por ciclo de carga (vermelho = falha)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "08_escore_anomalia_por_falha.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura salva em: {FIGURAS / '08_escore_anomalia_por_falha.png'}")


if __name__ == "__main__":
    main()
