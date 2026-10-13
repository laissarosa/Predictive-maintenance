import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ANTECEDENCIA_MIN_H, COMPONENTES, FALHAS, FIGURAS, INTERVALO_FUNDIR_H,
                    MODELOS, PARQUET_LIMPO, RESULTADOS)
from src import autoencoder as ae

QUANTIL = 0.99      # limiar = quantil do escore horário em dias normais de validação
PERSIST_H = 2       # horas seguidas acima do limiar para disparar
MIN_JANELAS_HORA = 3


def sobrepoe(a0, a1, b0, b1) -> float:
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds() / 3600)


def intervalos(alerta: pd.Series) -> list:
    horas = alerta.index[alerta]
    out = []
    for h in horas:
        if out and h - out[-1][1] <= pd.Timedelta(hours=INTERVALO_FUNDIR_H + 1):
            out[-1][1] = h
        else:
            out.append([h, h])
    return [(a, b + pd.Timedelta(hours=1)) for a, b in out]


def main():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)

    limpo = pd.read_parquet(PARQUET_LIMPO)
    ep_path = RESULTADOS / "episodios.parquet"
    ep = pd.read_parquet(ep_path) if ep_path.exists() else None
    if ep is not None:
        ep = ep[~ep["lacunas"]]

    # ---- janelas e separação treino / validação (por dia inteiro) ----
    m = ae.serie_minuto(limpo)
    w, fim = ae.janelas(m)
    excl = ae.zona_excluida(fim, ep)
    dia = (fim.normalize() - fim.min().normalize()).days.to_numpy()
    val = (dia % 5 == 0) & ~excl
    tr = ~excl & ~val
    print(f"Janelas de {ae.JANELA_MIN} min: {len(w):,} | treino {tr.sum():,} | "
          f"validação normal {val.sum():,} | excluídas (falha/episódio) {excl.sum():,}")

    # ---- treino ----
    print("Treinando autoencoder (só dados normais)...")
    modelo, mu, sd, _ = ae.treinar(w[tr], w[val])
    import torch
    torch.save({"state": modelo.state_dict(), "mu": mu, "sd": sd}, MODELOS / "autoencoder.pt")

    # ---- escores ----
    err = ae.erro_por_sensor(modelo, w, mu, sd)
    escala = np.quantile(err[val], 0.95, axis=0) + 1e-6
    esc = ae.escores_componentes(err, escala, fim)
    esc["validacao"] = val
    esc["excluida"] = excl
    esc.to_parquet(RESULTADOS / "ae_escores.parquet")

    # ---- alerta horário: mediana do escore na hora + persistência ----
    sis = esc["sistema"]
    n_h = sis.resample("1h").count()
    horario = sis.resample("1h").median().where(n_h >= MIN_JANELAS_HORA)
    hv = esc.loc[esc["validacao"], "sistema"].resample("1h").median()
    hv = hv.where(sis[esc["validacao"]].resample("1h").count() >= MIN_JANELAS_HORA).dropna()
    limiar = float(hv.quantile(QUANTIL))
    acima = (horario > limiar).astype(float).where(horario.notna())
    alerta = acima.rolling(PERSIST_H, min_periods=PERSIST_H).min().fillna(0).astype(bool)
    print(f"\nLimiar (quantil {QUANTIL} de {len(hv)} horas normais de validação): {limiar:.2f}"
          f" | persistência: {PERSIST_H} h")

    # ---- avaliação por intervalos ----
    prev = intervalos(alerta)
    linhas = []
    for ini, fi in FALHAS:
        a, b = pd.Timestamp(ini), pd.Timestamp(fi)
        tocam = [(x, y) for x, y in prev if sobrepoe(x, y, a - pd.Timedelta(hours=24), b) > 0]
        dado24 = horario[(horario.index >= a - pd.Timedelta(hours=24)) & (horario.index < a)].notna().mean()
        lin = {"falha": ini, "dado_24h_antes": round(float(dado24), 2), "detectada": bool(tocam),
               "inicio_alerta": tocam[0][0] if tocam else pd.NaT}
        if tocam:
            lead = (a - tocam[0][0]).total_seconds() / 3600
            lin["antecedencia_h"] = round(lead, 1)          # negativo = depois do início da falha
            lin["cumpre_2h"] = lead >= ANTECEDENCIA_MIN_H
            sub = esc[(esc.index >= tocam[0][0]) & (esc.index <= b)][list(COMPONENTES)]
            lin["componentes"] = ", ".join(f"{c}:{v:.1f}" for c, v in
                                           sub.median().sort_values(ascending=False).head(3).items())
        linhas.append(lin)
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "ae_avaliacao.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nPor falha (antecedência > 0 = antes do início; < 0 = depois):")
    print(tab.to_string(index=False))

    # falsos alarmes: intervalos longe de falhas e episódios (±24 h)
    conhecidos = [(pd.Timestamp(a) - pd.Timedelta(hours=24), pd.Timestamp(b) + pd.Timedelta(hours=24))
                  for a, b in FALHAS]
    if ep is not None:
        conhecidos += [(r.ini - pd.Timedelta(hours=24), r.fim + pd.Timedelta(hours=24))
                       for r in ep.itertuples()]
    fp = [(x, y) for x, y in prev if not any(sobrepoe(x, y, c0, c1) > 0 for c0, c1 in conhecidos)]
    em_ep = [(x, y) for x, y in prev if (x, y) not in fp
             and not any(sobrepoe(x, y, pd.Timestamp(a) - pd.Timedelta(hours=24),
                                  pd.Timestamp(b)) > 0 for a, b in FALHAS)]
    longe = pd.Series(~ae.zona_excluida(horario.index, ep), index=horario.index) & horario.notna()
    sem_conhecidos = longe & ~pd.Series(
        [any(c0 <= h < c1 for c0, c1 in conhecidos) for h in horario.index], index=horario.index)
    semanas = sem_conhecidos.sum() / 24 / 7
    fp_val = [(x, y) for x, y in fp if bool(esc["validacao"].reindex(
        pd.date_range(x, y, freq="10min")).fillna(False).any())]
    print(f"\nIntervalos previstos: {len(prev)} | falsos positivos: {len(fp)} "
          f"(~{len(fp) / max(semanas, 1e-9):.2f}/semana em {semanas:.1f} semanas normais)"
          f" | em episódios sem rótulo: {len(em_ep)}")
    print(f"Falsos positivos que caem em dias de validação (não vistos no treino): {len(fp_val)}")
    print("Obs.: os limiares usam as horas de validação, e parte das semanas normais foi vista no "
          "treino; trate a taxa como otimista.")

    # ---- figura ----
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(12, 3.2 * len(FALHAS)))
    for ax, (ini, fi) in zip(axes, FALHAS):
        a, b = pd.Timestamp(ini), pd.Timestamp(fi)
        x0, x1 = a - pd.Timedelta(days=4), b + pd.Timedelta(days=1.5)
        h = horario[(horario.index >= x0) & (horario.index <= x1)]
        ax.plot(h.index, h.values, ls="none", marker=".", ms=4, label="escore do sistema (mediana/h)")
        al = h[alerta.reindex(h.index).fillna(False)]
        ax.plot(al.index, al.values, ls="none", marker="o", ms=5, color="darkorange", label="alerta")
        ax.axhline(limiar, color="red", ls=":", label="limiar")
        ax.axvspan(a, b, color="red", alpha=0.25)
        ax.set_yscale("log")
        ax.set_title(f"Falha em {ini}", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Autoencoder 1D: erro de reconstrução relativo (vermelho = falha)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "12_autoencoder.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura: {FIGURAS / '12_autoencoder.png'}")
    print(f"Escores por componente (para o demo): {RESULTADOS / 'ae_escores.parquet'}")


if __name__ == "__main__":
    main()
