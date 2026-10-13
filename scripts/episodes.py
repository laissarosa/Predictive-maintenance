import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from config import FALHAS, FIGURAS, PARQUET_LIMPO, RESULTADOS
from src.episodios import detectar_episodios


def main():
    # dataset limpo: sem trechos congelados; instantes sem dado (NaN) saem da série
    df = pd.read_parquet(PARQUET_LIMPO)
    df = df[df["TP3"].notna()].reset_index(drop=True)
    ep = detectar_episodios(df)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    ep.to_parquet(RESULTADOS / "episodios.parquet")

    pd.set_option("display.width", 200)
    print(ep.round(2).to_string())
    ok = ep[~ep["lacunas"]]
    print(f"\nEpisódios: {len(ep)} ({len(ok)} com cobertura >= 80%)")
    rot = ep[ep["falha_rotulada"].notna()]
    print(f"Cobrem falha rotulada: {len(rot)}; falhas rotuladas cobertas: "
          f"{rot['falha_rotulada'].nunique()} de {len(FALHAS)}")
    sem = [a for a, _ in FALHAS if pd.Timestamp(a) not in set(rot["falha_rotulada"])]
    print(f"Falhas sem episódio: {sem}")
    print(f"Episódios sem rótulo (candidatos): {int(ep['falha_rotulada'].isna().sum())}")

    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 3))
    for r in ep.itertuples():
        cor = "red" if pd.notna(r.falha_rotulada) else ("gray" if r.lacunas else "orange")
        ax.axvspan(r.ini, r.fim, color=cor, alpha=0.6)
    for a, b in FALHAS:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), ymin=0, ymax=0.15, color="black")
    ax.set_yticks([])
    ax.set_title("Episódios de carga contínua (vermelho = rotulado, laranja = sem rótulo, "
                 "cinza = muita lacuna; barra preta = janela de falha)")
    fig.savefig(FIGURAS / "09_episodios.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
