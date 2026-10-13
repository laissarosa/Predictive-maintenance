import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import LIMIAR_CARGA, RESULTADOS
from src.data import carregar_dados

# proxies de vazamento: durante o descanso (sem carga) a pressão dos reservatórios cai mais
# rápido e o compressor religa mais cedo; durante o episódio o DV_pressure fica elevado
PROXIES = ["queda_reserv_descanso", "descanso_min", "dur_carga_min"]


def pre_episodio(ciclos: pd.DataFrame, ini: pd.Timestamp) -> dict:
    """Mediana das últimas 24 h antes do episódio contra as 7 dias anteriores a isso
    (só passado). Razão > 1 em `queda` (mais negativa) ou < 1 em `descanso` = piora."""
    ult = ciclos[(ciclos.index >= ini - pd.Timedelta(hours=24)) & (ciclos.index < ini)]
    ref = ciclos[(ciclos.index >= ini - pd.Timedelta(days=8)) & (ciclos.index < ini - pd.Timedelta(hours=24))]
    out = {"ciclos_24h": len(ult), "ciclos_ref": len(ref)}
    for p in PROXIES:
        a, b = ult[p].median(), ref[p].median()
        out[f"{p}_razao"] = a / b if len(ult) >= 5 and len(ref) >= 20 and b != 0 else np.nan
    return out


def dentro_episodio(d: pd.DataFrame, ini, fim) -> dict:
    s = d[(d["timestamp"] >= ini) & (d["timestamp"] <= fim)]
    t = (s["timestamp"] - s["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600
    return {
        "dv_pressure": s["DV_pressure"].mean(),
        "tp3_media": s["TP3"].mean(),
        "tp3_inclinacao_bar_h": np.polyfit(t, s["TP3"].to_numpy(), 1)[0],
        "caudal_media": s["Caudal_impulses"].mean(),
        "oleo_media": s["Oil_temperature"].mean(),
    }


def main():
    d = carregar_dados()
    ciclos = pd.read_parquet(RESULTADOS / "ciclos.parquet")
    ep = pd.read_parquet(RESULTADOS / "episodios.parquet")
    ep = ep[~ep["lacunas"]].reset_index(drop=True)

    linhas = []
    for r in ep.itertuples():
        linhas.append({"ini": r.ini, "dur_h": round(r.duracao_h, 1), "tipo": r.tipo,
                       "rotulado": pd.notna(r.falha_rotulada),
                       **dentro_episodio(d, r.ini, r.fim), **pre_episodio(ciclos, r.ini)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "episodios_proxies.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string())
    ref = tab[~tab["rotulado"]]
    print("\nRazões pré-episódio (últimas 24 h / semana anterior):")
    for p in PROXIES:
        c = f"{p}_razao"
        print(f"  {p}: rotulados {tab.loc[tab.rotulado, c].round(2).tolist()}"
              f" | sem rótulo {ref[c].round(2).tolist()}")


if __name__ == "__main__":
    main()
