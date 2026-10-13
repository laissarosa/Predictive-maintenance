import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import FALHAS, RESULTADOS
from scripts.leak_proxy import PROXIES

# Controle de demanda: compara as 24 h antes do episódio com as mesmas 24 h de relógio em
# outros dias (só dias anteriores, sem episódios nem falhas), em vez de "a semana anterior".
# Mesmo dia da semana = mesmo padrão de operação do trem.
DIAS_REF = 28
MIN_CICLOS = 15


def _proibidos(ep: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    h = pd.Timedelta(hours=2)
    p = [(r.ini - h, r.fim + h) for r in ep.itertuples()]
    p += [(pd.Timestamp(a) - h, pd.Timestamp(b) + h) for a, b in FALHAS]
    return p


def referencias(ciclos, ini, proibidos, mesmo_dia_semana: bool):
    passo = 7 if mesmo_dia_semana else 1
    vals = []
    for k in range(passo, DIAS_REF + 1, passo):
        a, b = ini - pd.Timedelta(hours=24) - pd.Timedelta(days=k), ini - pd.Timedelta(days=k)
        if any(a < pb and b > pa for pa, pb in proibidos):
            continue
        s = ciclos[(ciclos.index >= a) & (ciclos.index < b)]
        if len(s) >= MIN_CICLOS:
            vals.append(s[PROXIES].median())
    return pd.DataFrame(vals)


def main():
    ciclos = pd.read_parquet(RESULTADOS / "ciclos.parquet")
    ep_all = pd.read_parquet(RESULTADOS / "episodios.parquet")
    proib = _proibidos(ep_all)
    ep = ep_all[~ep_all["lacunas"]]

    linhas = []
    for r in ep.itertuples():
        ult = ciclos[(ciclos.index >= r.ini - pd.Timedelta(hours=24)) & (ciclos.index < r.ini)]
        for nome, sem in (("todos_dias", False), ("mesmo_dia_semana", True)):
            ref = referencias(ciclos, r.ini, proib, sem)
            lin = {"ini": r.ini, "rotulado": pd.notna(r.falha_rotulada), "ref": nome,
                   "n_ref": len(ref), "ciclos_24h": len(ult)}
            for p in PROXIES:
                if len(ult) < MIN_CICLOS or len(ref) < 3:
                    lin[p] = np.nan
                else:
                    # percentil do valor observado entre os dias de referência (0-100)
                    lin[p] = 100 * (ref[p] < ult[p].median()).mean()
            linhas.append(lin)
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "episodios_proxies_demanda.csv", index=False)
    pd.set_option("display.width", 250)
    print("Percentil das 24 h antes do episódio entre dias de referência "
          "(50 = típico; descanso alto / queda baixa = menos demanda)\n")
    print(tab.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
