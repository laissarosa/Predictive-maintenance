import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (CALIBRACAO_FIM, COMPONENTES, FALHAS, JANELA_ANTES_DIAS, JANELA_DEPOIS_DIAS,
                    PARQUET_LIMPO, RESULTADOS)
from src.alertas import NIVEIS, alertas, calibrar_limiares
from src.escores import escores_componentes

def horas_normais_ep(index, ep):
    m = np.ones(len(index), bool)
    for r in ep.itertuples():
        m &= ~((index >= r.ini - pd.Timedelta(days=1)) & (index <= r.fim + pd.Timedelta(days=1)))
    return m


def horas_normais(index, ep):
    """Horas longe de falhas rotuladas e de episódios de carga contínua."""
    m = np.ones(len(index), bool)
    for a, b in FALHAS:
        m &= ~((index >= pd.Timestamp(a) - pd.Timedelta(days=JANELA_ANTES_DIAS))
               & (index <= pd.Timestamp(b) + pd.Timedelta(days=JANELA_DEPOIS_DIAS)))
    for r in ep.itertuples():
        m &= ~((index >= r.ini - pd.Timedelta(days=1)) & (index <= r.fim + pd.Timedelta(days=1)))
    return m


def main():
    limpo = pd.read_parquet(PARQUET_LIMPO)
    esc, z = escores_componentes(limpo)
    esc.to_parquet(RESULTADOS / "escores.parquet")
    z.to_parquet(RESULTADOS / "z_atributos.parquet")
    ep = pd.read_parquet(RESULTADOS / "episodios.parquet")
    ep = ep[~ep["lacunas"]]
    com_dado = esc.notna().any(axis=1)
    em_episodio = ~horas_normais_ep(esc.index, ep)
    lim = calibrar_limiares(esc, em_episodio)
    print('Limiares (info, aviso, crítico) por componente:')
    for c, v in lim.items():
        print(f'  {c:9s}', tuple(round(x, 1) for x in v))
    niv = alertas(esc.fillna(0), lim)
    niv["com_dado"] = com_dado
    niv.to_parquet(RESULTADOS / "alertas.parquet")

    normal = (horas_normais(esc.index, ep) & com_dado.to_numpy()
              & (esc.index >= pd.Timestamp(CALIBRACAO_FIM)))
    dias = normal.sum() / 24
    print(f"Horas normais avaliadas: {normal.sum()} ({dias:.0f} dias)")
    print("\nFalsos alarmes em período normal (horas em cada nível e eventos por dia):")
    for c in list(COMPONENTES) + ["sistema"]:
        s = niv[c][normal]
        ev = ((niv[c] >= 2) & (niv[c].shift().fillna(0) < 2))[normal].sum()
        print(f"  {c:9s} info+ {np.mean(s>=1):5.1%}  aviso+ {np.mean(s>=2):5.1%}  "
              f"crítico {np.mean(s>=3):5.1%}  | eventos aviso+/dia: {ev/dias:.2f}")

    print("\nPor falha: primeiro alerta antes do início e estado durante a falha")
    for ini, fim in FALHAS:
        a, b = pd.Timestamp(ini), pd.Timestamp(fim)
        pre = niv[(niv.index >= a - pd.Timedelta(hours=24)) & (niv.index < a)]
        dur = niv[(niv.index >= a.floor("h")) & (niv.index <= b)]
        linha = f"  {ini}: dados nas 24 h antes {pre['com_dado'].mean():.0%}"
        for nome, lim in (("info", 1), ("aviso", 2), ("crítico", 3)):
            q = niv.loc[(niv.index >= a - pd.Timedelta(hours=24)) & (niv.index < a), "sistema"]
            q = q[q >= lim]
            linha += f" | {nome}: " + (f"{(a - q.index[0]).total_seconds()/3600:.0f} h antes" if len(q) else "-")
        print(linha)
        if dur["com_dado"].any():
            frac = {NIVEIS[k]: f"{(dur.loc[dur.com_dado,'sistema']==k).mean():.0%}" for k in range(4)}
            top = [c for c in COMPONENTES if (dur[c] >= 2).mean() > 0.3]
            print(f"      durante a falha: {frac}; componentes em aviso+ >30% do tempo: {top}")


if __name__ == "__main__":
    main()
