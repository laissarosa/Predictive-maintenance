import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (ANTECEDENCIA_MIN_H, CALIBRACAO_FIM, FALHAS, INTERVALO_FUNDIR_H, RESULTADOS)

NIVEL = {"informacao": 1, "aviso": 2, "critico": 3}
# episódios prováveis (sem rótulo, mesma assinatura das falhas): ver episodios_nao_rotulados.csv
PROVAVEIS = ["2020-04-12", "2020-05-13", "2020-05-19"]


def intervalos_previstos(sistema: pd.Series, com_dado: pd.Series, nivel: int) -> list:
    """Trechos com nível >= `nivel`; alertas a <= INTERVALO_FUNDIR_H horas viram um intervalo."""
    horas = sistema.index[(sistema >= nivel) & com_dado]
    out = []
    for h in horas:
        if out and h - out[-1][1] <= pd.Timedelta(hours=INTERVALO_FUNDIR_H + 1):
            out[-1][1] = h
        else:
            out.append([h, h])
    return [(a, b + pd.Timedelta(hours=1)) for a, b in out]


def sobrepoe(a0, a1, b0, b1) -> float:
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds() / 3600)


def main():
    niv = pd.read_parquet(RESULTADOS / "alertas.parquet")
    ep = pd.read_parquet(RESULTADOS / "episodios.parquet")
    ep = ep[~ep["lacunas"]]
    ini_teste = pd.Timestamp(CALIBRACAO_FIM)

    obs = [("falha", pd.Timestamp(a), pd.Timestamp(b)) for a, b in FALHAS]
    prov = [("provavel", r.ini, r.fim) for r in ep.itertuples()
            if pd.isna(r.falha_rotulada) and r.ini.strftime("%Y-%m-%d") in PROVAVEIS]
    # episódios sem rótulo que não são 'prováveis' (março, ciclagem 13 jul): desconhecidos
    desc = [(r.ini, r.fim) for r in ep.itertuples()
            if pd.isna(r.falha_rotulada) and r.ini.strftime("%Y-%m-%d") not in PROVAVEIS
            and r.ini >= ini_teste]

    linhas, fps = [], []
    for nome, lim in (("informacao", 1), ("aviso", 2), ("critico", 3)):
        prev = intervalos_previstos(niv["sistema"], niv["com_dado"], lim)
        prev = [(a, b) for a, b in prev if b > ini_teste]
        for tipo, i0, i1 in obs + prov:
            if i0 < ini_teste:
                continue
            h_dado = niv.loc[(niv.index >= i0 - pd.Timedelta(hours=24)) & (niv.index < i1),
                             "com_dado"].mean()
            tocam = [(a, b) for a, b in prev if sobrepoe(a, b, i0 - pd.Timedelta(hours=24), i1) > 0]
            dentro = [(a, b) for a, b in tocam if sobrepoe(a, b, i0, i1) > 0 or b <= i0]
            ant = [(i0 - a).total_seconds() / 3600 for a, b in dentro]
            linhas.append({
                "nivel": nome, "tipo": tipo, "inicio": i0, "dur_h": round((i1 - i0).total_seconds() / 3600, 1),
                "dado_24h_antes_e_dentro": round(float(h_dado), 2),
                "detectado": bool(dentro),
                "antecedencia_h": round(max(ant), 1) if dentro else np.nan,
                "cumpre_2h": bool(dentro and max(ant) >= ANTECEDENCIA_MIN_H),
                "cobertura_do_evento": round(sum(sobrepoe(a, b, i0, i1) for a, b in dentro)
                                             / ((i1 - i0).total_seconds() / 3600), 2),
            })
        # falsos positivos: intervalos previstos que não tocam evento nem episódio desconhecido
        conhecidos = [(i0 - pd.Timedelta(hours=24), i1) for _, i0, i1 in obs + prov] \
            + [(a - pd.Timedelta(hours=24), b + pd.Timedelta(hours=24)) for a, b in desc]
        fp = [(a, b) for a, b in prev
              if not any(sobrepoe(a, b, c0, c1) > 0 for c0, c1 in conhecidos)]
        fps.append({"nivel": nome, "intervalos_previstos": len(prev), "falsos_positivos": len(fp),
                    "fp_por_semana": round(len(fp) / (niv.index[niv["com_dado"]].to_series()
                                                      .ge(ini_teste).sum() / 24 / 7), 2)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "avaliacao_intervalos.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))
    print()
    print(pd.DataFrame(fps).to_string(index=False))
    for nome in NIVEL:
        t = tab[(tab.nivel == nome)]
        for tipo in ("falha", "provavel"):
            s = t[t.tipo == tipo]
            print(f"{nome:10s} {tipo:9s}: detectados {int(s.detectado.sum())}/{len(s)} "
                  f"| >=2 h antes: {int(s.cumpre_2h.sum())}")


if __name__ == "__main__":
    main()
