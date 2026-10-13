import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ANTECEDENCIA_MIN_H, FALHAS, INTERVALO_FUNDIR_H, RESULTADOS

PERSIST_H = 2
QUANTIS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]


def sobrepoe(a0, a1, b0, b1):
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds() / 3600)


def intervalos(alerta):
    out = []
    for h in alerta.index[alerta]:
        if out and h - out[-1][1] <= pd.Timedelta(hours=INTERVALO_FUNDIR_H + 1):
            out[-1][1] = h
        else:
            out.append([h, h])
    return [(a, b + pd.Timedelta(hours=1)) for a, b in out]


def main():
    e = pd.read_parquet(RESULTADOS / "ae_escores.parquet")
    s = e["sistema"]
    h = s.resample("1h").median().where(s.resample("1h").count() >= 3)
    val = e["validacao"].resample("1h").max().fillna(False).astype(bool)
    normal = h[val].dropna()

    ep = pd.read_parquet(RESULTADOS / "episodios.parquet")
    ep = ep[~ep["lacunas"]]
    d24 = pd.Timedelta(hours=24)
    conhecidos = [(pd.Timestamp(a) - d24, pd.Timestamp(b) + d24) for a, b in FALHAS]
    conhecidos += [(r.ini - d24, r.fim + d24) for r in ep.itertuples()]
    longe = pd.Series([not any(c0 <= t < c1 for c0, c1 in conhecidos) for t in h.index],
                      index=h.index) & h.notna()
    semanas = longe.sum() / 24 / 7

    linhas = []
    for q in QUANTIS:
        lim = float(normal.quantile(q))
        acima = (h > lim).astype(float).where(h.notna())
        alerta = acima.rolling(PERSIST_H, min_periods=PERSIST_H).min().fillna(0).astype(bool)
        prev = intervalos(alerta)
        det, lead2 = 0, 0
        for ini, fim in FALHAS:
            a, b = pd.Timestamp(ini), pd.Timestamp(fim)
            t = [(x, y) for x, y in prev if sobrepoe(x, y, a - d24, b) > 0]
            if t:
                det += 1
                lead2 += (a - t[0][0]).total_seconds() / 3600 >= ANTECEDENCIA_MIN_H
        fp = [(x, y) for x, y in prev
              if not any(sobrepoe(x, y, c0, c1) > 0 for c0, c1 in conhecidos)]
        linhas.append({"quantil": q, "limiar": round(lim, 2), "falhas_detectadas": f"{det}/4",
                       "com_2h_antes": lead2, "falsos_intervalos": len(fp),
                       "falsos_por_semana": round(len(fp) / semanas, 2)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "ae_curva_operacao.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Semanas normais avaliadas: {semanas:.1f} | persistência: {PERSIST_H} h\n")
    print(tab.to_string(index=False))
    print("\nTodos os pontos foram avaliados DEPOIS de ver as falhas: reporte como curva de "
          "operação (trade-off), nunca como o desempenho de um limiar escolhido antes.")


if __name__ == "__main__":
    main()
