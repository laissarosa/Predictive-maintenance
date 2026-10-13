import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import COMPONENTES, RESULTADOS


def perfil(z: pd.DataFrame, ini, fim) -> pd.Series:
    """Mediana do |z| de cada atributo durante o episódio (horas com dado)."""
    return z[(z.index >= ini.floor("h")) & (z.index <= fim)].median()


def cosseno(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ep = pd.read_parquet(RESULTADOS / "episodios.parquet")
    ep = ep[~ep["lacunas"]].reset_index(drop=True)
    niv = pd.read_parquet(RESULTADOS / "alertas.parquet")
    esc = pd.read_parquet(RESULTADOS / "escores.parquet")
    z = pd.read_parquet(RESULTADOS / "z_atributos.parquet")

    perfis = {r.ini: perfil(z, r.ini, r.fim) for r in ep.itertuples()}
    rotulados = [r.ini for r in ep.itertuples() if pd.notna(r.falha_rotulada)]
    linhas = []
    for r in ep.itertuples():
        h = niv[(niv.index >= r.ini.floor("h")) & (niv.index <= r.fim) & niv["com_dado"]]
        p = perfis[r.ini]
        ref = pd.concat([perfis[i] for i in rotulados if i != r.ini], axis=1).median(axis=1)
        lin = {"ini": r.ini, "dur_h": round(r.duracao_h, 1), "tipo": r.tipo,
               "rotulado": pd.notna(r.falha_rotulada), "horas_com_dado": len(h)}
        if len(h):
            for k, nome in ((1, "info+"), (2, "aviso+"), (3, "critico")):
                lin[nome] = round((h["sistema"] >= k).mean(), 2)
            sc = esc.loc[h.index].max()
            lin["comp_pico"] = ", ".join(f"{c}:{sc[c]:.0f}" for c in sc.sort_values(ascending=False).index[:3])
            lin["semelhanca_rotulados"] = round(cosseno(p, ref), 2)
        linhas.append(lin)
    tab = pd.DataFrame(linhas)
    tab.to_csv(RESULTADOS / "episodios_nao_rotulados.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))

    print("\nAtributos mais anômalos (mediana |z|) por episódio:")
    for r in ep.itertuples():
        p = perfis[r.ini].dropna().sort_values(ascending=False).head(5)
        print(f"  {r.ini:%Y-%m-%d} {'[ROT]' if pd.notna(r.falha_rotulada) else '     '}",
              ", ".join(f"{k}={v:.1f}" for k, v in p.items()))


if __name__ == "__main__":
    main()
