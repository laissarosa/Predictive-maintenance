import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import FALHAS, PARQUET_LIMPO, RESULTADOS
from src.data import carregar_dados
from src.limpeza import limpar, marcar_congelado


def main():
    bruto = carregar_dados()
    cong = marcar_congelado(bruto)
    print(f"Linhas brutas: {len(bruto):,}; congeladas (cópias): {int(cong.sum()):,}"
          f" ({cong.mean():.1%})")

    limpo = limpar(bruto)
    PARQUET_LIMPO.parent.mkdir(parents=True, exist_ok=True)
    limpo.to_parquet(PARQUET_LIMPO, index=False)
    valido = limpo["TP3"].notna()
    print(f"Grade regular: {len(limpo):,} instantes; válidos {valido.mean():.1%}; "
          f"interpolados {limpo['interpolado'].mean():.2%}; "
          f"congelados {limpo['congelado'].mean():.1%}; "
          f"trechos contínuos: {int(limpo['segmento'].max())}")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    linhas = []
    for ini, fim in FALHAS:
        a, b = pd.Timestamp(ini), pd.Timestamp(fim)
        for nome, x, y in (("24h_antes", a - pd.Timedelta(hours=24), a), ("dentro", a, b)):
            s = limpo[(limpo["timestamp"] >= x) & (limpo["timestamp"] < y)]
            linhas.append({"falha": ini, "janela": nome,
                           "cobertura_bruta": bruto["timestamp"].between(x, y).sum()
                                              * 10 / (y - x).total_seconds(),
                           "cobertura_limpa": s["TP3"].notna().mean(),
                           "congelado": s["congelado"].mean()})
    q = pd.DataFrame(linhas)
    q.to_csv(RESULTADOS / "qualidade_falhas.csv", index=False)
    print("\nQualidade ao redor das falhas (cobertura de dado válido):")
    print(q.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
