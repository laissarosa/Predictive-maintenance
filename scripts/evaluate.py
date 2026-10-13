import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import FIGURAS, RESULTADOS
from src import report


def main():
    dados = pd.read_parquet(RESULTADOS / "oof.parquet")
    y = dados.pop("falha_proxima")
    oof = dados
    tabela = pd.read_csv(RESULTADOS / "metricas.csv", index_col=0)
    modelo = next(c for c in tabela.index if c != "regra_base_duty_cycle")

    report.curva_precisao_recall(y, oof)
    report.probabilidade_por_falha(oof, modelo)
    cm = report.matriz_confusao(y, oof[modelo])

    print(f"Matriz de confusão ({modelo}, limiar alto):")
    print(pd.DataFrame(cm, index=["real: sem falha", "real: falha"],
                       columns=["previsto: sem", "previsto: alerta"]).to_string())
    print(f"\nFiguras salvas em: {FIGURAS}")


if __name__ == "__main__":
    main()
