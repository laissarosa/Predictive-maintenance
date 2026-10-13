import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import CENARIOS, RESULTADOS
from src import report
from src.simulador import resumo, simular


def main():
    dados = pd.read_parquet(RESULTADOS / "oof.parquet")
    dados.pop("falha_proxima")
    tabela = pd.read_csv(RESULTADOS / "metricas.csv", index_col=0)
    modelo = next(c for c in tabela.index if c != "regra_base_duty_cycle")
    oof = dados[modelo]

    linhas = []
    for nome in CENARIOS:
        linha = simular(oof, nome)
        if linha.empty:
            print(f"{nome}: sem previsões fora da amostra no período")
            continue
        report.simulacao(linha, nome)
        linhas.append(resumo(linha, nome))

    print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == "__main__":
    main()
