import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import MODELOS, RESULTADOS
from src.data import carregar_dados
from src.model import metricas, treinar_final, validar


def main():
    df = carregar_dados()
    res = validar(df)
    y, oof = res["y"], res["oof"]

    # comparar todos nas mesmas horas: as que têm previsão de todos os modelos
    comum = oof.notna().all(axis=1)
    y, oof = y[comum], oof[comum]
    print(f"Horas comparadas: {len(y)} (positivas: {int(y.sum())}, prevalência: {y.mean():.3f})")

    tabela = pd.DataFrame({c: metricas(y, oof[c]) for c in oof.columns}).T
    tabela = tabela.sort_values("PR-AUC", ascending=False)
    print("Validação por falha (fora da amostra):")
    print(tabela.to_string())

    base = "regra_base_duty_cycle"
    ml = tabela.drop(index=base)
    melhor = ml["PR-AUC"].idxmax()
    pr_melhor, pr_base = ml.loc[melhor, "PR-AUC"], tabela.loc[base, "PR-AUC"]
    if pr_melhor > pr_base:
        print(f"\nDecisão: {melhor} supera a regra base em PR-AUC ({pr_melhor:.3f} > {pr_base:.3f}).")
    else:
        print(f"\nDecisão: nenhum modelo supera a regra base em PR-AUC "
              f"(melhor modelo: {melhor} = {pr_melhor:.3f}; regra base = {pr_base:.3f}).")
        print("O modelo salvo abaixo serve só para análise, não como solução.")
    treinar_final(df, melhor)
    print(f"Modelo salvo em: {MODELOS / (melhor + '.joblib')}")

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    oof.assign(falha_proxima=y).to_parquet(RESULTADOS / "oof.parquet")
    tabela.to_csv(RESULTADOS / "metricas.csv")
    print(f"Probabilidades e métricas salvas em: {RESULTADOS}")


if __name__ == "__main__":
    main()
