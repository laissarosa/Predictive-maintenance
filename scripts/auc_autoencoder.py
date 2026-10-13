import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import roc_auc_score

from config import FALHAS, RESULTADOS


def main():
    e = pd.read_parquet(RESULTADOS / "ae_escores.parquet")
    s = e["sistema"]
    n = s.resample("1h").count()
    h = s.resample("1h").median().where(n >= 3)
    v = e["validacao"].resample("1h").max().fillna(False).astype(bool)
    normal = h[v].dropna()
    print(f"Horas normais de validação: {len(normal)} | limiar (quantil 0,99): "
          f"{normal.quantile(0.99):.2f}\n")
    for a, b in FALHAS:
        d = h[(h.index >= pd.Timestamp(a)) & (h.index <= pd.Timestamp(b))].dropna()
        if len(d) < 3:
            print(f"{a}: poucos dados dentro da falha ({len(d)} h)")
            continue
        y = [0] * len(normal) + [1] * len(d)
        auc = roc_auc_score(y, list(normal) + list(d))
        print(f"{a} | horas na falha: {len(d):3d} | AUC: {auc:.3f} | "
              f"mediana na falha: {d.median():.1f} | mediana normal: {normal.median():.1f} | "
              f"acima do limiar: {(d > normal.quantile(0.99)).mean():.0%}")


if __name__ == "__main__":
    main()
