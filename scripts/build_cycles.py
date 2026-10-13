# scripts/build_cycles.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from config import PARQUET_LIMPO, RESULTADOS
from src.ciclos import extrair_ciclos

d = pd.read_parquet(PARQUET_LIMPO)
d = d[d["TP3"].notna()].reset_index(drop=True)
c = extrair_ciclos(d)
RESULTADOS.mkdir(parents=True, exist_ok=True)
c.to_parquet(RESULTADOS / "ciclos.parquet")
print(f"{len(c):,} ciclos")