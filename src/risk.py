import numpy as np
import pandas as pd

from config import RISCO_ALTO, RISCO_MEDIO


def nivel(prob: pd.Series) -> pd.Series:
    """Converte probabilidade de falha em nível de risco (NaN continua NaN)."""
    niveis = np.select(
        [prob >= RISCO_ALTO, prob >= RISCO_MEDIO],
        ["alto", "médio"],
        default="baixo",
    )
    return pd.Series(niveis, index=prob.index).where(prob.notna())
