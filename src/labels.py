import pandas as pd

from config import FALHAS, HORIZONTE_H


def rotular(feats: pd.DataFrame, horizonte_h: int = HORIZONTE_H) -> pd.Series:
    """Rótulo por hora: 1 se uma falha começa nas próximas `horizonte_h` horas,
    0 caso contrário. Horas dentro de uma falha ficam NaN (já em falha, não é previsão)."""
    t = feats.index.to_series()
    y = pd.Series(0.0, index=feats.index)

    for ini, fim in FALHAS:
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        antes = (t >= ini - pd.Timedelta(hours=horizonte_h)) & (t < ini)
        y[antes] = 1.0
        y[(t >= ini) & (t <= fim)] = float("nan")

    return y.rename("falha_proxima")
