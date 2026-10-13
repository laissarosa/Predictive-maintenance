import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from config import ALERTA_PERCENTIL, FALHAS, JANELA_ANTES_DIAS, JANELA_DEPOIS_DIAS

ATRIBUTOS = ["dur_carga_min", "corrente_carga", "corrente_carga_std", "tp2_carga", "tp2_pico",
             "tp2_subida_bar_min", "caudal_carga", "oleo_ini", "oleo_fim", "descanso_min",
             "reserv_fim_descanso", "queda_reserv_descanso", "oleo_subida_carga"]


def janelas_falha() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Período em torno de cada falha, excluído do treino de 'normal'."""
    return [(pd.Timestamp(i) - pd.Timedelta(days=JANELA_ANTES_DIAS),
             pd.Timestamp(f) + pd.Timedelta(days=JANELA_DEPOIS_DIAS)) for i, f in FALHAS]


def mascara_falha(index: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros(len(index), dtype=bool)
    for a, b in janelas_falha():
        m |= (index >= a) & (index <= b)
    return m


def treinar(ciclos: pd.DataFrame) -> tuple[object, float]:
    """Aprende o padrão 'normal' (sem as janelas de falha) e define o limiar de alarme
    como o percentil ALERTA_PERCENTIL dos próprios ciclos normais."""
    normal = ciclos[~mascara_falha(ciclos.index)]
    modelo = make_pipeline(
        RobustScaler(),
        IsolationForest(n_estimators=300, random_state=0),
    )
    x = normal[ATRIBUTOS].fillna(normal[ATRIBUTOS].median())
    modelo.fit(x)
    escore_normal = escores(modelo, normal, mediana=normal[ATRIBUTOS].median())
    limiar = float(np.percentile(escore_normal, ALERTA_PERCENTIL))
    return modelo, limiar


def escores(modelo, ciclos: pd.DataFrame, mediana: pd.Series) -> pd.Series:
    """Escore de anomalia: maior = mais diferente do normal."""
    x = ciclos[ATRIBUTOS].fillna(mediana)
    return pd.Series(-modelo.score_samples(x), index=ciclos.index, name="anomalia")
