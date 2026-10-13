import numpy as np
import pandas as pd

from config import CONGELADO_MIN_AMOSTRAS, INTERPOLAR_MAX_S, SEG_POR_LINHA

ANALOGICOS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAIS = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
            "Caudal_impulses"]


def marcar_congelado(df: pd.DataFrame) -> pd.Series:
    """True nas linhas em que todos os sensores analógicos repetem o valor anterior por
    >= CONGELADO_MIN_AMOSTRAS linhas seguidas (registrador travado: o dado é uma cópia,
    não uma medição)."""
    igual = df[ANALOGICOS].diff().abs().sum(axis=1) == 0
    corrida = (igual != igual.shift()).cumsum()
    tamanho = igual.groupby(corrida).transform("size")
    mask = igual & (tamanho >= CONGELADO_MIN_AMOSTRAS)
    # a primeira linha da corrida (a última medição real) também entra no trecho travado,
    # mas ela é válida; só as cópias são invalidadas
    return mask


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Grade regular de SEG_POR_LINHA s com colunas de qualidade:
    - `presente`: havia uma linha original naquele instante;
    - `congelado`: linha original descartada por ser cópia (registrador travado);
    - `interpolado`: valor preenchido numa lacuna curta;
    - `segmento`: id do trecho contínuo com dado válido (muda a cada lacuna longa).
    Lacunas longas e trechos congelados ficam NaN (não se inventa dado)."""
    d = df.copy()
    congelado = marcar_congelado(d)
    d.loc[congelado, ANALOGICOS + DIGITAIS] = np.nan

    freq = f"{SEG_POR_LINHA}s"
    d = d.set_index("timestamp")
    anal = d[ANALOGICOS].resample(freq).mean()
    dig = d[DIGITAIS].resample(freq).last()
    out = pd.concat([anal, dig], axis=1)
    out["presente"] = d["TP3"].resample(freq).count() > 0
    out["congelado"] = congelado.set_axis(d.index).resample(freq).max().fillna(False).astype(bool)

    limite = int(INTERPOLAR_MAX_S / SEG_POR_LINHA)
    faltava = out["TP3"].isna()
    out[ANALOGICOS] = out[ANALOGICOS].interpolate(limit=limite, limit_area="inside")
    out[DIGITAIS] = out[DIGITAIS].ffill(limit=limite)
    out["interpolado"] = faltava & out["TP3"].notna()

    valido = out["TP3"].notna()
    out["segmento"] = ((valido != valido.shift()) & valido).cumsum().where(valido)
    return out.reset_index()
