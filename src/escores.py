import numpy as np
import pandas as pd

from config import (BASE_ATRASO_H, BASE_DIAS, BASE_MIN_DIAS, COMPONENTES, ESCALA_REF_DIAS,
                    ESCORE_MAX, HORA_MIN_AMOSTRAS_CARGA, HORA_MIN_COBERTURA, LIMIAR_CARGA,
                    SEG_POR_LINHA)

# atributo -> componente. Atributos de pressão/corrente/óleo são condicionados ao estado
# (sob carga) para que a mudança de demanda do trem não pareça anomalia.
ATRIBUTOS = {
    "oleo": ["oleo_media", "oleo_carga", "oleo_subida_1h", "oleo_nivel"],
    "pressao": ["tp3_media", "tp3_std", "tp2_carga", "tp3_menos_reserv", "h1_media", "lps_frac"],
    "motor": ["corrente_carga", "corrente_carga_std", "duty_cycle", "transicoes"],
    "secador": ["dv_carga", "torres_trocas", "psw_media"],
    "valvulas": ["valv_desacordo", "comp_trocas", "mpg_media"],
    "vazao": ["caudal_carga", "caudal_media"],
}


def atributos_horarios(limpo: pd.DataFrame) -> pd.DataFrame:
    """Atributos causais por hora a partir do dataset limpo. Hora com cobertura baixa = NaN."""
    d = limpo.set_index("timestamp")
    carga = (d["Motor_current"] > LIMIAR_CARGA).where(d["Motor_current"].notna())
    sob = carga == 1
    r = lambda s: s.resample("1h")

    def cond(col):  # média na hora só enquanto sob carga
        x = d[col].where(sob)
        m = r(x).mean()
        return m.where(r(sob.astype(int)).sum() >= HORA_MIN_AMOSTRAS_CARGA)

    def trocas(s):
        return r(s.diff().abs()).sum()

    f = pd.DataFrame({
        "oleo_media": r(d["Oil_temperature"]).mean(),
        "oleo_carga": cond("Oil_temperature"),
        "oleo_nivel": r(d["Oil_level"]).mean(),
        "tp3_media": r(d["TP3"]).mean(),
        "tp3_std": r(d["TP3"]).std(),
        "tp2_carga": cond("TP2"),
        "tp3_menos_reserv": r((d["TP3"] - d["Reservoirs"]).abs()).mean(),
        "h1_media": r(d["H1"]).mean(),
        "lps_frac": r(d["LPS"]).mean(),
        "corrente_carga": cond("Motor_current"),
        "corrente_carga_std": r(d["Motor_current"].where(sob)).std(),
        "duty_cycle": r(carga).mean(),
        "transicoes": trocas(carga),
        "dv_carga": cond("DV_pressure"),
        "torres_trocas": trocas(d["Towers"]),
        "psw_media": r(d["Pressure_switch"]).mean(),
        "valv_desacordo": r((d["DV_eletric"] != carga).where(carga.notna())).mean(),
        "comp_trocas": trocas(d["COMP"]),
        "mpg_media": r(d["MPG"]).mean(),
        "caudal_carga": cond("Caudal_impulses"),
        "caudal_media": r(d["Caudal_impulses"]).mean(),
    })
    f["oleo_subida_1h"] = f["oleo_media"].diff()
    esperado = 3600 / SEG_POR_LINHA
    f["cobertura"] = r(d["TP3"]).count() / esperado
    cols = f.columns.difference(["cobertura"])
    f.loc[f["cobertura"] < HORA_MIN_COBERTURA, cols] = np.nan
    return f


def _mad(w: np.ndarray) -> float:
    m = np.nanmedian(w)
    return float(np.nanmedian(np.abs(w - m)))


def z_robusto(f: pd.DataFrame) -> pd.DataFrame:
    """z de cada atributo contra mediana/MAD da MESMA hora do dia nos BASE_DIAS dias anteriores
    (a referência termina BASE_ATRASO_H antes). Comparar hora com hora controla o ciclo diário
    de operação do trem. O piso da escala vem dos primeiros ESCALA_REF_DIAS dias."""
    f = f.astype(float)
    passo = BASE_ATRASO_H // 24
    med = pd.DataFrame(index=f.index, columns=f.columns, dtype=float)
    mad = med.copy()
    for h in range(24):
        sh = f[f.index.hour == h]
        r = sh.rolling(BASE_DIAS, min_periods=BASE_MIN_DIAS)
        med.loc[sh.index] = r.median().shift(passo)
        mad.loc[sh.index] = r.apply(_mad, raw=True).shift(passo)
    ref = f[f.index < f.index[0] + pd.Timedelta(days=ESCALA_REF_DIAS)]
    piso = 0.2 * ref.apply(lambda c: _mad(c.to_numpy())) * 1.4826
    piso = piso.where(piso > 0, 0.01 * ref.abs().median() + 1e-3)
    escala = (1.4826 * mad).clip(lower=piso, axis=1)
    return ((f - med) / escala).abs().clip(upper=ESCORE_MAX)


def _agregar(z: pd.DataFrame) -> pd.Series:
    """Segundo maior z entre os atributos do componente (um atributo isolado não dispara
    sozinho; com um só atributo, usa o próprio)."""
    ordenado = np.sort(z.to_numpy(), axis=1)  # NaN vai para o fim
    validos = z.notna().sum(axis=1).to_numpy()
    alvo = np.clip(validos - 2, 0, None)
    v = ordenado[np.arange(len(z)), alvo]
    return pd.Series(np.where(validos > 0, v, np.nan), index=z.index)


def escores_componentes(limpo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escore por componente = maior z entre seus atributos. Devolve (escores, z por atributo)."""
    f = atributos_horarios(limpo)
    z = z_robusto(f.drop(columns="cobertura"))
    esc = pd.DataFrame({c: _agregar(z[a]) for c, a in ATRIBUTOS.items()}).astype(float)
    assert set(esc.columns) == set(COMPONENTES)
    return esc, z
