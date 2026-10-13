import numpy as np
import pandas as pd

from config import CICLO_GAP_MAX_S, CICLO_MIN_AMOSTRAS, LIMIAR_CARGA


def _inclinacao(t_min: np.ndarray, y: np.ndarray) -> float:
    """Inclinação da reta y(t) em unidades por minuto; NaN se não houver dados suficientes."""
    ok = ~np.isnan(y)
    if ok.sum() < CICLO_MIN_AMOSTRAS:
        return np.nan
    return float(np.polyfit(t_min[ok], y[ok], 1)[0])


def extrair_ciclos(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ciclo de carga do compressor.

    Ciclo = um período de carga (motor sob carga) precedido pelo período sem carga
    (ocioso ou desligado). Atributos do período sem carga descrevem o descanso e a
    queda de pressão; atributos da carga descrevem o esforço do compressor.
    Ciclos com lacunas nos dados (intervalo entre amostras > CICLO_GAP_MAX_S) são descartados.
    """
    d = df[["timestamp", "Motor_current", "TP2", "Reservoirs", "Oil_temperature",
            "Caudal_impulses"]].copy()
    d["carga"] = d["Motor_current"] > LIMIAR_CARGA
    d["run"] = (d["carga"] != d["carga"].shift()).cumsum()
    d["t_min"] = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds() / 60
    d["gap"] = d["timestamp"].diff().dt.total_seconds().fillna(0)

    grupos = d.groupby("run")
    info = pd.DataFrame({
        "carga": grupos["carga"].first(),
        "ini": grupos["timestamp"].first(),
        "fim": grupos["timestamp"].last(),
        "n": grupos.size(),
        "gap_max": grupos["gap"].max(),
    })
    info = info.reset_index()
    # descanso = run não-carga imediatamente antes de cada run de carga
    cargas = info[info["carga"] & (info["n"] >= CICLO_MIN_AMOSTRAS)].copy()
    cargas = cargas[cargas["run"] > 1]

    partes = dict(list(d.groupby("run")))
    linhas = []
    for _, c in cargas.iterrows():
        pre = partes[c["run"] - 1]
        car = partes[c["run"]]
        dur_carga = (c["fim"] - c["ini"]).total_seconds() / 60
        dur_desc = (pre["timestamp"].iloc[-1] - pre["timestamp"].iloc[0]).total_seconds() / 60
        if max(pre["gap"].max(), car["gap"].max()) > CICLO_GAP_MAX_S:
            continue  # ciclo com lacuna nos dados: duração e inclinações não são confiáveis

        f = {
            "ini": c["ini"],
            "dur_carga_min": dur_carga,
            "corrente_carga": car["Motor_current"].mean(),
            "corrente_carga_std": car["Motor_current"].std(),
            "tp2_carga": car["TP2"].mean(),
            "tp2_pico": car["TP2"].max(),
            "tp2_subida_bar_min": _inclinacao(car["t_min"].to_numpy(), car["TP2"].to_numpy()),
            "caudal_carga": car["Caudal_impulses"].mean(),
            "oleo_ini": car["Oil_temperature"].iloc[0],
            "oleo_fim": car["Oil_temperature"].iloc[-1],
            "descanso_min": dur_desc,
            "reserv_fim_descanso": pre["Reservoirs"].iloc[-1],
            "queda_reserv_descanso": _inclinacao(pre["t_min"].to_numpy(),
                                                 pre["Reservoirs"].to_numpy()),
        }
        f["oleo_subida_carga"] = f["oleo_fim"] - f["oleo_ini"]
        linhas.append(f)

    return pd.DataFrame(linhas).set_index("ini").sort_index()
