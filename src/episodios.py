import pandas as pd

from config import (EPIS_COBERTURA_MIN, EPIS_FRAC_CARGA_CONT, EPIS_INTERRUPCAO_MIN, EPIS_MIN_H, FALHAS,
                    LIMIAR_CARGA, SEG_POR_LINHA)


def detectar_episodios(df: pd.DataFrame) -> pd.DataFrame:
    """Períodos de carga contínua longa (o compressor não consegue descarregar).

    Pausas sem carga menores que EPIS_INTERRUPCAO_MIN são absorvidas pelo episódio.
    Usa só o passado dentro do episódio; os limites são reais (início e fim da carga).
    """
    d = df[["timestamp", "Motor_current", "TP3", "Reservoirs", "Oil_temperature"]]
    carga = d["Motor_current"] > LIMIAR_CARGA
    run = (carga != carga.shift()).cumsum()
    g = d.groupby(run).agg(ini=("timestamp", "first"), fim=("timestamp", "last"),
                           n=("timestamp", "size"))
    g["carga"] = carga.groupby(run).first()
    g = g.reset_index(drop=True)

    pausa = pd.Timedelta(minutes=EPIS_INTERRUPCAO_MIN)
    blocos, atual = [], None
    for _, r in g[g["carga"]].iterrows():
        if atual is not None and r["ini"] - atual["fim"] <= pausa:
            atual["fim"] = r["fim"]
        else:
            if atual is not None:
                blocos.append(atual)
            atual = {"ini": r["ini"], "fim": r["fim"]}
    if atual is not None:
        blocos.append(atual)

    linhas = []
    for b in blocos:
        dur_h = (b["fim"] - b["ini"]).total_seconds() / 3600
        if dur_h < EPIS_MIN_H:
            continue
        seg = d[(d["timestamp"] >= b["ini"]) & (d["timestamp"] <= b["fim"])]
        linhas.append({
            "ini": b["ini"], "fim": b["fim"], "duracao_h": dur_h,
            "cobertura": len(seg) * SEG_POR_LINHA / (dur_h * 3600),
            "frac_carga": (seg["Motor_current"] > LIMIAR_CARGA).mean(),
            "corrente_media": seg["Motor_current"].mean(),
            "tp3_media": seg["TP3"].mean(),
            "reserv_media": seg["Reservoirs"].mean(),
            "oleo_max": seg["Oil_temperature"].max(),
        })
    out = pd.DataFrame(linhas)
    out["tipo"] = out["frac_carga"].map(lambda f: "continua" if f >= EPIS_FRAC_CARGA_CONT else "ciclagem_intensa")
    out["lacunas"] = out["cobertura"] < EPIS_COBERTURA_MIN
    out["falha_rotulada"] = [_falha_sobreposta(r.ini, r.fim) for r in out.itertuples()]
    return out


def _falha_sobreposta(ini: pd.Timestamp, fim: pd.Timestamp):
    """Início da falha rotulada que se sobrepõe ao episódio (ou None)."""
    for a, b in FALHAS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if ini <= b and fim >= a:
            return a
    return None
