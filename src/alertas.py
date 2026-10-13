import math

import pandas as pd

from config import (ALERTA_BASE_AVISO_H, ALERTA_BASE_CRITICO_H, ALERTA_COMPONENTES_PARA_CRITICO,
                    ALERTA_INFO_H, ALERTA_Z_AVISO, ALERTA_Z_CRITICO, ALERTA_Z_INFO, CALIBRACAO_FIM,
                    CALIBRACAO_QUANTIS, COMPONENTES, SISTEMA_INFO_COMPONENTES, SISTEMA_INFO_H)

NIVEIS = ["normal", "informacao", "aviso", "critico"]  # ordem crescente de gravidade


def horas_exigidas(severidade: int) -> dict[str, int]:
    return {"informacao": ALERTA_INFO_H,
            "aviso": math.ceil(ALERTA_BASE_AVISO_H / severidade),
            "critico": math.ceil(ALERTA_BASE_CRITICO_H / severidade)}


def _sustentado(acima: pd.Series, horas: int) -> pd.Series:
    """True quando a condição vale nas últimas `horas` horas seguidas (só olha o passado)."""
    return acima.astype(int).rolling(horas, min_periods=horas).min().fillna(0).astype(bool)


def calibrar_limiares(escores: pd.DataFrame, excluir=None) -> dict[str, tuple[float, float, float]]:
    """Limiares (informação, aviso, crítico) por componente: quantis dos escores no período
    de calibração (início da série). Nunca abaixo dos limiares globais z de config."""
    ref = escores[escores.index < pd.Timestamp(CALIBRACAO_FIM)]
    if excluir is not None:  # horas de episódios longos não entram no 'normal'
        ref = ref[~pd.Series(excluir, index=escores.index)[ref.index].to_numpy()]
    minimos = (ALERTA_Z_INFO, ALERTA_Z_AVISO, ALERTA_Z_CRITICO)
    out = {}
    for c in escores.columns:
        lim = [max(float(ref[c].quantile(q)), m) for q, m in zip(CALIBRACAO_QUANTIS, minimos)]
        for i in (1, 2):  # níveis distintos: cada um pelo menos 25% acima do anterior
            lim[i] = max(lim[i], 1.25 * lim[i - 1])
        out[c] = tuple(lim)
    return out


def nivel_componente(escore: pd.Series, severidade: int,
                     limiares=(ALERTA_Z_INFO, ALERTA_Z_AVISO, ALERTA_Z_CRITICO)) -> pd.Series:
    """Nível por hora de um componente a partir do escore de anomalia (z robusto)."""
    h = horas_exigidas(severidade)
    z_info, z_aviso, z_crit = limiares
    nivel = pd.Series(0, index=escore.index)
    nivel[_sustentado(escore >= z_info, h["informacao"])] = 1
    nivel[_sustentado(escore >= z_aviso, h["aviso"])] = 2
    nivel[_sustentado(escore >= z_crit, h["critico"])] = 3
    return nivel


def alertas(escores: pd.DataFrame, limiares: dict | None = None) -> pd.DataFrame:
    """`escores`: uma coluna por componente (nomes de config.COMPONENTES), uma linha por hora.
    Devolve o nível numérico (0-3) por componente e o nível do sistema."""
    limiares = limiares or {}
    niveis = pd.DataFrame({c: nivel_componente(escores[c], COMPONENTES[c]["severidade"],
                                               limiares.get(c, (ALERTA_Z_INFO, ALERTA_Z_AVISO, ALERTA_Z_CRITICO)))
                           for c in escores.columns})
    sistema = niveis.max(axis=1)
    # informação do sistema exige confirmação: vários componentes, por algumas horas
    info = _sustentado((niveis >= 1).sum(axis=1) >= SISTEMA_INFO_COMPONENTES, SISTEMA_INFO_H)
    sistema[(sistema == 1) & ~info] = 0
    varios = (niveis >= 2).sum(axis=1) >= ALERTA_COMPONENTES_PARA_CRITICO
    sistema[varios] = 3
    niveis["sistema"] = sistema
    return niveis


def rotulo(n: int) -> str:
    return NIVEIS[int(n)]
