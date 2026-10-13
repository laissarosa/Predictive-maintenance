"""Lógica do demo (sem Streamlit): escores por hora, limiar, alerta e texto do motivo."""
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent

COMPONENTES = ["oleo", "pressao", "motor", "secador", "valvulas", "vazao"]
NOMES = {"oleo": "Óleo", "pressao": "Pressão / ar", "motor": "Motor",
         "secador": "Secador", "valvulas": "Válvulas", "vazao": "Vazão"}
INSPECAO = {
    "oleo": "temperatura e nível do óleo, possível vazamento de óleo",
    "pressao": "linhas de ar e reservatórios, possível vazamento de ar",
    "motor": "motor e corrente de carga",
    "secador": "secador de ar, torres e válvula de purga",
    "valvulas": "válvulas de carga/descarga e do compressor",
    "vazao": "fluxo de ar e consumo",
}

# falhas informadas pela empresa (UCI MetroPT3)
FALHAS = {
    "Falha 1 - 18/04/2020": ("2020-04-18 00:00", "2020-04-18 23:59"),
    "Falha 2 - 29-30/05/2020": ("2020-05-29 23:30", "2020-05-30 06:00"),
    "Falha 3 - 05-07/06/2020": ("2020-06-05 10:00", "2020-06-07 14:30"),
    "Falha 4 - 15/07/2020": ("2020-07-15 14:30", "2020-07-15 19:00"),
}
CONTROLE = ("Controle normal - mar/2020", "2020-03-10", "2020-03-14")

PERSIST_H = 2            # horas seguidas acima do limiar para disparar o alerta
MIN_JANELAS_HORA = 3     # mínimo de janelas de 10 min na hora para ela contar
QUANTIL_PADRAO = 0.99    # limiar definido ANTES de ver as falhas


def achar(nome: str):
    for pasta in (AQUI / "dados_demo", AQUI / "reports" / "resultados",
                  AQUI.parent / "reports" / "resultados"):
        p = pasta / nome
        if p.exists():
            return p
    return None


def carregar_horario(caminho) -> tuple[pd.DataFrame, pd.Series]:
    """ae_escores.parquet (uma linha por janela de 10 min) -> mediana por hora e
    indicador de hora de validação (dia normal separado do treino)."""
    e = pd.read_parquet(caminho)
    r = e[COMPONENTES + ["sistema"]].resample("1h")
    h = r.median().where(r.count() >= MIN_JANELAS_HORA)
    val = e["validacao"].astype(float).resample("1h").max().fillna(0).astype(bool)
    return h, val.reindex(h.index, fill_value=False)


def limiar(h: pd.DataFrame, val: pd.Series, q: float = QUANTIL_PADRAO) -> float:
    return float(h.loc[val, "sistema"].dropna().quantile(q))


def alertas(sistema: pd.Series, lim: float) -> tuple[pd.Series, pd.Series]:
    """(acima, alerta): acima = escore > limiar nesta hora; alerta = PERSIST_H horas seguidas.
    Só olha o passado."""
    acima = (sistema > lim) & sistema.notna()
    a = acima.astype(float).where(sistema.notna())
    alerta = a.rolling(PERSIST_H, min_periods=PERSIST_H).min().fillna(0).astype(bool)
    return acima, alerta


def estado_em(h, acima, alerta, t) -> str:
    if pd.isna(h.loc[t, "sistema"]):
        return "sem_dado"
    if alerta.loc[t]:
        return "alerta"
    if acima.loc[t]:
        return "atencao"
    return "normal"


def horas_em_alerta(alerta: pd.Series, t) -> int:
    n = 0
    for v in alerta.loc[:t].iloc[::-1]:
        if not v:
            break
        n += 1
    return n


def principais(h: pd.DataFrame, t, n: int = 3, janela_h: int = 3) -> list[tuple[str, float]]:
    """Componentes mais afastados do normal (mediana das últimas `janela_h` horas)."""
    ult = h.loc[:t, COMPONENTES].iloc[-janela_h:]
    med = ult.median().dropna().sort_values(ascending=False)
    return [(c, float(v)) for c, v in med.head(n).items()]


def texto_motivo(estado: str, tops: list[tuple[str, float]], horas: int) -> str:
    if estado == "sem_dado":
        return "Sem dado válido neste instante (sensor travado ou lacuna na coleta)."
    if estado == "normal":
        return "O comportamento dos sensores está dentro do padrão aprendido em operação normal."
    lista = ", ".join(f"{NOMES[c]} ({v:.1f}x o erro típico)" for c, v in tops)
    if estado == "atencao":
        return (f"O escore passou do limiar nesta hora, mas ainda não houve confirmação de "
                f"{PERSIST_H} h seguidas. Componentes mais afastados: {lista}.")
    insp = "; ".join(INSPECAO[c] for c, _ in tops[:2])
    return (f"Desvio sustentado do comportamento normal há {horas} h. Componentes mais "
            f"afastados: {lista}. Sugestão de onde inspecionar (hipótese, não diagnóstico): {insp}.")


def recorte(h: pd.DataFrame, ini, fim) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(ini).floor("h"), pd.Timestamp(fim).ceil("h"), freq="1h")
    return h.reindex(idx)
