import pandas as pd

from config import CENARIOS, FALHAS, HORIZONTE_H
from src.risk import nivel


def simular(oof: pd.Series, nome_cenario: str) -> pd.DataFrame:
    """Linha do tempo de um cenário com probabilidades fora da amostra.
    `oof` é a probabilidade de falha por hora (saída de src.model.validar)."""
    ini, fim = pd.Timestamp(CENARIOS[nome_cenario][0]), pd.Timestamp(CENARIOS[nome_cenario][1])
    p = oof[(oof.index >= ini) & (oof.index < fim)]
    return pd.DataFrame({"probabilidade": p, "nivel": nivel(p)})


def resumo(linha: pd.DataFrame, nome_cenario: str) -> dict:
    """Quando o sistema emite o primeiro alerta alto e com quanta antecedência
    antes da falha (se houver falha no cenário)."""
    ini, fim = pd.Timestamp(CENARIOS[nome_cenario][0]), pd.Timestamp(CENARIOS[nome_cenario][1])
    alertas = linha.index[linha["nivel"] == "alto"]
    falhas_no_cenario = [pd.Timestamp(f[0]) for f in FALHAS if ini <= pd.Timestamp(f[0]) < fim]
    res = {"cenario": nome_cenario, "horas_avaliadas": len(linha),
           "horas_alerta_alto": len(alertas)}
    if falhas_no_cenario:
        falha = falhas_no_cenario[0]
        antes = alertas[alertas < falha]
        res["falha"] = str(falha)
        res["antecedencia_h"] = (falha - antes[0]).total_seconds() / 3600 if len(antes) else None
    else:
        res["falha"] = None
        res["antecedencia_h"] = None
    res["horizonte_h"] = HORIZONTE_H
    return res
