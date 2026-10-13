import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (CENARIOS, FALHAS, HORIZONTE_H, JANELA_ANTES_DIAS,
                    JANELA_DEPOIS_DIAS, MODELOS)
from src.features import features_horarias
from src.labels import rotular

BASICAS = ["duty_cycle", "transicoes", "tp3_media", "tp3_std", "corrente_media", "oleo_media"]


def montar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Atributos causais (só olham o passado) e rótulo, uma linha por hora."""
    f = features_horarias(df)
    X = f[BASICAS].copy()
    for c in BASICAS:
        X[f"{c}_lag3h"] = f[c].shift(3)
    X["oleo_tendencia_6h"] = f["oleo_media"] - f["oleo_media"].shift(6)
    return X, rotular(f)


def modelos() -> dict:
    return {
        "regressao_logistica": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=3, learning_rate=0.05,
            max_iter=200, random_state=0,
        ),
    }


def janelas_teste() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Uma janela de teste por falha, mais o cenário de controle sem falha."""
    janelas = []
    for ini, fim in FALHAS:
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        a = ini - pd.Timedelta(days=JANELA_ANTES_DIAS)
        b = fim + pd.Timedelta(days=JANELA_DEPOIS_DIAS)
        janelas.append((f"falha_{ini:%d_%b}", a, b))
    a, b = CENARIOS["normal_controle"]
    janelas.append(("controle", pd.Timestamp(a), pd.Timestamp(b)))
    return janelas


def divisoes(index: pd.DatetimeIndex):
    """Leave-one-failure-out: testa numa janela e treina com as demais horas,
    deixando uma margem de HORIZONTE_H horas para o rótulo não vazar."""
    margem = pd.Timedelta(hours=HORIZONTE_H)
    for nome, a, b in janelas_teste():
        teste = (index >= a) & (index <= b)
        treino = ~((index >= a - margem) & (index <= b + margem))
        yield nome, teste, treino


def validar(df: pd.DataFrame) -> dict:
    """Retorna as probabilidades fora da amostra (OOF) de cada modelo e da regra base."""
    X, y = montar_dataset(df)
    m = y.notna()
    X, y = X[m], y[m].astype(int)

    oof = {nome: pd.Series(np.nan, index=y.index) for nome in modelos()}
    for nome_janela, teste, treino in divisoes(y.index):
        if teste.sum() == 0:
            continue
        for nome, modelo in modelos().items():
            modelo.fit(X[treino], y[treino])
            oof[nome][teste] = modelo.predict_proba(X[teste])[:, 1]

    # regra base: a própria fração de tempo sob carga (sem aprendizado)
    oof["regra_base_duty_cycle"] = X["duty_cycle"].fillna(0)
    return {"y": y, "oof": pd.DataFrame(oof)}


def metricas(y: pd.Series, s: pd.Series) -> dict:
    ok = s.notna()
    return {
        "PR-AUC": average_precision_score(y[ok], s[ok]),
        "ROC-AUC": roc_auc_score(y[ok], s[ok]),
        "positivos": int(y[ok].sum()),
        "horas": int(ok.sum()),
    }


def treinar_final(df: pd.DataFrame, nome: str) -> object:
    """Treina com todas as horas rotuladas e salva em models/."""
    X, y = montar_dataset(df)
    m = y.notna()
    modelo = modelos()[nome]
    modelo.fit(X[m], y[m].astype(int))
    MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODELOS / f"{nome}.joblib")
    return modelo
