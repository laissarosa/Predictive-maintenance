from pathlib import Path

# ---------- caminhos ----------
RAIZ = Path(__file__).resolve().parent
CSV = RAIZ / "data" / "MetroPT3(AirCompressor).csv"
PARQUET = RAIZ / "data" / "processed" / "metropt3.parquet"
MODELOS = RAIZ / "models"
FIGURAS = RAIZ / "reports" / "figuras"
RESULTADOS = RAIZ / "reports" / "resultados"

# ---------- falhas reais (relatórios da empresa) ----------
FALHAS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),   # confirmado no gráfico: termina ~06:00
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]

# manutenções (a de maio é interpretação: a UCI escreve "30Apr")
MANUTENCOES = ["2020-05-30 12:00", "2020-06-08 16:00", "2020-07-16 00:00"]

# ---------- cenários da demo ----------
CENARIOS = {
    "falha_29_maio": ("2020-05-27", "2020-05-31"),
    "falha_05_junho": ("2020-06-03", "2020-06-09"),
    "normal_controle": ("2020-03-10", "2020-03-14"),
}

# ---------- parâmetros ----------
SEG_POR_LINHA = 10            # o CSV tem uma linha a cada ~10 s
LACUNA_MAX = "1min"           # acima disso, considera lacuna nos dados
MIN_COBERTURA = 0.8           # janela com menos de 80% das amostras vira NaN

# Definidos pelo histograma de Motor_current (figura 01): modos em ~0,04 A (desligado),
# ~3,8 A (ocioso) e ~5,7 A (carga). Vale entre ocioso e carga em ~4,0-4,5 A.
LIMIAR_SEM_CARGA = 1.0
LIMIAR_CARGA = 4.5

# horizonte de previsão: a IA prevê se haverá falha nas próximas HORIZONTE_H horas
HORIZONTE_H = 24

# validação por falha: janela de teste em torno de cada falha (dias antes / depois)
JANELA_ANTES_DIAS = 7
JANELA_DEPOIS_DIAS = 2

# ciclos de carga (abordagem de detecção de anomalia)
CICLO_MIN_AMOSTRAS = 3        # período com menos amostras que isso é ruído e não vira ciclo
CICLO_GAP_MAX_S = 60          # intervalo máximo entre amostras; acima disso o período tem lacuna
ALERTA_PERCENTIL = 99         # alarme = escore acima do percentil 99 dos ciclos normais

# níveis de risco sobre a probabilidade prevista (PROVISÓRIOS: ajustar com a curva PR)
RISCO_MEDIO = 0.2
RISCO_ALTO = 0.5
# episódios de carga contínua (compressor sem conseguir descarregar: assinatura de vazamento)
EPIS_MIN_H = 4                # duração mínima para virar episódio
EPIS_INTERRUPCAO_MIN = 10     # pausas sem carga menores que isso não encerram o episódio
EPIS_COBERTURA_MIN = 0.8      # abaixo disso o episódio tem muita lacuna nos dados
EPIS_FRAC_CARGA_CONT = 0.9    # fração sob carga a partir da qual o episódio é "contínuo" (senão, ciclagem intensa)

# limpeza do dataset
PARQUET_LIMPO = RAIZ / "data" / "processed" / "metropt3_limpo.parquet"
CONGELADO_MIN_AMOSTRAS = 6    # sensores analógicos idênticos por >= 6 linhas (1 min) = registrador travado
INTERPOLAR_MAX_S = 60         # lacunas até 60 s são preenchidas (e marcadas); acima disso ficam NaN

# ---------- componentes, criticidade e alertas ----------
# Criticidade 1-5 (5 = mais crítico). Síntese própria a partir do artigo MetroPT (Veloso et al.):
# a APU não tem redundância; o vazamento de óleo foi a falha mais grave (danificou o motor e
# retirou o trem), vazamento de ar para os clientes (freio, suspensão) também retirou o trem,
# vazamento no secador deixou o trem seguir até a oficina. NÃO é uma escala oficial: revisar.
COMPONENTES = {
    "oleo":      {"sensores": ["Oil_temperature", "Oil_level"], "severidade": 5},
    "pressao":   {"sensores": ["TP2", "TP3", "Reservoirs", "H1", "LPS"], "severidade": 5},
    "motor":     {"sensores": ["Motor_current"], "severidade": 4},
    "secador":   {"sensores": ["DV_pressure", "Towers", "Pressure_switch"], "severidade": 3},
    "valvulas":  {"sensores": ["COMP", "DV_eletric", "MPG"], "severidade": 3},
    "vazao":     {"sensores": ["Caudal_impulses"], "severidade": 2},
}

# Três níveis: informacao, aviso, critico. Escore = anomalia robusta (z) do componente.
# Persistência exigida (horas seguidas) diminui com a criticidade: ceil(BASE / severidade).
ALERTA_Z_INFO, ALERTA_Z_AVISO, ALERTA_Z_CRITICO = 3.0, 4.0, 5.0
ALERTA_BASE_AVISO_H = 12
ALERTA_BASE_CRITICO_H = 6
ALERTA_INFO_H = 1
ALERTA_COMPONENTES_PARA_CRITICO = 2   # dois componentes em aviso ao mesmo tempo = crítico

# escores de anomalia por componente (z robusto contra a mediana/MAD dos últimos dias)
BASE_DIAS = 14                # dias de referência (mesma hora do dia)
BASE_ATRASO_H = 24            # a referência termina 24 h antes: uma deriva lenta não é absorvida na hora
BASE_MIN_DIAS = 5             # mínimo de dias válidos na referência
ESCALA_REF_DIAS = 28          # início da série, usado só para o piso da escala (evita z explosivo)
ESCORE_MAX = 30.0
HORA_MIN_COBERTURA = 0.8
HORA_MIN_AMOSTRAS_CARGA = 18  # atributos condicionados à carga exigem >= 3 min sob carga na hora

# calibração dos limiares por componente com o início da série (antes de qualquer falha rotulada);
# a avaliação de falsos alarmes usa só o que vem depois
CALIBRACAO_FIM = "2020-04-10"
CALIBRACAO_QUANTIS = (0.95, 0.99, 0.998)   # informação, aviso, crítico

# avaliação por intervalos (protocolo da Fig. 9 de Veloso et al., arXiv 2207.05466)
INTERVALO_FUNDIR_H = 3        # alertas separados por <= 3 h viram um só intervalo previsto
ANTECEDENCIA_MIN_H = 2        # requisito da empresa: detectar >= 2 h antes de o trem parar

# nível "informação" do SISTEMA: >= N componentes em informação por >= H horas seguidas.
# (1, 1) restaura o comportamento antigo (qualquer componente, 1 h): 7,85 falsos intervalos/semana.
SISTEMA_INFO_COMPONENTES = 2
SISTEMA_INFO_H = 2
