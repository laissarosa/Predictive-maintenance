import time
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import demo_logic as L
from demo_plot import desenhar

st.set_page_config(page_title="Monitor de compressor - MetroPT3", layout="wide")

# ------------------------------------------------------------------ dados
caminho = L.achar("ae_escores.parquet")
if caminho is None:
    st.error("Arquivo `ae_escores.parquet` não encontrado. Copie-o para a pasta `dados_demo/` "
             "(ele fica em `reports/resultados/` no projeto principal). Veja o LEIA-ME.")
    st.stop()


@st.cache_data(show_spinner="Carregando escores...")
def carregar(p):
    return L.carregar_horario(p)


h, val = carregar(str(caminho))

# ------------------------------------------------------------------ barra lateral
st.sidebar.header("Cenário")
cenarios = list(L.FALHAS) + [L.CONTROLE[0]]
nome = st.sidebar.selectbox("Escolha o período", cenarios)
if nome == L.CONTROLE[0]:
    ini_c, fim_c = pd.Timestamp(L.CONTROLE[1]), pd.Timestamp(L.CONTROLE[2])
    falha = None
else:
    f_ini, f_fim = (pd.Timestamp(x) for x in L.FALHAS[nome])
    ini_c, fim_c = f_ini - pd.Timedelta(days=3), f_fim + pd.Timedelta(days=1)
    falha = (f_ini, f_fim)

st.sidebar.header("Sensibilidade")
q = st.sidebar.select_slider("Quantil do limiar (dias normais)",
                             options=[0.95, 0.97, 0.98, 0.99, 0.995], value=L.QUANTIL_PADRAO)
if q != L.QUANTIL_PADRAO:
    st.sidebar.caption("Valores diferentes de 0,99 foram avaliados depois de ver as falhas: "
                       "mostram o trade-off, não um desempenho validado.")
mostrar_falha = st.sidebar.checkbox("Mostrar janela de falha rotulada", value=True)
velocidade = st.sidebar.select_slider("Velocidade da reprodução", options=["lenta", "média", "rápida"],
                                      value="média")
pausa = {"lenta": 0.5, "média": 0.2, "rápida": 0.05}[velocidade]

lim = L.limiar(h, val, q)
acima_t, alerta_t = L.alertas(h["sistema"], lim)
idx = pd.date_range(ini_c.floor("h"), fim_c.ceil("h"), freq="1h")
hs = h.reindex(idx)
acima = acima_t.reindex(idx, fill_value=False)
alerta = alerta_t.reindex(idx, fill_value=False)

# ------------------------------------------------------------------ cabeçalho
st.title("Monitor de condição do compressor de ar (MetroPT3)")
st.caption("Prova de conceito - SIN 323. Autoencoder convolucional treinado só em operação normal; "
           "o alerta indica desvio do comportamento normal. É uma recomendação para a equipe de "
           "manutenção, não um comando de desligamento.")

aba_demo, aba_val = st.tabs(["Replay do cenário", "Validação e limitações"])


# ------------------------------------------------------------------ painel
def painel(t):
    estado = L.estado_em(hs, acima, alerta, t)
    horas = L.horas_em_alerta(alerta, t)
    tops = L.principais(hs, t)
    texto = L.texto_motivo(estado, tops, horas)

    st.markdown(f"**Instante:** {t:%d/%m/%Y %H:%M}")
    if estado == "alerta":
        st.error(f"ALERTA - {texto}")
    elif estado == "atencao":
        st.warning(texto)
    elif estado == "sem_dado":
        st.info(texto)
    else:
        st.success(texto)

    c1, c2, c3 = st.columns(3)
    s = hs.loc[t, "sistema"]
    c1.metric("Escore do sistema", "-" if pd.isna(s) else f"{s:.1f}x")
    c2.metric("Limiar de alerta", f"{lim:.1f}x")
    c3.metric("Horas em alerta", horas)

    fig = desenhar(hs, alerta, lim, t, falha, mostrar_falha)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ aba demo
with aba_demo:
    st.subheader(nome)
    if falha:
        inicio_padrao = (falha[0] - pd.Timedelta(hours=24)).floor("h")
    else:
        inicio_padrao = idx[0]
    inicio_padrao = min(max(inicio_padrao, idx[0]), idx[-1])
    t_sel = st.slider("Instante do replay (o sistema só vê o passado)",
                      min_value=idx[0].to_pydatetime(), max_value=idx[-1].to_pydatetime(),
                      value=inicio_padrao.to_pydatetime(), step=timedelta(hours=1),
                      format="DD/MM HH:mm", key=f"t_{nome}")
    tocar = st.button("▶ Reproduzir a partir do instante escolhido")
    area = st.empty()
    if tocar:
        for t in idx[idx >= pd.Timestamp(t_sel)]:
            with area.container():
                painel(t)
            time.sleep(pausa)
    else:
        with area.container():
            painel(pd.Timestamp(t_sel))

# ------------------------------------------------------------------ aba validação
with aba_val:
    st.subheader("Resultados medidos")
    p_av = L.achar("ae_avaliacao.csv")
    if p_av:
        st.markdown("**Alarme com limiar fixado antes de ver as falhas** (quantil 0,99, 2 h seguidas). "
                    "Antecedência negativa = o alerta veio depois do início da falha.")
        st.dataframe(pd.read_csv(p_av), use_container_width=True, hide_index=True)
    p_cv = L.achar("ae_curva_operacao.csv")
    if p_cv:
        st.markdown("**Curva de operação** (avaliada depois de ver as falhas: mostra o trade-off "
                    "entre falhas detectadas e falsos alarmes).")
        st.dataframe(pd.read_csv(p_cv), use_container_width=True, hide_index=True)
    if not (p_av or p_cv):
        st.info("Copie `ae_avaliacao.csv` e `ae_curva_operacao.csv` para `dados_demo/` para ver as tabelas.")

    st.subheader("Limitações")
    st.markdown("""
- **Quatro falhas rotuladas.** Qualquer métrica tem grande incerteza; trate os números como provisórios.
- **Antecipação em 1 de 4 falhas.** Nas demais o alerta veio de 0,5 a 2 h depois do início: é detecção rápida, não previsão.
- **Pode estar marcando "carga contínua".** Os episódios de carga contínua foram retirados do treino, então o modelo também dispara em episódios sem rótulo; os dados não permitem separar "rótulos incompletos" de "falso alarme".
- **Taxa de falsos alarmes otimista.** Parte das semanas normais entrou no treino.
- **Decisões tomadas depois de ver resultados:** a escala por sensor (quantil 0,95 do erro normal) e qualquer sensibilidade diferente de 0,99.
- **Um compressor, um trem.** Os resultados podem não se transferir a outros equipamentos.
- **Supervisão humana.** O sistema recomenda inspeção; quem decide a manutenção é o técnico. Falsos alarmes custam inspeção desnecessária; falhas perdidas custam trem fora de serviço.
- **Explicação do alerta:** componentes ordenados pelo erro de reconstrução. É uma pista de onde olhar, não um diagnóstico.
""")
