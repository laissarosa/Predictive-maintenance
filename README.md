# Monitor de condição do compressor de ar de um metrô (MetroPT3)

Prova de Conceito (PoC) - SIN 323 Inteligência Artificial (2026/2), Projeto Prático.

## Resumo em quatro linhas

- **Problema:** a equipe de manutenção de um metrô só descobre que o compressor de ar de um trem está falhando quando ele quebra, e o trem sai de serviço.
- **Solução:** um autoencoder convolucional 1D, treinado apenas com operação normal, mede o quanto o comportamento multissensor se afasta do normal, aponta quais componentes mais se afastaram e dispara um alerta.
- **Resultado medido:** o alerta detecta as 4 falhas conhecidas com cerca de 0,2 falsos alarmes por semana, mas **só em 1 das 4 o aviso veio antes do início da falha** (6,5 h). Nas outras três veio de 0,5 a 2 h depois.
- **Conclusão honesta:** o sistema é um **detector precoce de estado anômalo**, e não um previsor com horas de antecedência. Isso é um resultado parcialmente positivo e todos os números são provisórios (apenas 4 eventos).

---

## 1. Problema real

**Quem tem o problema:** a equipe de manutenção do operador de metrô (dados do Metro do Porto, compressor de ar de um trem).

**O que fazem hoje:** manutenção em intervalo fixo e reparo reativo depois da falha.

**Por que é relevante:** o compressor (unidade de produção de ar, APU) alimenta freios e portas e não tem redundância. Quando falha, o trem é retirado de serviço, o que gera parada não planejada e custo.

**O que se pretende melhorar:** dar à equipe um aviso precoce, com taxa de falsos alarmes conhecida, que mostre **qual componente** se comporta de forma anormal, para priorizar a inspeção.

---

## 2. IA como parte essencial da solução

**Tarefa da IA:** aprender, sem rótulos, como é o comportamento normal do compressor e medir o desvio em tempo (quase) real.

| | |
|---|---|
| **Entrada** | Janelas de 60 min dos 15 sensores (pressões TP2/TP3/reservatórios/H1, temperatura e nível do óleo, corrente do motor, pressão da válvula de dreno, vazão e sinais digitais de válvulas e interruptores), reamostradas a 1 min a partir do dataset limpo |
| **Modelo** | Autoencoder convolucional 1D (PyTorch): comprime cada janela em 120 números e a reconstrói |
| **Treino** | Só janelas normais (fora de ±7 dias das falhas e dos episódios de carga contínua). Sem uso de rótulos de falha |
| **Saída** | Erro de reconstrução por sensor, agrupado em 6 componentes (óleo, pressão, motor, secador, válvulas, vazão) e um escore do sistema (o pior componente), relativo ao erro típico (1 = normal) |
| **Alerta** | Escore horário acima de um limiar (quantil 0,99 de dias normais separados) por 2 horas seguidas |

**Por que IA e não uma regra fixa:** o "normal" depende de 15 sensores ao mesmo tempo e muda com a demanda do trem. Escrever limiares à mão para cada combinação não escala. Só há 4 falhas rotuladas, o que inviabiliza um classificador supervisionado (ver seção 6). Um modelo que aprende só o normal é a formulação que cabe nos dados disponíveis.

**Como participa da solução:** o escore e o erro por componente alimentam diretamente o alerta e a explicação ("óleo 9,5x o erro típico") mostrados no demo.

---

## 3. Produto funcional (demo)

O demo é um app Streamlit (pasta `demo_metropt3/`) com replay de cenários:

- seleção de uma das 4 falhas ou de um período normal de controle;
- replay causal: o sistema só enxerga o passado;
- linha do tempo do escore com o limiar e os alertas;
- mapa de calor do desvio por componente;
- texto com o motivo do alerta e uma sugestão de onde inspecionar (hipótese, não diagnóstico);
- aba "Validação e limitações" com as tabelas medidas.

```bash
cd demo_metropt3
python -m pip install -r requirements.txt
# copie reports/resultados/ae_escores.parquet (e os .csv) para demo_metropt3/dados_demo/
python -m streamlit run app.py
```

Roteiro sugerido de apresentação: **15/07** (único com alerta antecipado, 6,5 h), **18/04** (alerta depois do início; só 21% de dado válido nas 24 h anteriores) e **controle de março** (sem alerta).

---

## 4. Validação

**Protocolo.** As falhas e os episódios de carga contínua ficam fora do treino. O limiar vem de dias normais separados do treino (dias inteiros, sem sobreposição). O alerta usa só o passado. A avaliação segue a ideia da Fig. 9 de Veloso et al. (arXiv 2207.05466): um intervalo de alerta que sobrepõe uma falha observada é acerto; um que não sobrepõe nada conhecido é falso positivo. Falsos alarmes são medidos em semanas normais longe de falhas e episódios (±24 h).

### 4.1 Resultado final (autoencoder, limiar fixado antes: quantil 0,99 = 4,68x, persistência de 2 h)

| Falha | Dado válido nas 24 h antes | Detectada | Antecedência | Componentes principais | AUC (falha vs normal) |
|---|---|---|---|---|---|
| 18/04/2020 | 21% | sim | -2,0 h (depois) | pressão, óleo, secador | 0,997 |
| 29-30/05/2020 | 79% | sim | -0,5 h (depois) | óleo, pressão, secador | 0,997 |
| 05-07/06/2020 | 100% | sim | -1,0 h (depois) | óleo, pressão, secador | 0,998 |
| 15/07/2020 | 71% | sim | **+6,5 h (antes)** | pressão, válvulas, secador | 0,999 |

Falsos alarmes: 4 intervalos em cerca de 18 semanas normais (~0,2 por semana). Esta taxa é **otimista**, pois parte dessas semanas entrou no treino.

### 4.2 Curva de operação (avaliada depois de ver as falhas: mostra o trade-off, não um desempenho validado)

| Quantil do limiar | Limiar | Falhas detectadas | Com ≥ 2 h antes | Falsos por semana |
|---|---|---|---|---|
| 0,900 | 0,99 | 4/4 | 1 | 2,18 |
| 0,950 | 2,14 | 4/4 | 1 | 1,49 |
| 0,970 | 3,66 | 4/4 | 1 | 0,64 |
| 0,980 | 4,27 | 4/4 | 1 | 0,35 |
| **0,990** | **4,68** | **4/4** | **1** | **0,20** |
| 0,995 | 5,90 | 4/4 | 1 | 0,15 |

### 4.3 Comparação com as outras abordagens tentadas

| Abordagem | Tipo | Resultado |
|---|---|---|
| Regra do ciclo de trabalho (24 h) | sem aprendizado | PR-AUC 0,126 (referência) |
| Regressão logística, 24 h | supervisionado | PR-AUC 0,081; ROC-AUC 0,37. **Negativo** |
| Gradient boosting, 24 h | supervisionado | PR-AUC 0,072; ROC-AUC 0,29. **Negativo** |
| Isolation Forest por ciclo de carga | não supervisionado | Nenhum alarme nas 24 h antes de nenhuma falha; PR-AUC 0,307 contra 0,423 da regra; 1,25 alarmes/dia no controle normal. **Negativo** |
| Alertas por componente (z-score robusto) | estatística robusta | 4/4 falhas no nível aviso, 0,07 falsos intervalos/semana, aviso 3-8 h **depois** do início; parâmetros ajustados no período de teste |
| **Autoencoder 1D** | **rede neural, não supervisionado** | **4/4 falhas, 1/4 com antecedência ≥ 2 h, ~0,2 falsos/semana** |

Atenção: as abordagens não foram avaliadas exatamente no mesmo período e protocolo. A comparação é indicativa.

---

## 5. Análise crítica

**Limitações dos dados**
- **Apenas 4 eventos de falha.** Toda métrica tem grande incerteza. Não há intervalo de confiança confiável.
- **Cobertura ruim em alguns eventos:** só 21% de dado válido nas 24 h antes de 18/04 e 71% antes de 15/07. Cerca de 38 dias do período estão ausentes e 2,8% das linhas são de um registrador congelado.
- **Rótulos provavelmente incompletos:** três episódios sem rótulo (12/04, 13/05, 19-20/05) têm a mesma assinatura das falhas. Sem registros de manutenção, não dá para confirmar.
- **Um compressor, um trem, 2020.** O período coincide com a pandemia, que mudou a demanda do metrô. Os resultados podem não se transferir.

**Limitações do resultado**
- **Não é previsão.** Só 1 de 4 falhas foi avisada antes do início.
- **Pode estar detectando "carga contínua", e não "falha".** Os episódios de carga contínua foram retirados do treino, então o modelo os vê como anômalos. 7 dos 16 intervalos de alerta caíram em episódios sem rótulo. Os dados não permitem separar "rótulos incompletos" de "o modelo marca qualquer episódio longo".
- **Falsos alarmes otimistas** (semanas normais usadas no treino).
- **Decisões tomadas depois de ver resultados:** ver seção 6.

**Riscos e uso responsável**
- **Supervisão humana:** o sistema recomenda inspeção, não ordena desligamento. A decisão é do técnico.
- **Custos assimétricos:** falso alarme custa inspeção desnecessária; falha perdida custa trem fora de serviço. O limiar deve refletir esse trade-off e ser discutido com a equipe de manutenção.
- **Confiança excessiva:** um painel limpo não garante saúde do compressor. O sistema só enxerga o que os 15 sensores mostram.
- **Explicação limitada:** a lista de componentes é uma pista de onde olhar, não um diagnóstico.
- **Privacidade:** dados anonimizados de máquina, sem dados pessoais; risco baixo. O uso deve seguir os termos do dataset UCI.
- **Segurança:** em produção, o fluxo de sensores precisaria de integridade e autenticação, pois dados adulterados poderiam esconder ou criar alertas.
- **Vieses:** o modelo aprende o "normal" de um único trem em fevereiro-setembro de 2020, e qualquer mudança de rota, demanda ou manutenção pode ser lida como anomalia.

---

## 6. O que deu errado (para o grupo)

Este registro existe para que todos saibam explicar o que foi tentado e por quê.

1. **A previsão supervisionada de 24 h falhou.** Com 4 eventos, nenhum modelo aprende de forma confiável, e a regra simples de ciclo de trabalho ganhou dos dois modelos. A formulação "falha nas próximas 24 h" também não casa com os dados: os eventos aparecem como início abrupto de carga contínua.
2. **O Isolation Forest por ciclo não funcionou como alarme.** Ele é cego em carga contínua (18/04 e 05/06 quase não geram ciclos) e dispara 1,25 vezes por dia em período normal com limiar de percentil 99 por ciclo. Mostrou uma deriva pré-falha em 15/07, que depois o autoencoder confirmou de forma independente. Também tinha um bug na máscara de avaliação (ciclos dentro de uma falha voltavam como negativos pela janela da falha seguinte), corrigido.
3. **O primeiro autoencoder deu 1 de 4 por um defeito de projeto.** O erro de cada sensor era dividido pela mediana do seu erro; sensores quase constantes tinham mediana perto de zero e dominavam o escore, o que levou o limiar a 52x. Corrigimos usando o quantil 0,95 do erro normal e treinando 120 épocas em vez de 40. **Essa correção foi feita depois de ver o resultado.**
4. **Uma rodada ocorreu sem `episodios.parquet`** (arquivo perdido), então o treino incluiu episódios de carga contínua. Os números dessa rodada foram descartados e o experimento foi refeito com os episódios.
5. **Parâmetros do z-score ajustados no teste.** `SISTEMA_INFO_*` e outros foram escolhidos olhando o período avaliado (o comentário em `config.py` registra 7,85 falsos/semana contra 0,46). Os números do z-score deixam de ser "fora da amostra".
6. **Circularidade nos episódios "prováveis".** A semelhança com as falhas rotuladas vem do próprio detector, e depois eles são contados como "detectados 3/3". Devem ser reportados separadamente, como "não confirmados".
7. **Problemas de ambiente (OneDrive).** O projeto em `OneDrive\Desktop` corrompeu um parquet (`Couldn't deserialize thrift`), bloqueou gravação de figuras (`Errno 22`) e fez um arquivo de resultados sumir. **Rode o projeto fora do OneDrive** (por exemplo, `C:\projetos\`) ou pause a sincronização.
8. **Terminal do Windows:** `python -c` de várias linhas não funciona no `cmd`. Use sempre arquivos `.py`.

### Decisões posteriores aos resultados (declarar no pôster)

| Decisão | Quando | Efeito |
|---|---|---|
| Escala por sensor pelo quantil 0,95 (antes: mediana) | depois da 1ª rodada do autoencoder | corrigiu defeito de projeto; subiu a detecção de 1/4 para 4/4 |
| 120 épocas (antes: 40) | depois da 1ª rodada | perda de validação ainda caía |
| Curva de operação com outros quantis | depois de ver as falhas | mostra trade-off, não desempenho validado |
| Parâmetros do z-score (`SISTEMA_INFO_*`, `ALERTA_BASE_*`) | durante o desenvolvimento, olhando o teste | números não são fora da amostra |

---

## 7. Próximos passos

- **Obter registros de manutenção** para confirmar os episódios sem rótulo. Isso decide se o modelo erra ou se os rótulos estão incompletos.
- **Treinar apenas com dados anteriores a cada falha** (validação temporal estrita), para eliminar o uso de informação futura no treino do normal.
- **Atributos em nível de ciclo** (taxa de queda de pressão em repouso, tempo de recarga) como entrada adicional.
- **Regra de alerta com tendência** (inclinação do escore) em vez de limiar fixo, validada sem ajustar no teste.
- **Mais equipamentos/períodos** para testar generalização.

---

## 8. Como reproduzir

```bash
conda create --name sin323 python=3.12.14
conda activate sin323
pip install -r requirements.txt
pip install torch streamlit
```

Coloque `MetroPT3(AirCompressor).csv` em `data/` e rode, na raiz:

```bash
python scripts/prepare_data.py       # CSV -> Parquet
python scripts/clean_data.py         # grade regular de 10 s, flags de qualidade
python scripts/episodes.py           # episódios de carga contínua
python -u scripts/run_autoencoder.py # treino, alerta e avaliação por intervalos
python scripts/auc_autoencoder.py    # separabilidade por falha (AUC)
python scripts/curva_operacao_ae.py  # trade-off falhas detectadas x falsos alarmes
```

Comparações (opcionais): `run_alerts.py`, `evaluate_intervals.py` (z-score por componente); `train.py`, `evaluate.py` (supervisionado); `build_cycles.py` e `anomaly_eval.py` (Isolation Forest).

## 9. Estrutura do repositório

```
config.py                    caminhos, falhas, limiares, parâmetros
src/autoencoder.py           janelas, autoencoder 1D, treino, escores por componente
src/limpeza.py, data.py      limpeza (registrador congelado, lacunas), carga de dados
src/episodios.py             episódios de carga contínua
src/escores.py, alertas.py   z-score por componente e níveis de alerta
src/ciclos.py, anomalia.py   atributos por ciclo e Isolation Forest
src/features.py, labels.py, model.py, risk.py, simulador.py, report.py   abordagem supervisionada
scripts/                     pipeline e avaliações
demo_metropt3/               demo Streamlit (app.py, demo_logic.py, demo_plot.py, dados_demo/)
reports/figuras/, reports/resultados/   saídas geradas
models/                      modelos salvos
```

## Referências

- Dataset MetroPT3 (Air Compressor), UCI Machine Learning Repository.
- Veloso et al., *The MetroPT dataset for predictive maintenance*, arXiv:2207.05466 (protocolo de avaliação por intervalos e descrição das falhas).
