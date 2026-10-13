# Demo - Monitor de condição do compressor (MetroPT3)

Replay de cenários com o autoencoder: escore do sistema, limiar, alerta e motivo por componente.
O app lê só resultados já gerados; não treina nada e não precisa do CSV de 1,5 milhão de linhas.

## Arquivos necessários (copie para `dados_demo/`)

Vêm da pasta `reports/resultados/` do projeto principal:

| Arquivo | Obrigatório |
|---|---|
| `ae_escores.parquet` | sim (gerado por `scripts/run_autoencoder.py`) |
| `ae_avaliacao.csv` | não (tabela da aba "Validação") |
| `ae_curva_operacao.csv` | não (gerado por `scripts/curva_operacao_ae.py`) |

Atalho: se você descompactar esta pasta dentro do projeto principal, o app também procura em
`reports/resultados/` automaticamente.

## Como executar

```bash
conda activate sin323
pip install -r requirements.txt
streamlit run app.py
```

Abre em http://localhost:8501.

## Uso

1. Escolha o cenário (4 falhas ou um período normal) na barra lateral.
2. Arraste o instante do replay ou clique em "Reproduzir". O sistema só enxerga o passado.
3. Leia o motivo do alerta e o mapa de calor por componente.
4. A aba "Validação e limitações" traz as tabelas medidas e as limitações.

Dica: se der erro de gravação/leitura, copie a pasta para fora do OneDrive.
