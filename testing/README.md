# Testing environment

Dati fake in **CSV** (no PyArrow obbligatorio — evita crash TF+parquet su Mac).

## Ciclo CDM (ogni 15 giorni)

Il training si fa **una volta** (o quando cambia il catalogo / i dati). Ogni quindicina si gira **solo l’inferenza**:

```bash
cd ~/Desktop/modello-forecast

# Chiusura = ultimo giorno in fact_daily (simula «oggi»)
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_forecast

# Chiusura scelta (es. lunedì di riunione CDM)
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_forecast --as-of 2026-08-16
```

Output da portare al CDM:
- `testing/data/output/previsioni_cdm.csv` — lista PDV×SKU (domanda 15 gg, qty, vs modello CDM)
- `reports/cdm_dashboard.html` — filtri, priorità, spiegazioni

Cosa usare in riunione: **`pred_tf`** (domanda attesa) e **`qty_suggerita`**. `y_vera` c’è solo se i 15 giorni successivi sono già nel dataset (backtest), non in una chiusura live.

## Altri comandi

```bash
cd ~/Desktop/modello-forecast

# SEMPRE il venv (su Mac usa .venv2 se .venv crasha su numpy/TF)
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_validate    # dati + feature, NO TensorFlow
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train_lite  # + casi didattici operativi
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train       # genera + train TF + report HTML
.venv2/bin/python -m testing.run_report_html                   # solo report (dopo run_train)
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_explain_report  # spiegazioni + report (senza retrain)
.venv2/bin/python -m testing.run_bi_dashboard                 # solo dashboard interattiva
open reports/cdm_dashboard.html                               # mini-BI per CDM
```

### Dashboard interattiva (mini-BI)

`reports/cdm_dashboard.html` — consultabile in browser, senza Cursor:
- filtri per paese, città, categoria, **SKU**, stagione, tipo prodotto, qty, confidence
- tab **Per SKU** (previsione 15 gg in ogni negozio) / Priorità / Perché questa previsione / Accuratezza
- grafici qty per paese, negozio e **stagione (effetto meteo)**
- click su riga → dettaglio con spiegazione

Il catalogo testing è un negozio **fast fashion teenager**: crop, felpe, piumini, sandali, collane/bracciali/orecchini.
Le vendite reagiscono al meteo della città (caldo → estate, freddo → inverno, pioggia → meno gioielli).

### Spiegazioni nel report

Il report mostra **«Perché questa previsione»**: narrativa in italiano + driver (andamento vendite, cold start, promo, meteo…).
Il confronto ufficiale è vs il <strong>modello CDM originale</strong> (regole Subdued).
Il naive run-rate resta un secondo controllo.

### GPU / CPU su Mac Apple Silicon (M1–M5)

**Default sul Mac = CPU.** TensorFlow-Metal vede la GPU M5, ma su questo modello dual-head
la testa SKU resta congelata (MAE fissa ~27). Il training corretto gira su CPU.

```bash
# Training corretto (CPU, default su Mac)
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train

# GPU sperimentale (sconsigliato: SKU non impara)
FORECAST_USE_GPU=1 OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train

# Meteo reale (Open-Meteo, nessuna API key) al posto del simulatore
FORECAST_WEATHER_API=1 OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train
.venv2/bin/python -m production.connectors.weather_api   # probe: Milan ultimi giorni + ora 15:00
```

Per training GPU affidabile: Google Colab (`notebooks/colab_train.ipynb`).

Apri il report: `open reports/report_produzione.html`

## Output

```
testing/data/output/
├── pdv_master.csv          ← anagrafica negozi (anche nuovi PDV)
├── products.csv            ← SKU fast fashion teenager (nome, stagione, crop/piumini/gioielli…)
├── fact_daily.csv          ← aggregato PDV×SKU×giorno (rolling / train), con ora media
├── fact_tickets.csv        ← come Valido: 1 riga = 1 pezzo venduto con data_ora
├── calendar.csv
├── dataset_meta.csv
├── dual_head_testing.keras
├── forecast_artefacts.pkl  ← encoder + analoghi (serve a run_forecast)
├── previsioni_cdm.csv      ← pacchetto operativo quindicinale per CDM
├── focus_casi_tf.csv
├── priorita_riassortimento.csv
├── report_metrics.json
└── training_history.csv

reports/
└── report_produzione.html   ← report visivo per Produzione
```

## Troubleshooting Mac (TensorFlow / mutex lock)

**Sintomi:** `run_train` si blocca o crasha con `[mutex.cc : 452] RAW: Lock blocking` o `mutex lock failed: Invalid argument`. `run_validate` e `run_train_lite` funzionano.

**Causa tipica:** Python 3.9 da **Xcode Command Line Tools** (`/Library/Developer/CommandLineTools/...`) nel venv. Su macOS recenti (es. macOS 26, Apple Silicon) TensorFlow non è compatibile con questo interprete, anche con `tensorflow-macos`.

**Falso allarme venv (risolto):** `.venv/bin/python` è un symlink che su Mac risolve verso il Python CLT; il controllo venv usa ora `sys.prefix != sys.base_prefix` / `VIRTUAL_ENV`, non il path risolto dell'eseguibile.

### Soluzioni

1. **Docker (consigliato su Mac finché il venv locale non funziona):**
   ```bash
   ./scripts/run_train_docker.sh
   ```

2. **Venv con Python 3.11+** (python.org o Homebrew — **non** CLT `python3.9`):
   ```bash
   ./scripts/setup_venv_mac.sh
   OMP_NUM_THREADS=1 .venv/bin/python -m testing.run_train
   ```

3. **Workaround senza TensorFlow** (dati + baseline naive + operativo):
   ```bash
   .venv/bin/python -m testing.run_train_lite
   ```

**Nota:** i dati sono in CSV e TensorFlow viene importato solo dopo il load dati (`testing/tensorflow_env.py` gestisce errori di import con messaggio chiaro invece di hang).

## Regola

**Non importare da `production/`** — testing è autocontenuto.
