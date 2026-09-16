# modello-forecast

Prototipo **domanda attesa / riassortimento** — TensorFlow/Keras.  
Vive in sandbox agente: `~/Desktop/modello-forecast`

## Due ambienti separati

| Cartella | Scopo | Dati |
|----------|--------|------|
| **`testing/`** | Sperimenti, training, imparare | **Fake** (generati, volume alto) |
| **`production/`** | Codice su cui lavori davvero | **Valido + CSV + API** (da collegare) |

```
modello-forecast/
├── testing/           ← dati fake + train TensorFlow
├── production/        ← connettori reali (scheletro)
├── data/manual/       ← CSV PDV/calendario (file team)
├── docs/              ← contratti dati riunione
└── src/               ← vecchio prototipo PyTorch (legacy)
```

## Quick start — TESTING

```bash
cd ~/Desktop/modello-forecast

# Pipeline dati (funziona sempre, no TensorFlow)
.venv/bin/python -m testing.run_validate

# Dati + baseline naive + operativo (no TensorFlow)
.venv/bin/python -m testing.run_train_lite

# Training TensorFlow (usa SEMPRE .venv/bin/python)
OMP_NUM_THREADS=1 .venv/bin/python -m testing.run_train

# Report visivo per Produzione (generato da run_train, o rigenerabile)
open reports/report_produzione.html
```

Genera ~340k righe fake (20 PDV, 42 SKU), sample rolling ~42k righe.

## Mac: TensorFlow non parte?

Su macOS recenti, **Python 3.9 da Command Line Tools** nel venv spesso non esegue TensorFlow (mutex lock / hang all'import). `run_validate` e `run_train_lite` restano OK.

| Opzione | Comando |
|---------|---------|
| **Docker** (consigliato) | `./scripts/run_train_docker.sh` |
| **Venv Python 3.11+** | `./scripts/setup_venv_mac.sh` poi `run_train` |
| **Senza TF** | `.venv/bin/python -m testing.run_train_lite` |

Dettagli: `testing/README.md`

## Quick start — PRODUCTION (scheletro)

```bash
.venv/bin/python -m production.pipeline
```

Carica CSV PDV/calendario; si ferma finché non implementi Valido e meteo.

## Stack

- Python 3.11+ consigliato per TensorFlow su Mac (3.9 CLT spesso insufficiente)
- TensorFlow / Keras (`tensorflow>=2.15,<2.17` in `requirements.txt`)
- pandas, numpy (testing usa **CSV**, non parquet)
- TensorFlow solo in `run_train` — vedi `requirements-testing.txt` per validazione dati

## Documentazione

- `docs/DATA_CONTRACTS.md` — campi Valido / riunione 11 ago
- `testing/README.md` — ambiente fake
- `production/README.md` — ordine di implementazione

## Legacy

La cartella `src/` contiene il primo prototipo **PyTorch** (3 PDV, pochi SKU).  
Usa `testing/` + `production/` per il lavoro nuovo su **TensorFlow**.
