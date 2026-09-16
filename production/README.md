# Ambiente PRODUCTION (codice vero)

Qui lavori sul **codice definitivo**. Nessun dato inventato: ogni connettore va completato con fonti reali.

## Struttura

```
production/
├── config.py                 ← path file, credenziali (env vars)
├── connectors/
│   ├── valido.py             ← TODO: query DB vendite
│   ├── pdv_master.py         ← legge data/manual/pdv_master.csv
│   ├── weather_api.py        ← TODO: API meteo
│   └── calendar.py           ← legge calendario da CSV team
├── features/
│   ├── store_features.py     ← bande mq / regime meteo
│   └── build_features.py     ← feature table rolling
├── model/
│   ├── dual_head_tf.py       ← architettura TensorFlow
│   └── train.py              ← training + salvataggio artefatti
├── operational/
│   └── replenishment.py      ← qty suggerita + review
└── pipeline.py               ← entry point end-to-end
```

## Ordine di lavoro consigliato

1. `connectors/valido.py` — accesso read-only Valido (con IT)
2. `data/manual/pdv_master.csv` — compilare tutti i PDV
3. `connectors/weather_api.py` — Open-Meteo (già collegato, nessuna API key)
4. `model/train.py` — primo training rolling su dati reali
5. `pipeline.py` — inferenza periodica + export per CDM

## Comando (quando pronto)

```bash
cd ~/Desktop/modello-forecast
.venv/bin/python -m production.pipeline
```

Oggi `pipeline.py` si ferma su Valido finché le vendite reali non sono collegate. Il meteo Open-Meteo è pronto:

```bash
.venv2/bin/python -m production.connectors.weather_api
```

## Regola d'oro

**Non importare da `testing/`** — production non deve mai usare dati fake.
