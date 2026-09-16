# HANDOFF — modello-forecast (prompt per altre AI)

Documento autocontenuto: logica del progetto Python + **prompt pronto da copiare** in un’altra AI.

- Repo: `/Users/mirkobergantino/Desktop/modello-forecast`
- Aggiornato: 2026-09-07
- Source of truth ML: cartella `testing/` (non `src/` legacy PyTorch, non lo stub `production/model/`)

---

## PROMPT DA DARE AD ALTRE AI (copia da qui)

```text
Sei un assistente di coding sul progetto "modello-forecast" (Subdued / CDM).

## Obiettivo
Prevedere la DOMANDA ATTESA a 15 giorni per ogni PDV×SKU e suggerire una quantità di riassortimento.
È uno strumento INTERNO di supporto al team CDM: review umana obbligatoria, NON ordini automatici a Produzione.
Baseline ufficiale da battere: regole CDM (non solo naive).

## Stack e isolation
- testing/ = pipeline completa su dati fake Valido-like (TensorFlow dual-head). È la source of truth.
- production/ = skeleton connettori reali. NON importare testing↔production.
- src/ = LEGACY PyTorch: ignoralo.
- Dati testing in CSV (no parquet: crash TF+pyarrow su Mac).
- Su Mac: sempre OMP_NUM_THREADS=1; usare .venv2 (Python 3.11+); default CPU (Metal freezes SKU head).

## Ciclo operativo
- Train: raro (quando cambia storico/catalogo) → `python -m testing.run_train`
- Inferenza CDM: ogni 15 giorni → `python -m testing.run_forecast --as-of YYYY-MM-DD`
- Train campiona as_of ogni 7 giorni (densità); ciclo meeting = 15 giorni.

## Target e feature
- y_sku = somma vendite_qty su (as_of, as_of+15]
- y_cat = somma y_sku per pdv×categoria (aux)
- Finestre storico: 7/14/28 giorni; run_rate_14d centrale
- ~57 feature numeriche (RR, trend, stock, sconto, mq, meteo, cold start, calendario, sku_weather)
- Embedding categorici: country, pdv, sku, cat, commerciale, modello, tessuto, colore, campagna, classe, sclasse, ssclasse, tipologia
- NESSUN tag stagione ESTATE/INVERNO in input rete: stagione appresa da calendario + temperatura + profilo vendite SKU (sku_weather)
- sku_prior = max(run_rate_14d × 15, 0.05)

## Architettura rete (testing/model/dual_head_tf.py) ~380k params
- Input: embeddings + numeric BatchNorm + sku_prior
- Trunk residuale Dense: 256 → 256 → 128 (dropout 0.15/0.15/0.10)
- Head SKU: y_sku = ReLU(sku_prior × softplus(scale) + delta)  # init ≈ naive
- Head CAT: stop_gradient(trunk) → Dense → softplus; target log1p(y_cat)
- Loss: Huber(y_sku, δ=5)×1.0 + Huber(y_cat, δ=1)×0.15
- Adam + CosineDecay(lr=3e-3); early stop su val_y_sku_mae (patience 12)
- UNK index 0; UNK dropout train 2%

## Post-modello
- Blend confidence/analogues per SKU giovani (testing/inference/confidence.py)
- Baseline CDM: rr14×15 × coeff_mese × cluster_perf + cold-start rules
- Qty: daily=forecast/15; need=forecast+daily×transit; raw=max(0,need−giacenza)×clip(0.45+0.55×confidence,0.45,1); qty=round(raw)
- azione = monitora se qty=0 else valuta_riassortimento

## Metriche attuali (fake data, holdout temporale cut 2026-07-22)
- TF blend MAE 3.49 | R² 0.91 | vs CDM MAE 8.15 → +57%
- LightGBM challenger MAE 3.61 (TF vince di ~0.12)
- n_train ~2.8M, n_test ~220k, epochs_run 20

## Dati Valido / csv_MIRKONE.txt (NON ancora ingestiti)
Colonne: IDAnagrafica, iso3, data, IDArticolo, tipodocumento, DescrizioneSuStampa, Qta, PrezzoPubblico, PrezzoPubblicoReale, giacenza
Regole tipodocumento:
- 100 = vendita (unica fonte DOMANDA); Qta tipicamente negativa → abs(Qta)
- 47 = carico fattura (stock)
- 20 = azzeramento magazzino (stock)
- Giacenza ricostruita con 100+47+20
Manca per train reale: catalogo prodotto (modello/tessuto/colore/…) e master PDV completo (city, mq, transit, commerciale, data_apertura).
production/connectors/valido.py = NotImplementedError.

## File chiave
- docs/DATA_CONTRACTS.md
- testing/config.py, features/build_features.py, features/schema.py, features/sku_weather.py
- testing/model/dual_head_tf.py
- testing/run_train.py, run_forecast.py, run_lgbm_challenger.py
- testing/inference/confidence.py, cdm_baseline.py
- testing/data/output/report_metrics.json
- production/connectors/valido.py, pipeline.py
- data/manual/pdv_master.csv (oggi solo 3 PDV esempio)

## Vincoli quando lavori
1. Non rompere isolation testing/production.
2. Non reintrodurre tag stagione come input modello.
3. Mantieni protocollo split TEMPORALE (no shuffle random su as_of).
4. Confronta sempre vs CDM e vs naive.
5. Su Mac: OMP_NUM_THREADS=1.
6. Non committare/pushare a meno che non te lo chieda l’utente.
7. Preferisci modificare testing/ come reference; porta in production solo quando Valido è cablato.

## Prossimi step tipici
1. Parser csv_MIRKONE → fact_daily (filtri 100 / stock 47+20)
2. Catalogo SKU + pdv_master completo
3. Retrain su dati reali + stesso protocollo metriche
4. Porting dual-head+blend+replenishment in production/

Quando rispondi: sii concreto, cita path file, non inventare metriche diverse da report_metrics.json se parli dei risultati attuali.
```

---

## 1. Contesto di business

| Voce | Dettaglio |
|------|-----------|
| Problema | Domanda attesa 15 giorni per PDV×SKU + qty riassortimento |
| Cliente interno | Team CDM (non Produzione autonoma) |
| Output | `vendite_previste_15gg`, `qty_suggerita`, `azione`, `confidence_score` |
| Governance | Review umana obbligatoria; anche +1% vs processo attuale è progresso |
| Baseline | Regole CDM Subdued ricostruite in codice |

---

## 2. Architettura repo

```
modello-forecast/
├── testing/          # ML completo + fake data (SOURCE OF TRUTH)
├── production/       # connettori reali (skeleton)
├── data/manual/      # pdv_master, calendari team
├── docs/             # DATA_CONTRACTS.md
├── reports/          # HTML dashboard / schede
├── examples/         # esperimenti (es. XGBoost R) — non core
├── src/              # LEGACY — non usare
├── csv_MIRKONE.txt   # export Valido grezzo ~1.3GB — non ancora ingestito
└── requirements.txt
```

**Regola dura:** `testing/` ↛ `production/` e viceversa.

---

## 3. Contratto dati Valido

### tipodocumento

| Codice | Ruolo | Domanda (y) | Giacenza |
|--------|-------|-------------|----------|
| **100** | Vendita al pubblico | Sì (`abs(Qta)`) | Scarico |
| **47** | Fattura / carico | No | Carico |
| **20** | Azzeramento | No | Reset livello |
| Altri | Movimenti vari | No | Caso per caso |

### Colonne `csv_MIRKONE.txt`

`IDAnagrafica, iso3, data, IDArticolo, tipodocumento, DescrizioneSuStampa, Qta, PrezzoPubblico, PrezzoPubblicoReale, giacenza`

### Presente / mancante

| Blocco | Stato |
|--------|--------|
| Storico vendite + prezzi | ✅ nel CSV |
| Movimenti stock 47/20 | ✅ nel CSV |
| Catalogo (modello, tessuto, colore, classe…) | ❌ |
| Master PDV (city, mq, transit, commerciale…) | ❌ (solo 3 righe esempio) |
| Ora scontrino | ❌ solo data |
| Pipeline ingest Python | ❌ |

---

## 4. Feature engineering (logica)

Per ogni `as_of` e ogni PDV×SKU:

1. Storico fino a `as_of` (7/14/28 gg) → run-rate, trend, volatilità, giorni con vendita  
2. Target (solo train): vendite in `(as_of, as_of+15]`  
3. Cold start: giorni da lancio SKU, età PDV, flag new  
4. Prezzo/sconto da listino vs pagato (soglia sconto 5%)  
5. Store da `pdv_master` + meteo città  
6. `sku_weather`: profilo causale vendite×temperatura (no tag stagione)  
7. Calendario: mese, weekend, festivi, saldi, Fourier sin/cos  

**TrainConfig chiave** (`testing/config.py`):

- `horizon_sku_days=15`, `as_of_step_days=7`, `forecast_cycle_days=15`
- `train_window_days=365`, `n_val_as_of=4`, `epochs=80`, `batch_size=1024`, `lr=3e-3`

---

## 5. Modello dual-head (logica)

```
[Embeddings cat + bande] + [Numeric BN] + [sku_prior]
                    ↓
        Residual 256 → 256 → 128
           ↓                    ↓
    Head SKU              Head CAT (stop_grad)
 prior×scale + delta         softplus
        ↓
      y_sku (pezzi 15gg)
```

- Init: scale≈1, delta≈0 → parte dal naive  
- Sample weight ↑ su SKU meteo-sensibili e nuovi  
- Artefatti: `testing/data/output/dual_head_testing.keras`, `forecast_artefacts.pkl`

---

## 6. Inferenza e layer operativo

```
fact → features(as_of) → TF pred_model
     → blend/analogues → pred_final
     → CDM baseline → pred_cdm (confronto)
     → qty_suggerita / azione
     → previsioni_cdm.csv + dashboard
```

**Qty:**

```
daily = forecast / 15
need  = forecast + daily * transit_time
raw   = max(0, need - giacenza) * clip(0.45 + 0.55*confidence, 0.45, 1)
qty   = round(raw)
```

---

## 7. Metriche di riferimento (non inventarne altre)

Da `testing/data/output/report_metrics.json` (dati **sintetici**):

| Modello | MAE | Note |
|---------|-----|------|
| TF blend | **3.49** | R² 0.91 |
| TF raw | 3.48 | |
| LightGBM | 3.61 | challenger |
| Naive RR×15 | 6.19 | |
| CDM regole | 8.15 | |

Lift TF vs CDM ≈ **+57%**. Caveat obbligatorio: non ancora su Valido reale.

---

## 8. Comandi utili

```bash
cd ~/Desktop/modello-forecast

# Train
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_train

# Forecast ciclo CDM
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_forecast --as-of 2026-08-16

# Dashboard
.venv2/bin/python -m testing.run_bi_dashboard
open reports/cdm_dashboard.html

# Challenger LightGBM
OMP_NUM_THREADS=1 .venv2/bin/python -m testing.run_lgbm_challenger
```

---

## 9. Decisioni di design da NON contraddire

1. Niente tag stagione catalogo come input rete  
2. Domanda solo da tipodocumento 100; stock da 100+47+20  
3. Split temporale, non random  
4. Prior run-rate come ancora della head SKU  
5. Output = domanda attesa + suggerimento, non ordine  
6. Confrontare sempre vs CDM  
7. testing = reference ML finché Valido non è live  

---

## 10. Cosa chiedere all’altra AI (esempi di task)

- “Implementa l’ingest di `csv_MIRKONE.txt` in `fact_daily` rispettando 100/47/20”  
- “Porta dual_head + blend + replenishment da testing a production”  
- “Aggiungi export train/test CSV per challenger R/XGBoost sullo stesso split”  
- “Scrivi test di non-leakage sulle feature rispetto ad as_of”  

---

## 11. Mini glossario

| Termine | Significato |
|---------|-------------|
| as_of | Data di chiusura / “oggi” della previsione |
| pred_final | Previsione dopo blend confidence |
| pred_cdm | Baseline regole CDM |
| sku_prior | RR_14d × 15 |
| cold start | SKU/PDV giovani con poco storico |
| tipodocumento 100 | Vendita al pubblico |

---

Fine handoff. Per aggiornare questo file dopo cambi sostanziali: allineare metriche a `report_metrics.json` e contratto a `docs/DATA_CONTRACTS.md`.
