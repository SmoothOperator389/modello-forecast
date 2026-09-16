# Challenger XGBoost in R — esempio didattico

Script: `forecast_xgb_example.R`

Allineato al protocollo del repo (orizzonte 15gg, split temporale, MAE vs naive).
**Non** sostituisce TensorFlow / pipeline Python.

## Setup

```r
install.packages(c("xgboost", "data.table", "Matrix"))
```

## Prova immediata (dati toy)

```bash
cd ~/Desktop/modello-forecast
Rscript examples/xgboost_r/forecast_xgb_example.R
```

Genera `examples/xgboost_r/data/toy_train.csv`, predizioni e importance.

## Con dati del progetto Python

1. Esporta dallo stesso split di `run_train` / challenger due CSV con almeno:
   - `as_of`, `y_sku`, `run_rate_14d`
   - feature numeriche utili (`sconto_pct`, `temp_media`, …)
   - categoriche (`country`, `pdv_id`, `sku_id`, `categoria_id`, …)
2. Salvali come:
   - `examples/xgboost_r/data/train.csv`
   - `examples/xgboost_r/data/test.csv`
3. Lancia:

```bash
USE_EXPORTED_CSV=1 Rscript examples/xgboost_r/forecast_xgb_example.R
```

## Cosa mostra il codice

| Pezzo | Ruolo |
|-------|--------|
| `make_toy_samples` | mini dataset PDV×SKU×as_of |
| split su `as_of` | holdout temporale |
| `xgb.train` + early stopping | anti-overfit basico |
| `metrics` | MAE / MAPE / R² |
| `xgb.importance` | interpretabilità per la ricercatrice |

## Bridge consigliato col repo

Python resta la source of truth delle feature; R rilegge CSV e fa challenger / paper.
Così non riscrivete Valido e la dashboard due volte.
