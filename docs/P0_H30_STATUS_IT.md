# Report P0 h30 / SKU quality (IT) — 12 Nov 2026 (Europe/Rome)

## Verdetto

**Blocco rete (ancora attivo, retry Europe/Rome 17:06):** il cloud agent managed **non raggiunge** `modelcdm@10.11.1.139` (ping/SSH timeout). Nessun `CURSOR_API_KEY` / `TS_AUTHKEY` in env. Worker Mac **BERGANTINO-MAC** connected + `eligibleForSubagent`, ma da questo runtime non c’è tool CreateAgent/API key per spawnarlo. Retrain VM / holdout Sep after **pending**.

Policy **Jul/Aug = NON AFFIDABILE** è codificata: **nessuna attesa Export luglio**. BI API: `/api/month/reliability`, `/api/scoreboard`.

## Cosa è stato preparato nel repo (branch `cursor/p0-h30-sku-quality-5008`)

| Pezzo | Path |
|-------|------|
| Lock training (no orphan) | `production/ops/bi_training_lock.py` |
| Jul/Aug NON AFFIDABILE | `production/ops/month_reliability.py` |
| Exclude poisoned as_of | `production/ops/poisoned_as_of.py` |
| YoY / seasonality h30 | `production/ops/h30_yoy_features.py` |
| Train P0 h30 | `production/run_train_h30_p0.py` |
| tip_budget → weekly/monthly | `production/ops/tip_budget_pipeline.py` |
| Scoreboard CDM top-80 | `production/ops/bi_scoreboard.py` |
| Runner nohup VM | `scripts/vm_p0_h30_runner.sh` |

## Metriche Sep h30 CDM wMAPE (before)

Da artefatto già in repo (`reports/data/cdm_compare_h30_h60_020H3LI0KU.json`), **test globale nativo h30**:

- **wMAPE h30 ≈ 43.07%** (n=328 570, MAE 4.37) — baseline «before» finché non si rigenera holdout fair post-retrain.

**After:** *n/d* (serve run su VM).

## Come sbloccare e monitorare (VM)

Sul Mac (LAN) o dopo tunnel/Tailscale:

```bash
ssh modelcdm@10.11.1.139
cd /opt/modello-forecast
# deploy branch, poi:
mkdir -p logs
nohup bash scripts/vm_p0_h30_runner.sh all > logs/p0_h30_runner.nohup.out 2>&1 &
echo $! | tee logs/p0_h30_runner.pid
tail -f logs/p0_h30_runner.log logs/train_h30_p0.out
```

Lock: `python -m production.ops.bi_training_lock status`  
BI: `http://10.11.1.139:8765`

## Gap rimasti

1. Accesso SSH cloud→VM (Tailscale / My Machine BERGANTINO-MAC / reverse tunnel).
2. Codice VM-only (`bi_engine`, tip_budget reale, pipeline weekly) non è su GitHub `main` — va syncato da `/opt/modello-forecast`.
3. Holdout Sep (+Oct) fair Confronta + scoreboard live post-retrain.
4. wMAPE Sep h30 CDM **after** da misurare dopo train.

## Log cloud agent

- Probe SSH: `logs/vm_ssh_probe.log` (PID in `logs/vm_ssh_probe.pid`)
- Handoff peer/Mac: store `HANDOFF_P0_H30.md`
