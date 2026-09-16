"""
train_dual_head_demo.py
Mini-training DualHeadForecast su dati fake + inference + layer operativo.

Idea rolling:
  stato a fine periodo (feature da reali) → target = vendite nei N giorni successivi
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from store_features import MQ_BANDS, WEATHER_REGIMES, attach_store_weather


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    horizon_sku_days: int = 15
    run_rate_window: int = 14
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42


# =============================================================================
# 1) GENERATORE DATI FAKE (più storia del solo abbozzo)
# =============================================================================

def generate_daily_fact(start: date, end: date, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simula fact giornaliero tipo Valido.
    Tre casi didattici dentro il dataset:
      - MIL_01 + SKU_A  (caldo)
      - ROM_NEW + SKU_A (PDV nuovo da 2026-07-20)
      - MIL_01 + SKU_Z  (SKU nuovo da 2026-08-08)
    + altri SKU/PDV per dare massa al training
    """
    stores = [
        ("IT", "MIL_01", date(2024, 3, 1), 1.0),   # maturo, alto traffico
        ("IT", "ROM_NEW", date(2026, 7, 20), 0.35),  # nuovo, traffico basso
        ("IT", "NAP_02", date(2023, 6, 1), 0.8),
    ]
    skus = [
        ("SKU_A", "MAG", date(2026, 2, 1), 2.0),   # caldo
        ("SKU_B", "MAG", date(2026, 3, 15), 1.2),
        ("SKU_C", "ACC", date(2026, 1, 10), 0.8),
        ("SKU_Z", "MAG", date(2026, 8, 8), 1.5),   # brand new
    ]

    rows = []
    d = start
    while d <= end:
        weekend = 1.25 if d.weekday() >= 5 else 1.0
        for country, pdv, opened, traffic in stores:
            if d < opened:
                continue
            pdv_age = (d - opened).days
            new_pdv_factor = 0.4 + 0.6 * min(pdv_age / 60.0, 1.0)  # ramp-up 60gg
            for sku, cat, launch, strength in skus:
                if d < launch:
                    continue
                age = (d - launch).days
                # cold start SKU: prime 2 settimane più basse
                sku_factor = 0.35 + 0.65 * min(age / 14.0, 1.0)
                lam = strength * traffic * new_pdv_factor * sku_factor * weekend
                qty = int(rng.poisson(max(lam, 0.05)))
                # giacenza grezza (non realistica ma utile)
                giac = int(max(0, 30 - qty + rng.integers(-3, 4)))
                prezzo = 29.99 if cat == "MAG" else 19.99
                if sku == "SKU_Z":
                    prezzo = 23.99
                rows.append((d, country, pdv, sku, cat, qty, giac, prezzo, opened, launch))
        d += timedelta(days=1)

    return pd.DataFrame(
        rows,
        columns=[
            "data", "country", "pdv_id", "sku_id", "categoria_id",
            "vendite_qty", "giacenza_qty", "prezzo_pagato",
            "data_apertura", "data_lancio_sku",
        ],
    )


def generate_calendar(country: str, start: date, end: date) -> pd.DataFrame:
    rows = []
    d = start
    while d <= end:
        week = int(d.strftime("%W"))
        angle = 2 * np.pi * week / 52.0
        rows.append(
            {
                "data": d,
                "country": country,
                "mese": d.month,
                "settimana_anno": week,
                "sin_anno": float(np.sin(angle)),
                "cos_anno": float(np.cos(angle)),
                "is_festivo": int(d == date(2026, 8, 15)),
                "is_saldi": int(d.month == 7 and d.day <= 31),  # semplificato
                "giorni_da_inizio_collezione": (d - date(2026, 7, 1)).days,
            }
        )
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# =============================================================================
# 2) DATASET ROLLING: feature a as_of → y = vendite prossimi 15gg
# =============================================================================

def build_training_frames(
    fact: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: TrainConfig,
    as_of_list: list[date],
) -> pd.DataFrame:
    """
    Per ogni as_of e ogni (country,pdv,sku):
      feature dalla finestra [as_of-13, as_of]
      target = somma vendite (as_of+1 .. as_of+horizon)
    """
    samples = []
    for as_of in as_of_list:
        w_start = as_of - timedelta(days=cfg.run_rate_window - 1)
        y_end = as_of + timedelta(days=cfg.horizon_sku_days)

        hist = fact[(fact["data"] >= w_start) & (fact["data"] <= as_of)]
        future = fact[(fact["data"] > as_of) & (fact["data"] <= y_end)]
        if hist.empty or future.empty:
            continue

        g = (
            hist.groupby(["country", "pdv_id", "sku_id", "categoria_id"], as_index=False)
            .agg(
                vendite_window=("vendite_qty", "sum"),
                giorni_con_vendita=("vendite_qty", lambda s: int((s > 0).sum())),
                giacenza=("giacenza_qty", "last"),
                prezzo_pagato=("prezzo_pagato", "last"),
                data_apertura=("data_apertura", "first"),
                data_lancio_sku=("data_lancio_sku", "first"),
            )
        )
        y = (
            future.groupby(["country", "pdv_id", "sku_id"], as_index=False)
            .agg(y_sku=("vendite_qty", "sum"))
        )
        m = g.merge(y, on=["country", "pdv_id", "sku_id"], how="inner")
        m["run_rate_14d"] = m["vendite_window"] / float(cfg.run_rate_window)
        m["has_history"] = (m["giorni_con_vendita"] > 0).astype(float)
        m["giorni_da_lancio"] = m["data_lancio_sku"].apply(lambda d: (as_of - d).days)
        m["pdv_age_days"] = m["data_apertura"].apply(lambda d: (as_of - d).days)
        m["is_new_pdv"] = (m["pdv_age_days"] < 60).astype(float)

        cal = calendar[calendar["data"] == as_of]
        m = m.merge(
            cal[
                [
                    "country",
                    "mese",
                    "sin_anno",
                    "cos_anno",
                    "is_festivo",
                    "is_saldi",
                    "giorni_da_inizio_collezione",
                ]
            ],
            on="country",
            how="left",
        )
        # mq PDV + meteo (API) + bande per embedding
        m = attach_store_weather(m, as_of, cfg.run_rate_window)
        # proxy categoria: somma y degli sku stessa cat/country nello stesso as_of
        # (per multi-task grezzo nella demo)
        cat_y = (
            m.groupby(["country", "categoria_id"], as_index=False)["y_sku"]
            .sum()
            .rename(columns={"y_sku": "y_cat"})
        )
        m = m.merge(cat_y, on=["country", "categoria_id"], how="left")
        m["as_of"] = as_of
        samples.append(m)

    return pd.concat(samples, ignore_index=True)


# =============================================================================
# 3) ENCODING + MODELLO
# =============================================================================

class DualHeadForecast(nn.Module):
    def __init__(self, n_numeric: int, n_country: int, n_pdv: int, n_sku: int, n_cat: int):
        super().__init__()
        self.emb_country = nn.Embedding(n_country, 4)
        self.emb_pdv = nn.Embedding(n_pdv, 4)
        self.emb_sku = nn.Embedding(n_sku, 6)
        self.emb_cat = nn.Embedding(n_cat, 4)
        # embedding su bande mq + regime meteo (da riunione: mq espositivi, mq stock, meteo)
        self.emb_mq_espo = nn.Embedding(MQ_BANDS, 3)
        self.emb_mq_stock = nn.Embedding(MQ_BANDS, 3)
        self.emb_weather = nn.Embedding(WEATHER_REGIMES, 3)
        in_dim = 4 + 4 + 6 + 4 + 3 + 3 + 3 + n_numeric
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head_sku = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())
        self.head_cat = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, country, pdv, sku, cat, mq_espo_b, mq_stock_b, weather_reg, numeric):
        x = torch.cat(
            [
                self.emb_country(country),
                self.emb_pdv(pdv),
                self.emb_sku(sku),
                self.emb_cat(cat),
                self.emb_mq_espo(mq_espo_b),
                self.emb_mq_stock(mq_stock_b),
                self.emb_weather(weather_reg),
                numeric,
            ],
            dim=1,
        )
        h = self.encoder(x)
        return self.head_sku(h).squeeze(1), self.head_cat(h).squeeze(1)


NUMERIC_COLS = [
    "run_rate_14d",
    "has_history",
    "giorni_da_lancio",
    "pdv_age_days",
    "is_new_pdv",
    "giacenza",
    "prezzo_pagato",
    "mq_espositivi",
    "mq_stock",
    "mq_totali",
    "numero_fatturazioni",
    "temp_media",
    "giorni_pioggia",
    "mese",
    "sin_anno",
    "cos_anno",
    "is_festivo",
    "is_saldi",
    "giorni_da_inizio_collezione",
]


def fit_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for col in ["country", "pdv_id", "sku_id", "categoria_id"]:
        uniq = sorted(df[col].astype(str).unique())
        maps[col] = {v: i for i, v in enumerate(uniq)}
    return maps


def encode_frame(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    country = np.array([maps["country"][str(v)] for v in df["country"]], dtype=np.int64)
    pdv = np.array([maps["pdv_id"][str(v)] for v in df["pdv_id"]], dtype=np.int64)
    sku = np.array([maps["sku_id"][str(v)] for v in df["sku_id"]], dtype=np.int64)
    cat = np.array([maps["categoria_id"][str(v)] for v in df["categoria_id"]], dtype=np.int64)
    cats = {
        "country": country,
        "pdv": pdv,
        "sku": sku,
        "cat": cat,
        "mq_espo_b": df["mq_espositivi_band"].astype(int).to_numpy(),
        "mq_stock_b": df["mq_stock_band"].astype(int).to_numpy(),
        "weather_reg": df["weather_regime"].astype(int).to_numpy(),
    }

    num = df[NUMERIC_COLS].astype(float).to_numpy()
    # standardizzazione semplice sul train (qui caller passa già split)
    y_sku = df["y_sku"].astype(float).to_numpy()
    y_cat = df["y_cat"].astype(float).to_numpy()
    return cats, num, y_sku, y_cat


def train_model(
    model: DualHeadForecast,
    train_df: pd.DataFrame,
    maps: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    cfg: TrainConfig,
) -> DualHeadForecast:
    cats, num, y_sku, y_cat = encode_frame(train_df, maps)
    num = (num - num_mean) / num_std

    ds = TensorDataset(
        torch.tensor(cats["country"]),
        torch.tensor(cats["pdv"]),
        torch.tensor(cats["sku"]),
        torch.tensor(cats["cat"]),
        torch.tensor(cats["mq_espo_b"]),
        torch.tensor(cats["mq_stock_b"]),
        torch.tensor(cats["weather_reg"]),
        torch.tensor(num, dtype=torch.float32),
        torch.tensor(y_sku, dtype=torch.float32),
        torch.tensor(y_cat, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.epochs):
        total, n = 0.0, 0
        mae_sku, mae_cat = 0.0, 0.0
        for c, p, s, k, me, ms, wr, x, ys, yc in loader:
            opt.zero_grad()
            pred_s, pred_c = model(c, p, s, k, me, ms, wr, x)
            # multi-task: forecast SKU + forecast categoria
            loss = loss_fn(pred_s, ys) + 0.25 * loss_fn(pred_c, yc)
            loss.backward()
            opt.step()
            total += loss.item() * len(ys)
            n += len(ys)
            mae_sku += (pred_s.detach() - ys).abs().sum().item()
            mae_cat += (pred_c.detach() - yc).abs().sum().item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"epoca {epoch+1:>2}/{cfg.epochs}  "
                f"loss={total/n:.3f}  MAE_sku={mae_sku/n:.2f}  MAE_cat={mae_cat/n:.2f}"
            )
    return model


@torch.no_grad()
def predict(
    model: DualHeadForecast,
    df: pd.DataFrame,
    maps: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    cats, num, _, _ = encode_frame(df, maps)
    num = (num - num_mean) / num_std
    pred_s, pred_c = model(
        torch.tensor(cats["country"]),
        torch.tensor(cats["pdv"]),
        torch.tensor(cats["sku"]),
        torch.tensor(cats["cat"]),
        torch.tensor(cats["mq_espo_b"]),
        torch.tensor(cats["mq_stock_b"]),
        torch.tensor(cats["weather_reg"]),
        torch.tensor(num, dtype=torch.float32),
    )
    return pred_s.numpy(), pred_c.numpy()


def suggest_replenishment(forecast: float, giacenza: float, transit: float, horizon: int, immature: bool) -> dict:
    daily = forecast / max(horizon, 1)
    need = forecast + daily * transit
    raw = max(0.0, need - giacenza)
    if immature:
        raw *= 0.5
    qty = float(np.rint(raw))
    return {
        "vendite_previste_15gg": round(float(forecast), 1),
        "giacenza": giacenza,
        "qty_suggerita": qty,
        "azione": "monitora" if qty == 0 else "valuta_riassortimento",
    }


# =============================================================================
# 4) MAIN
# =============================================================================

def main() -> None:
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # storia più lunga + as_of più fitti → più sample (toy ma meno povero)
    fact = generate_daily_fact(date(2026, 2, 1), date(2026, 8, 31), rng)
    calendar = generate_calendar("IT", date(2026, 2, 1), date(2026, 8, 31))

    # as_of ogni 3 giorni
    as_of_all = []
    d = date(2026, 3, 15)
    while d <= date(2026, 8, 10):
        as_of_all.append(d)
        d += timedelta(days=3)

    samples = build_training_frames(fact, calendar, cfg, as_of_all)
    print(f"1. Sample rolling generati: {len(samples)} righe")

    # split temporale: ultimi 2 as_of = test
    cut = sorted(samples["as_of"].unique())[-2]
    train_df = samples[samples["as_of"] < cut].copy()
    test_df = samples[samples["as_of"] >= cut].copy()
    print(f"   train={len(train_df)}  test={len(test_df)}  cut_as_of>={cut}")

    maps = fit_encoders(samples)
    num_train = train_df[NUMERIC_COLS].astype(float).to_numpy()
    num_mean = num_train.mean(axis=0)
    num_std = num_train.std(axis=0)
    num_std[num_std == 0] = 1.0

    model = DualHeadForecast(
        n_numeric=len(NUMERIC_COLS),
        n_country=len(maps["country"]),
        n_pdv=len(maps["pdv_id"]),
        n_sku=len(maps["sku_id"]),
        n_cat=len(maps["categoria_id"]),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"2. Modello DualHead — parametri: {n_params}")
    print("3. Training...")
    model = train_model(model, train_df, maps, num_mean, num_std, cfg)

    pred_s, pred_c = predict(model, test_df, maps, num_mean, num_std)
    mae = np.abs(pred_s - test_df["y_sku"].to_numpy()).mean()
    # baseline naive: run_rate * horizon
    naive = test_df["run_rate_14d"].to_numpy() * cfg.horizon_sku_days
    mae_naive = np.abs(naive - test_df["y_sku"].to_numpy()).mean()
    print(f"\n4. Test MAE SKU: modello={mae:.2f}  naive(run_rate*15)={mae_naive:.2f}")

    # focus sui 3 casi didattici all'ultimo as_of di test
    last = test_df["as_of"].max()
    focus = test_df[test_df["as_of"] == last].copy()
    # riallinea predizioni focus
    mask = test_df["as_of"] == last
    focus = focus.copy()
    focus["pred_sku"] = pred_s[mask.to_numpy()]
    focus["pred_cat"] = pred_c[mask.to_numpy()]

    interesting = focus[
        ((focus["pdv_id"] == "MIL_01") & (focus["sku_id"].isin(["SKU_A", "SKU_Z"])))
        | ((focus["pdv_id"] == "ROM_NEW") & (focus["sku_id"] == "SKU_A"))
    ].copy()

    rows = []
    for _, r in interesting.iterrows():
        immature = (r["has_history"] < 0.5) or (r["is_new_pdv"] > 0.5)
        op = suggest_replenishment(
            forecast=float(r["pred_sku"]),
            giacenza=float(r["giacenza"]),
            transit=5.0,
            horizon=cfg.horizon_sku_days,
            immature=bool(immature),
        )
        rows.append(
            {
                "as_of": r["as_of"],
                "pdv_id": r["pdv_id"],
                "sku_id": r["sku_id"],
                "y_vera": r["y_sku"],
                "pred_nn": round(float(r["pred_sku"]), 1),
                "pred_cat_head": round(float(r["pred_cat"]), 1),
                "run_rate_14d": round(float(r["run_rate_14d"]), 3),
                "is_new_pdv": int(r["is_new_pdv"]),
                "has_history": int(r["has_history"]),
                **op,
            }
        )

    print("\n5. Focus casi didattici (NN addestrata + operativo):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
