"""
report_pandas_visivo.py
Esegue la demo DualHead e produce un report HTML con pandas + grafici matplotlib.
Output: /tmp/modello-forecast/reports/report_forecast.html
"""

from __future__ import annotations

import base64
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# riusa la logica già collaudata
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dual_head_demo import (  # noqa: E402
    NUMERIC_COLS,
    DualHeadForecast,
    TrainConfig,
    build_training_frames,
    encode_frame,
    fit_encoders,
    generate_calendar,
    generate_daily_fact,
    predict,
    suggest_replenishment,
    train_model,
)

OUT_DIR = Path("/tmp/modello-forecast/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def run_experiment() -> dict:
    cfg = TrainConfig(epochs=60)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    fact = generate_daily_fact(date(2026, 2, 1), date(2026, 8, 31), rng)
    calendar = generate_calendar("IT", date(2026, 2, 1), date(2026, 8, 31))

    as_of_all = []
    d = date(2026, 3, 15)
    while d <= date(2026, 8, 10):
        as_of_all.append(d)
        d += timedelta(days=3)

    samples = build_training_frames(fact, calendar, cfg, as_of_all)
    cut = sorted(samples["as_of"].unique())[-2]
    train_df = samples[samples["as_of"] < cut].copy()
    test_df = samples[samples["as_of"] >= cut].copy()

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

    # training con history per il grafico
    history = {"epoch": [], "mae_sku": [], "mae_cat": [], "loss": []}
    cats, num, y_sku, y_cat = encode_frame(train_df, maps)
    num_n = (num - num_mean) / num_std
    ds = TensorDataset(
        torch.tensor(cats["country"]),
        torch.tensor(cats["pdv"]),
        torch.tensor(cats["sku"]),
        torch.tensor(cats["cat"]),
        torch.tensor(num_n, dtype=torch.float32),
        torch.tensor(y_sku, dtype=torch.float32),
        torch.tensor(y_cat, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(cfg.epochs):
        total, n = 0.0, 0
        mae_s, mae_c = 0.0, 0.0
        for c, p, s, k, x, ys, yc in loader:
            opt.zero_grad()
            pred_s, pred_c = model(c, p, s, k, x)
            loss = loss_fn(pred_s, ys) + 0.25 * loss_fn(pred_c, yc)
            loss.backward()
            opt.step()
            total += loss.item() * len(ys)
            n += len(ys)
            mae_s += (pred_s.detach() - ys).abs().sum().item()
            mae_c += (pred_c.detach() - yc).abs().sum().item()
        history["epoch"].append(epoch + 1)
        history["loss"].append(total / n)
        history["mae_sku"].append(mae_s / n)
        history["mae_cat"].append(mae_c / n)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoca {epoch+1}/{cfg.epochs}  MAE_sku={mae_s/n:.2f}")

    pred_s, pred_c = predict(model, test_df, maps, num_mean, num_std)
    y_true = test_df["y_sku"].to_numpy()
    naive = test_df["run_rate_14d"].to_numpy() * cfg.horizon_sku_days
    mae_model = float(np.abs(pred_s - y_true).mean())
    mae_naive = float(np.abs(naive - y_true).mean())

    result = test_df.copy()
    result["pred_nn"] = pred_s
    result["pred_cat"] = pred_c
    result["naive"] = naive
    result["err_nn"] = (result["pred_nn"] - result["y_sku"]).abs()
    result["err_naive"] = (result["naive"] - result["y_sku"]).abs()

    last = result["as_of"].max()
    focus = result[result["as_of"] == last].copy()
    focus_rows = []
    for _, r in focus.iterrows():
        if not (
            (r["pdv_id"] == "MIL_01" and r["sku_id"] in ("SKU_A", "SKU_Z"))
            or (r["pdv_id"] == "ROM_NEW" and r["sku_id"] == "SKU_A")
        ):
            continue
        immature = (r["has_history"] < 0.5) or (r["is_new_pdv"] > 0.5)
        op = suggest_replenishment(
            float(r["pred_nn"]), float(r["giacenza"]), 5.0, cfg.horizon_sku_days, bool(immature)
        )
        caso = (
            "SKU_nuovo"
            if r["sku_id"] == "SKU_Z"
            else ("PDV_nuovo" if int(r["is_new_pdv"]) == 1 else "caldo")
        )
        focus_rows.append(
            {
                "pdv_id": r["pdv_id"],
                "sku_id": r["sku_id"],
                "caso": caso,
                "y_vera": round(float(r["y_sku"]), 1),
                "pred_nn": round(float(r["pred_nn"]), 1),
                "naive": round(float(r["naive"]), 1),
                "giacenza": float(r["giacenza"]),
                **op,
            }
        )
    focus_df = pd.DataFrame(focus_rows)

    summary = pd.DataFrame(
        [
            {"metrica": "sample_train", "valore": len(train_df)},
            {"metrica": "sample_test", "valore": len(test_df)},
            {"metrica": "MAE_modello", "valore": round(mae_model, 2)},
            {"metrica": "MAE_naive", "valore": round(mae_naive, 2)},
            {"metrica": "miglioramento_vs_naive_%", "valore": round(100 * (1 - mae_model / mae_naive), 1)},
            {"metrica": "parametri_modello", "valore": sum(p.numel() for p in model.parameters())},
        ]
    )

    return {
        "history": pd.DataFrame(history),
        "test": result,
        "focus": focus_df,
        "summary": summary,
        "mae_model": mae_model,
        "mae_naive": mae_naive,
    }


def build_charts(exp: dict) -> dict[str, str]:
    imgs = {}

    # 1) learning curve
    h = exp["history"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(h["epoch"], h["mae_sku"], label="MAE SKU (train)", color="#2563eb", linewidth=2)
    ax.plot(h["epoch"], h["mae_cat"], label="MAE categoria (train)", color="#64748b", linewidth=1.5)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MAE (pezzi)")
    ax.set_title("Curva di apprendimento — DualHeadForecast")
    ax.legend()
    ax.grid(True, alpha=0.3)
    imgs["learning"] = fig_to_base64(fig)

    # 2) MAE confronto
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(
        ["Modello NN", "Naive run_rate×15"],
        [exp["mae_model"], exp["mae_naive"]],
        color=["#16a34a", "#94a3b8"],
    )
    ax.set_ylabel("MAE test (pezzi)")
    ax.set_title("Accuratezza sul test set (più basso = meglio)")
    for b, v in zip(bars, [exp["mae_model"], exp["mae_naive"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    imgs["mae"] = fig_to_base64(fig)

    # 3) focus pred vs true
    f = exp["focus"]
    if not f.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(f))
        w = 0.35
        labels = [f"{r.pdv_id}\n{r.sku_id}" for r in f.itertuples()]
        ax.bar(x - w / 2, f["y_vera"], width=w, label="Vendite vere", color="#64748b")
        ax.bar(x + w / 2, f["pred_nn"], width=w, label="Previsione NN", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Pezzi (15 giorni)")
        ax.set_title("Tre casi didattici — verità vs previsione")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        imgs["focus"] = fig_to_base64(fig)

    # 4) scatter test
    t = exp["test"]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(t["y_sku"], t["pred_nn"], alpha=0.75, c="#2563eb", edgecolors="none", s=40)
    lim = max(t["y_sku"].max(), t["pred_nn"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#94a3b8", label="perfetto")
    ax.set_xlabel("y vera (pezzi)")
    ax.set_ylabel("pred NN (pezzi)")
    ax.set_title("Test set — predizione vs realtà")
    ax.legend()
    ax.grid(True, alpha=0.3)
    imgs["scatter"] = fig_to_base64(fig)

    return imgs


def df_html(df: pd.DataFrame) -> str:
    styler = (
        df.style.hide(axis="index")
        .format(precision=2)
        .set_table_styles(
            [
                {"selector": "th", "props": [("background", "#0f172a"), ("color", "white"), ("padding", "8px")]},
                {"selector": "td", "props": [("padding", "8px"), ("border-bottom", "1px solid #e2e8f0")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
            ]
        )
    )
    return styler.to_html()


def write_report(exp: dict, imgs: dict[str, str]) -> Path:
    focus_show = exp["focus"].copy()
    test_preview = (
        exp["test"][
            ["as_of", "pdv_id", "sku_id", "y_sku", "pred_nn", "naive", "err_nn", "run_rate_14d", "is_new_pdv"]
        ]
        .sort_values("err_nn", ascending=False)
        .head(12)
        .copy()
    )
    test_preview["as_of"] = test_preview["as_of"].astype(str)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>Report Forecast DualHead</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .sub {{ color: #64748b; margin-bottom: 24px; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    img {{ max-width: 100%; height: auto; }}
    .kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }}
    .kpi .card {{ text-align: center; }}
    .kpi .v {{ font-size: 22px; font-weight: 700; }}
    .kpi .l {{ font-size: 12px; color: #64748b; }}
    @media (max-width: 800px) {{ .grid, .kpi {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Report visivo — Modello Forecast</h1>
    <p class="sub">DualHead PyTorch · dati fake demo · generato con pandas + matplotlib · ambiente agente</p>

    <div class="kpi">
      <div class="card"><div class="v">{exp["mae_model"]:.2f}</div><div class="l">MAE modello (test)</div></div>
      <div class="card"><div class="v">{exp["mae_naive"]:.2f}</div><div class="l">MAE naive</div></div>
      <div class="card"><div class="v">{exp["summary"].loc[exp["summary"].metrica=="miglioramento_vs_naive_%","valore"].iloc[0]}%</div><div class="l">Miglioramento vs naive</div></div>
      <div class="card"><div class="v">{int(exp["summary"].loc[exp["summary"].metrica=="sample_train","valore"].iloc[0])}</div><div class="l">Sample train</div></div>
    </div>

    <h2>1) Sintesi metriche (pandas)</h2>
    <div class="card">{df_html(exp["summary"])}</div>

    <h2>2) Apprendimento</h2>
    <div class="card"><img alt="learning curve" src="data:image/png;base64,{imgs["learning"]}" /></div>

    <div class="grid">
      <div class="card">
        <h2 style="margin-top:0">3) MAE modello vs naive</h2>
        <img alt="mae" src="data:image/png;base64,{imgs["mae"]}" />
      </div>
      <div class="card">
        <h2 style="margin-top:0">4) Pred vs realtà (test)</h2>
        <img alt="scatter" src="data:image/png;base64,{imgs["scatter"]}" />
      </div>
    </div>

    <h2>5) Casi didattici (forecast + operativo)</h2>
    <div class="card">
      <img alt="focus" src="data:image/png;base64,{imgs.get("focus", "")}" />
      {df_html(focus_show) if not focus_show.empty else ""}
    </div>

    <h2>6) Peggiori errori sul test (top 12)</h2>
    <div class="card">{df_html(test_preview)}</div>

    <p class="sub">Codice: /tmp/modello-forecast/src/report_pandas_visivo.py · Non usare questi numeri come prova Valido reale.</p>
  </div>
</body>
</html>
"""
    path = OUT_DIR / "report_forecast.html"
    path.write_text(html, encoding="utf-8")
    # anche CSV per ispezione
    exp["summary"].to_csv(OUT_DIR / "summary.csv", index=False)
    exp["focus"].to_csv(OUT_DIR / "focus_casi.csv", index=False)
    exp["test"].to_csv(OUT_DIR / "test_predictions.csv", index=False)
    return path


def main() -> None:
    print("1. Training + metriche...")
    exp = run_experiment()
    print("2. Grafici matplotlib...")
    imgs = build_charts(exp)
    print("3. Report HTML pandas...")
    path = write_report(exp, imgs)
    print(f"OK → {path}")
    print(exp["summary"].to_string(index=False))
    print()
    print(exp["focus"].to_string(index=False))


if __name__ == "__main__":
    main()
