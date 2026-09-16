"""Training rolling — PRODUCTION (da completare con dati Valido)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from production.config import MODEL_DIR
from production.features.build_features import NUMERIC_COLS
from production.model.dual_head_tf import build_dual_head_model, fit_encoders, frame_to_tf_inputs


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    model_path: Path = MODEL_DIR / "dual_head.keras"
    encoders_path: Path = MODEL_DIR / "encoders.npz"


def train_and_save(train_df: pd.DataFrame, cfg: TrainConfig | None = None) -> keras.Model:
    cfg = cfg or TrainConfig()
    maps = fit_encoders(train_df)
    num_mean = train_df[NUMERIC_COLS].astype(float).mean().to_numpy(copy=True)
    num_std = train_df[NUMERIC_COLS].astype(float).std().to_numpy(copy=True)
    num_std[num_std == 0] = 1.0

    model = build_dual_head_model(
        n_numeric=len(NUMERIC_COLS),
        n_country=len(maps["country"]),
        n_pdv=len(maps["pdv_id"]),
        n_sku=len(maps["sku_id"]),
        n_cat=len(maps["categoria_id"]),
    )
    x = frame_to_tf_inputs(train_df, maps, num_mean, num_std)
    y = {
        "y_sku": train_df["y_sku"].astype(float).to_numpy(),
        "y_cat": train_df["y_cat"].astype(float).to_numpy(),
    }
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss={"y_sku": "mse", "y_cat": "mse"},
        metrics={"y_sku": "mae", "y_cat": "mae"},
    )
    model.fit(x, y, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=2)
    model.save(cfg.model_path)
    np.savez(cfg.encoders_path, maps=maps, num_mean=num_mean, num_std=num_std)
    return model
