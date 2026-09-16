"""Dual-head TensorFlow/Keras — PRODUCTION."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tensorflow import keras

from production.features.build_features import NUMERIC_COLS
from production.features.store_features import MQ_BANDS, WEATHER_REGIMES


def fit_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for col in ["country", "pdv_id", "sku_id", "categoria_id"]:
        uniq = sorted(df[col].astype(str).unique())
        maps[col] = {v: i for i, v in enumerate(uniq)}
    return maps


def build_dual_head_model(
    n_numeric: int,
    n_country: int,
    n_pdv: int,
    n_sku: int,
    n_cat: int,
) -> keras.Model:
    inp_country = keras.Input(shape=(1,), name="country", dtype="int32")
    inp_pdv = keras.Input(shape=(1,), name="pdv", dtype="int32")
    inp_sku = keras.Input(shape=(1,), name="sku", dtype="int32")
    inp_cat = keras.Input(shape=(1,), name="cat", dtype="int32")
    inp_mq_e = keras.Input(shape=(1,), name="mq_espo_b", dtype="int32")
    inp_mq_s = keras.Input(shape=(1,), name="mq_stock_b", dtype="int32")
    inp_weather = keras.Input(shape=(1,), name="weather_reg", dtype="int32")
    inp_num = keras.Input(shape=(n_numeric,), name="numeric", dtype="float32")

    flat = [
        keras.layers.Flatten()(keras.layers.Embedding(n, d)(inp))
        for n, d, inp in [
            (n_country, 4, inp_country),
            (n_pdv, 4, inp_pdv),
            (n_sku, 6, inp_sku),
            (n_cat, 4, inp_cat),
            (MQ_BANDS, 3, inp_mq_e),
            (MQ_BANDS, 3, inp_mq_s),
            (WEATHER_REGIMES, 3, inp_weather),
        ]
    ]
    x = keras.layers.Concatenate()(flat + [inp_num])
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation="relu")(x)

    model = keras.Model(
        inputs={
            "country": inp_country,
            "pdv": inp_pdv,
            "sku": inp_sku,
            "cat": inp_cat,
            "mq_espo_b": inp_mq_e,
            "mq_stock_b": inp_mq_s,
            "weather_reg": inp_weather,
            "numeric": inp_num,
        },
        outputs={
            "y_sku": keras.layers.Dense(1, activation="softplus", name="y_sku")(x),
            "y_cat": keras.layers.Dense(1, activation="softplus", name="y_cat")(x),
        },
    )
    return model


def frame_to_tf_inputs(df, maps, num_mean, num_std):
    num = (df[NUMERIC_COLS].astype(float).to_numpy() - num_mean) / num_std
    return {
        "country": df["country"].map(maps["country"]).to_numpy(dtype=np.int32).reshape(-1, 1),
        "pdv": df["pdv_id"].map(maps["pdv_id"]).to_numpy(dtype=np.int32).reshape(-1, 1),
        "sku": df["sku_id"].map(maps["sku_id"]).to_numpy(dtype=np.int32).reshape(-1, 1),
        "cat": df["categoria_id"].map(maps["categoria_id"]).to_numpy(dtype=np.int32).reshape(-1, 1),
        "mq_espo_b": df["mq_espositivi_band"].astype(int).to_numpy(dtype=np.int32).reshape(-1, 1),
        "mq_stock_b": df["mq_stock_band"].astype(int).to_numpy(dtype=np.int32).reshape(-1, 1),
        "weather_reg": df["weather_regime"].astype(int).to_numpy(dtype=np.int32).reshape(-1, 1),
        "numeric": num.astype(np.float32),
    }


def predict(model, df, maps, num_mean, num_std):
    pred = model.predict(frame_to_tf_inputs(df, maps, num_mean, num_std), verbose=0)
    return pred["y_sku"].reshape(-1), pred["y_cat"].reshape(-1)
