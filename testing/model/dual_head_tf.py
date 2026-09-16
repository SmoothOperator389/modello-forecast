"""Dual-head TensorFlow/Keras per TESTING — versione ottimizzata produzione."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from testing.config import DATA_DIR
from testing.features.build_features import NUMERIC_COLS
from testing.features.schema import BAND_INPUT_COLS, CATEGORICAL_COLS, EMB_DIMS
from testing.features.store_features import MQ_BANDS, WEATHER_REGIMES

UNK_TOKEN = "__UNK__"
UNK_DROPOUT = 0.02


def load_encoder_vocab(data_dir: Path | None = None) -> pd.DataFrame:
    """Anagrafica catalogo/negozi: vocabolario noto anche senza vendite in train."""
    root = data_dir or DATA_DIR
    parts: list[pd.DataFrame] = []
    products_path = root / "products.csv"
    pdv_path = root / "pdv_master.csv"
    if products_path.exists():
        prod = pd.read_csv(products_path)
        prod["categoria_id"] = prod.get("categoria_id", prod.get("categoria", "NA"))
        parts.append(prod)
    if pdv_path.exists():
        parts.append(pd.read_csv(pdv_path))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for col in CATEGORICAL_COLS:
        if col not in out.columns:
            out[col] = "NA"
    return out


def fit_encoders(
    train_df: pd.DataFrame,
    catalog_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, int]]:
    """Dizionario su train + anagrafica. Indice 0 = UNK (entità mai viste)."""
    frames = [train_df]
    if catalog_df is not None and not catalog_df.empty:
        frames.append(catalog_df)
    vocab = pd.concat(frames, ignore_index=True)
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        uniq = sorted({str(v) for v in vocab[col].astype(str).fillna("NA").unique() if str(v) != UNK_TOKEN})
        maps[col] = {UNK_TOKEN: 0, **{v: i + 1 for i, v in enumerate(uniq)}}
    return maps


def _map_cat(
    series: pd.Series,
    mapping: dict[str, int],
    unk_dropout: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    unk = int(mapping.get(UNK_TOKEN, 0))
    mapped = series.astype(str).fillna("NA").map(mapping)
    arr = mapped.fillna(unk).to_numpy(dtype=np.int32)
    if unk_dropout > 0 and rng is not None and len(arr):
        drop = rng.random(len(arr)) < unk_dropout
        arr = np.where(drop, unk, arr).astype(np.int32)
    return arr.reshape(-1, 1)


def _residual_block(x, units, dropout_rate=0.15):
    shortcut = x
    h = keras.layers.Dense(units)(x)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Activation("relu")(h)
    h = keras.layers.Dropout(dropout_rate)(h)
    h = keras.layers.Dense(units)(h)
    h = keras.layers.BatchNormalization()(h)
    if shortcut.shape[-1] != units:
        shortcut = keras.layers.Dense(units, use_bias=False)(shortcut)
    h = keras.layers.Add()([shortcut, h])
    h = keras.layers.Activation("relu")(h)
    return h


def build_dual_head_model(
    n_numeric: int,
    maps: dict[str, dict[str, int]],
    emb_dims: dict[str, int] | None = None,
) -> keras.Model:
    emb = emb_dims or EMB_DIMS

    inputs: dict[str, keras.Input] = {}
    embeddings = []

    for col in CATEGORICAL_COLS:
        if col == "pdv_id":
            inp_name = "pdv"
        elif col == "sku_id":
            inp_name = "sku"
        elif col == "categoria_id":
            inp_name = "cat"
        else:
            inp_name = col

        n = len(maps[col])
        dim = emb.get(col, 4)
        inputs[inp_name] = keras.Input(shape=(1,), name=inp_name, dtype="int32")
        embeddings.append(keras.layers.Flatten()(keras.layers.Embedding(n, dim)(inputs[inp_name])))

    for src_col, inp_name in BAND_INPUT_COLS.items():
        n_vocab = MQ_BANDS if "mq" in inp_name else WEATHER_REGIMES
        dim = emb["mq"] if "mq" in inp_name else emb["weather"]
        inputs[inp_name] = keras.Input(shape=(1,), name=inp_name, dtype="int32")
        embeddings.append(keras.layers.Flatten()(keras.layers.Embedding(n_vocab, dim)(inputs[inp_name])))

    inputs["numeric"] = keras.Input(shape=(n_numeric,), name="numeric", dtype="float32")
    inputs["sku_prior"] = keras.Input(shape=(1,), name="sku_prior", dtype="float32")
    num_bn = keras.layers.BatchNormalization(name="num_bn")(inputs["numeric"])
    x = keras.layers.Concatenate()(embeddings + [num_bn])

    x = _residual_block(x, 256, dropout_rate=0.15)
    x = _residual_block(x, 256, dropout_rate=0.15)
    x = _residual_block(x, 128, dropout_rate=0.10)

    # Testa SKU: run-rate × scala + residuo in pezzi (meteo/promo).
    # All'init scala≈1 e residuo=0 → parte dalla naive.
    sku_head = keras.layers.Dense(64, activation="relu")(x)
    sku_head = keras.layers.Dropout(0.10)(sku_head)
    sku_scale = keras.layers.Dense(
        1,
        activation="softplus",
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(0.541324854),
        name="sku_scale",
    )(sku_head)
    scaled = keras.layers.Multiply(name="sku_scaled")([inputs["sku_prior"], sku_scale])
    # Residuo in pezzi (meteo/promo): parte da 0, può alzare o abbassare rispetto al run-rate.
    sku_delta = keras.layers.Dense(
        1,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="sku_delta",
    )(sku_head)
    out_sku = keras.layers.ReLU(name="y_sku")(keras.layers.Add()([scaled, sku_delta]))

    cat_head = keras.layers.Dense(64, activation="relu")(
        keras.layers.Lambda(
            lambda t: tf.stop_gradient(t),
            output_shape=lambda s: s,
            name="cat_stop_grad",
        )(x)
    )
    cat_head = keras.layers.Dropout(0.10)(cat_head)
    out_cat = keras.layers.Dense(1, activation="softplus", name="y_cat")(cat_head)

    return keras.Model(inputs=inputs, outputs={"y_sku": out_sku, "y_cat": out_cat})


def frame_to_tf_inputs(
    df: pd.DataFrame,
    maps: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    unk_dropout: float = 0.0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    num = (df[NUMERIC_COLS].astype(float).to_numpy() - num_mean) / num_std

    out: dict[str, np.ndarray] = {"numeric": num.astype(np.float32)}
    rng = np.random.default_rng(seed) if unk_dropout > 0 else None

    col_to_input = {
        "country": "country",
        "pdv_id": "pdv",
        "sku_id": "sku",
        "categoria_id": "cat",
        "commerciale_riferimento": "commerciale_riferimento",
        "modello": "modello",
        "tessuto": "tessuto",
        "colore": "colore",
        "campagna": "campagna",
        "classe": "classe",
        "sclasse": "sclasse",
        "ssclasse": "ssclasse",
        "tipologia": "tipologia",
    }
    for col, inp_name in col_to_input.items():
        out[inp_name] = _map_cat(df[col], maps[col], unk_dropout=unk_dropout, rng=rng)

    for src_col, inp_name in BAND_INPUT_COLS.items():
        out[inp_name] = df[src_col].astype(int).to_numpy(dtype=np.int32).reshape(-1, 1)

    horizon = 15
    prior = df["run_rate_14d"].astype(float).to_numpy() * float(horizon)
    out["sku_prior"] = np.maximum(prior, 0.05).astype(np.float32).reshape(-1, 1)

    return out


def _temporal_val_mask(train_df: pd.DataFrame, n_val_as_of: int = 8) -> np.ndarray:
    as_ofs = sorted(train_df["as_of"].unique())
    n_val = min(max(1, n_val_as_of), max(1, len(as_ofs) // 4))
    val_set = set(as_ofs[-n_val:])
    return train_df["as_of"].isin(val_set).to_numpy()


def _sku_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Pesa di più gli SKU meteo-sensibili (profilo imparato dalle vendite, non dal tag)."""
    w = np.ones(len(df), dtype=np.float32)
    if "sku_share_hot" in df.columns and "sku_share_cold" in df.columns:
        sens = (df["sku_share_hot"].astype(float) - df["sku_share_cold"].astype(float)).abs()
        w = (1.0 + 1.8 * sens.clip(lower=0.0, upper=1.0)).to_numpy(dtype=np.float32)
        if "sku_temp_fit" in df.columns:
            mismatch = df["sku_temp_fit"].astype(float) < -0.15
            w = np.where(mismatch.to_numpy(), w * 1.35, w).astype(np.float32)
    if "is_new_sku" in df.columns:
        w[df["is_new_sku"].astype(float) >= 0.5] *= 1.4
    return w


def train_model(
    model: keras.Model,
    train_df: pd.DataFrame,
    maps: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    cfg,
) -> tuple[keras.Model, dict[str, list[float]]]:
    n_val = int(getattr(cfg, "n_val_as_of", 8))
    val_mask = _temporal_val_mask(train_df, n_val_as_of=n_val)
    train_part = train_df.loc[~val_mask]
    val_part = train_df.loc[val_mask]
    x_tr = frame_to_tf_inputs(
        train_part,
        maps,
        num_mean,
        num_std,
        unk_dropout=UNK_DROPOUT,
        seed=int(getattr(cfg, "seed", 42)),
    )
    x_val = frame_to_tf_inputs(val_part, maps, num_mean, num_std)
    y_tr = {
        "y_sku": train_part["y_sku"].astype(float).to_numpy(),
        "y_cat": np.log1p(train_part["y_cat"].astype(float).clip(lower=0).to_numpy()),
    }
    y_val = {
        "y_sku": val_part["y_sku"].astype(float).to_numpy(),
        "y_cat": np.log1p(val_part["y_cat"].astype(float).clip(lower=0).to_numpy()),
    }
    sw = _sku_sample_weights(train_part)
    sw_tr = {"y_sku": sw, "y_cat": sw}

    n_steps = max(1, len(y_tr["y_sku"]) // cfg.batch_size + 1)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.lr,
        decay_steps=cfg.epochs * n_steps,
        alpha=1e-5,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss={
            "y_sku": keras.losses.Huber(delta=5.0),
            "y_cat": keras.losses.Huber(delta=1.0),
        },
        loss_weights={"y_sku": 1.0, "y_cat": 0.15},
        metrics={"y_sku": "mae", "y_cat": "mae"},
    )

    patience = int(getattr(cfg, "early_stopping_patience", 20))
    min_delta = float(getattr(cfg, "early_stopping_min_delta", 0.02))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_y_sku_mae",
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta,
            mode="min",
        ),
    ]

    history = model.fit(
        x_tr,
        y_tr,
        sample_weight=sw_tr,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history.history


def predict(model: keras.Model, df: pd.DataFrame, maps, num_mean, num_std) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(frame_to_tf_inputs(df, maps, num_mean, num_std), verbose=0)
    return pred["y_sku"].reshape(-1), pred["y_cat"].reshape(-1)
