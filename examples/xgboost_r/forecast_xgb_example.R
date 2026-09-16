# Esempio didattico: forecast 15gg con XGBoost in R
# --------------------------------------------------
# Allineato al protocollo modello-forecast:
#   - target y = pezzi nei 15 giorni successivi ad as_of
#   - prior = run_rate_14d * 15
#   - split TEMPORALE (non random)
#   - metriche MAE / MAPE / R2 vs naive e vs "CDM" semplificato
#
# NON è il rewrite completo del repo (Valido, dual-head TF, dashboard…).
# Serve a te + ricercatrice per sperimentare lo stesso problema in R.
#
# Installazione (una volta):
#   install.packages(c("xgboost", "data.table", "Matrix"))
#
# Uso:
#   Rscript examples/xgboost_r/forecast_xgb_example.R
#
# Oppure, con export Python (stesso split del progetto):
#   Metti train.csv / test.csv in examples/xgboost_r/data/
#   con colonne: as_of, y_sku, run_rate_14d, + feature numeriche/categoriali
#   poi: USE_EXPORTED_CSV=1 Rscript examples/xgboost_r/forecast_xgb_example.R

suppressPackageStartupMessages({
  library(xgboost)
  library(data.table)
  library(Matrix)
})

HORIZON <- 15L
SEED <- 42L
set.seed(SEED)

ROOT <- tryCatch(
  {
    # quando lanciato con Rscript dal path del file
    args <- commandArgs(trailingOnly = FALSE)
    f <- grep("^--file=", args, value = TRUE)
    if (length(f)) dirname(normalizePath(sub("^--file=", "", f))) else getwd()
  },
  error = function(e) getwd()
)
DATA_DIR <- file.path(ROOT, "data")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

`%||%` <- function(a, b) if (!is.null(a) && !is.na(a) && a != "") a else b

# ---------------------------------------------------------------------------
# Metriche (stesso spirito di testing/run_lgbm_challenger.py)
# ---------------------------------------------------------------------------
metrics <- function(pred, y) {
  pred <- as.numeric(pred)
  y <- as.numeric(y)
  mae <- mean(abs(pred - y))
  mape <- mean(abs(pred - y) / pmax(y, 1)) * 100
  ss <- sum((y - mean(y))^2)
  r2 <- if (ss > 0) 1 - sum((y - pred)^2) / ss else NA_real_
  list(mae = round(mae, 4), mape = round(mape, 2), r2 = round(r2, 4))
}

print_metrics <- function(name, m) {
  cat(sprintf(
    "  %-18s  MAE=%6.3f  MAPE=%6.1f%%  R2=%6.3f\n",
    name, m$mae, m$mape, m$r2
  ))
}

# ---------------------------------------------------------------------------
# Dati sintetici MINI (se non hai ancora export Python)
# Grain: 1 riga = PDV x SKU x as_of
# ---------------------------------------------------------------------------
make_toy_samples <- function(n_pdv = 12L, n_sku = 40L, n_asof = 20L) {
  as_ofs <- as.Date("2025-07-01") + seq(0, by = 7, length.out = n_asof)
  grid <- CJ(
    pdv_id = sprintf("PDV_%02d", seq_len(n_pdv)),
    sku_id = sprintf("SKU_%03d", seq_len(n_sku)),
    as_of = as_ofs
  )
  n <- nrow(grid)
  grid[, country := fifelse(as.integer(substr(pdv_id, 5, 6)) <= 6, "ITA", "FRA")]
  grid[, categoria_id := fifelse(as.integer(substr(sku_id, 5, 7)) %% 3 == 0, "OUT", "MAG")]
  grid[, mese := as.integer(format(as_of, "%m"))]

  # segnale: run-rate + stagione grezza + rumore
  base_rr <- runif(n, 0.05, 2.5)
  season <- 1 + 0.25 * sin(2 * pi * (mese - 1) / 12)
  promo <- rbinom(n, 1, 0.15)
  grid[, run_rate_14d := pmax(0.01, base_rr * season * (1 + 0.2 * promo))]
  grid[, sconto_pct := fifelse(promo == 1, runif(n, 10, 35), runif(n, 0, 5))]
  grid[, is_new_sku := as.integer(runif(n) < 0.08)]
  grid[, temp_media := 8 + 14 * sin(2 * pi * (mese - 1) / 12) + rnorm(n, 0, 2)]
  grid[, mq_espositivi := as.numeric(80 + (as.integer(substr(pdv_id, 5, 6)) %% 5) * 15)]
  grid[, sku_prior := pmax(run_rate_14d * HORIZON, 0.05)]

  # target = prior * fattore + rumore (la "realtà" del toy)
  true_scale <- 0.85 + 0.01 * (temp_media - 15) + 0.004 * sconto_pct - 0.15 * is_new_sku
  grid[, y_sku := pmax(0, sku_prior * true_scale + rnorm(n, 0, 0.8))]
  grid[]
}

# ---------------------------------------------------------------------------
# Caricamento: export CSV dal progetto Python OPPURE toy
# ---------------------------------------------------------------------------
use_export <- identical(Sys.getenv("USE_EXPORTED_CSV", "0"), "1")
train_path <- file.path(DATA_DIR, "train.csv")
test_path <- file.path(DATA_DIR, "test.csv")

if (use_export && file.exists(train_path) && file.exists(test_path)) {
  cat(">> Carico export Python:", train_path, "\n")
  train <- fread(train_path)
  test <- fread(test_path)
} else {
  cat(">> Genero dati toy (per prova didattica). Per dati reali:\n")
  cat("   esporta train/test da Python e rilancia con USE_EXPORTED_CSV=1\n\n")
  all_df <- make_toy_samples()
  # split temporale: ultimi 3 as_of = test
  as_levels <- sort(unique(all_df$as_of))
  cut <- as_levels[length(as_levels) - 2L]
  train <- all_df[as_of < cut]
  test <- all_df[as_of >= cut]
  # salva per ispezionarli
  fwrite(train, file.path(DATA_DIR, "toy_train.csv"))
  fwrite(test, file.path(DATA_DIR, "toy_test.csv"))
  cat(sprintf("   cut_as_of=%s  n_train=%s  n_test=%s\n\n",
              as.character(cut), nrow(train), nrow(test)))
}

stopifnot("y_sku" %in% names(train), "run_rate_14d" %in% names(train))

# prior se manca
if (!"sku_prior" %in% names(train)) {
  train[, sku_prior := pmax(as.numeric(run_rate_14d) * HORIZON, 0.05)]
  test[, sku_prior := pmax(as.numeric(run_rate_14d) * HORIZON, 0.05)]
}

# ---------------------------------------------------------------------------
# Feature set (adatta i nomi a ciò che esporti da Python)
# ---------------------------------------------------------------------------
num_candidates <- c(
  "run_rate_14d", "run_rate_7d", "run_rate_28d",
  "sconto_pct", "temp_media", "mq_espositivi",
  "is_new_sku", "sku_prior", "mese"
)
cat_candidates <- c("country", "pdv_id", "sku_id", "categoria_id")

num_cols <- intersect(num_candidates, names(train))
cat_cols <- intersect(cat_candidates, names(train))

# encoding categoriale semplice: interi da train; unseen -> -1
encode_cats <- function(dt, maps = NULL, fit = FALSE) {
  out <- copy(dt)
  if (fit) {
    maps <- list()
    for (col in cat_cols) {
      vals <- sort(unique(as.character(out[[col]])))
      maps[[col]] <- setNames(seq_along(vals) - 1L, vals) # 0..K-1
    }
  }
  for (col in cat_cols) {
    m <- maps[[col]]
    codes <- unname(m[as.character(out[[col]])])
    codes[is.na(codes)] <- -1L
    out[, (col) := as.integer(codes)]
  }
  list(dt = out, maps = maps)
}

enc_tr <- encode_cats(train, fit = TRUE)
enc_te <- encode_cats(test, maps = enc_tr$maps, fit = FALSE)
train_e <- enc_tr$dt
test_e <- enc_te$dt
maps <- enc_tr$maps

feat_cols <- c(num_cols, cat_cols)
for (col in num_cols) {
  train_e[, (col) := as.numeric(get(col))]
  test_e[, (col) := as.numeric(get(col))]
}

X_train <- as.matrix(train_e[, ..feat_cols])
y_train <- as.numeric(train_e$y_sku)
X_test <- as.matrix(test_e[, ..feat_cols])
y_test <- as.numeric(test_e$y_sku)

# indici colonne categoriche per xgboost (0-based)
cat_idx <- which(feat_cols %in% cat_cols) - 1L

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest <- xgb.DMatrix(X_test, label = y_test)

# validazione temporale dentro il train: ultimi as_of del train
if ("as_of" %in% names(train_e)) {
  as_tr <- sort(unique(train_e$as_of))
  n_val <- min(4L, max(1L, length(as_tr) %/% 5))
  val_as <- tail(as_tr, n_val)
  is_val <- train_e$as_of %in% val_as
  dtr <- xgb.DMatrix(X_train[!is_val, , drop = FALSE], label = y_train[!is_val])
  dval <- xgb.DMatrix(X_train[is_val, , drop = FALSE], label = y_train[is_val])
  watch <- list(train = dtr, val = dval)
  train_for_fit <- dtr
} else {
  watch <- list(train = dtrain)
  train_for_fit <- dtrain
  dval <- NULL
}

# ---------------------------------------------------------------------------
# Train XGBoost — iperparametri "prudenti" anti-overfit (didattici)
# ---------------------------------------------------------------------------
params <- list(
  objective = "reg:squarederror",
  eval_metric = "mae",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  lambda = 1.0,   # L2
  alpha = 0.0,    # L1
  tree_method = "hist"
)

cat(">> Training XGBoost…\n")
bst <- xgb.train(
  params = params,
  data = train_for_fit,
  nrounds = 2000,
  watchlist = watch,
  early_stopping_rounds = 50,
  verbose = 0
)
cat(sprintf("   best_iteration = %s\n\n", bst$best_iteration))

pred_xgb <- pmax(0, predict(bst, dtest, ntreelimit = bst$best_iteration))

# Baseline naive = prior (run_rate * 15)
pred_naive <- as.numeric(test_e$sku_prior)

# CDM "toy": prior * piccolo coeff stagionale (ombra del baseline regole)
if ("mese" %in% names(test_e)) {
  season_c <- 1 + 0.1 * sin(2 * pi * (as.numeric(test_e$mese) - 1) / 12)
} else {
  season_c <- 1
}
pred_cdm <- pmax(0, pred_naive * season_c)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
cat("=== Metriche sul TEST (holdout temporale) ===\n")
print_metrics("Naive RR×15", metrics(pred_naive, y_test))
print_metrics("CDM toy", metrics(pred_cdm, y_test))
print_metrics("XGBoost", metrics(pred_xgb, y_test))

imp <- xgb.importance(feature_names = feat_cols, model = bst)
cat("\n=== Top 10 feature (Gain) ===\n")
print(head(imp, 10))

imp_path <- file.path(DATA_DIR, "xgb_importance.csv")
fwrite(imp, imp_path)

pred_path <- file.path(DATA_DIR, "xgb_test_predictions.csv")
fwrite(
  data.table(
    y_true = y_test,
    pred_xgb = pred_xgb,
    pred_naive = pred_naive,
    pred_cdm = pred_cdm
  ),
  pred_path
)

cat("\nSalvati:\n")
cat(" ", imp_path, "\n")
cat(" ", pred_path, "\n")
cat("\nNota: questo è un esempio didattico. In un rewrite completo andrebbero\n")
cat("aggiunti: import Valido (doc 100/47/20), build feature rolling, blend\n")
cat("cold-start, qty suggerita, dashboard.\n")
