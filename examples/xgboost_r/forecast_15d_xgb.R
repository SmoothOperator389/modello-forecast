# forecast_15d_xgb.R — esempio minimo, leggibile
# Idea: prevedere pezzi a 15 giorni con XGBoost (stesso problema del progetto)
#
# install.packages(c("xgboost", "data.table", "ggplot2"))
# Rscript examples/xgboost_r/forecast_15d_xgb.R

library(xgboost)
library(data.table)
library(ggplot2)

set.seed(42)
HORIZON <- 15

# --- 1) Dati toy: una riga = negozio × articolo × data di chiusura ----------
n <- 4000
dt <- data.table(
  pdv   = sample(sprintf("PDV%02d", 1:10), n, TRUE),
  sku   = sample(sprintf("SKU%03d", 1:50), n, TRUE),
  mese  = sample(1:12, n, TRUE),
  rr14  = pmax(0.05, rlnorm(n, meanlog = 0, sdlog = 0.6)),  # pezzi/giorno
  sconto = runif(n, 0, 30),
  temp   = 10 + 12 * sin(2 * pi * sample(1:12, n, TRUE) / 12) + rnorm(n)
)
dt[, prior := rr14 * HORIZON]
dt[, y := pmax(0, prior * (0.9 + 0.004 * sconto + 0.008 * (temp - 15)) + rnorm(n, 0, 1))]

# encoding categoriale grezzo
dt[, pdv_i := as.integer(as.factor(pdv)) - 1L]
dt[, sku_i := as.integer(as.factor(sku)) - 1L]

# --- 2) Split temporale simulato: prime 75% train, resto test ---------------
idx <- seq_len(n)
cut <- as.integer(0.75 * n)
train <- dt[idx <= cut]
test  <- dt[idx > cut]

feats <- c("rr14", "sconto", "temp", "mese", "prior", "pdv_i", "sku_i")
dtrain <- xgb.DMatrix(as.matrix(train[, ..feats]), label = train$y)
dtest  <- xgb.DMatrix(as.matrix(test[, ..feats]),  label = test$y)

# --- 3) XGBoost con early stopping -----------------------------------------
model <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    eval_metric = "mae",
    max_depth = 5,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 4,
    lambda = 1
  ),
  data = dtrain,
  nrounds = 800,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 40,
  verbose = 0
)

pred_xgb   <- pmax(0, predict(model, dtest))
pred_naive <- test$prior

mae <- function(p, y) mean(abs(p - y))
cat("MAE naive (prior): ", round(mae(pred_naive, test$y), 3), "\n")
cat("MAE XGBoost:       ", round(mae(pred_xgb, test$y), 3), "\n")
cat("best_iteration:    ", model$best_iteration, "\n")

# --- 4) Grafico pred vs reale ----------------------------------------------
plot_df <- rbindlist(list(
  data.table(reale = test$y, pred = pred_naive, modello = "Naive RR×15"),
  data.table(reale = test$y, pred = pred_xgb,   modello = "XGBoost")
))

p <- ggplot(plot_df, aes(reale, pred, color = modello)) +
  geom_point(alpha = 0.25, size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  facet_wrap(~ modello) +
  coord_equal() +
  labs(
    title = "Forecast 15 giorni — reale vs predetto",
    x = "Pezzi reali", y = "Pezzi predetti",
    caption = "Diagonale = previsione perfetta"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

out_dir <- "examples/xgboost_r/out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(out_dir, "pred_vs_real.png"), p, width = 8, height = 4, dpi = 140)

imp <- xgb.importance(feature_names = feats, model = model)
fwrite(imp, file.path(out_dir, "importance.csv"))
cat("Grafico: ", file.path(out_dir, "pred_vs_real.png"), "\n")
