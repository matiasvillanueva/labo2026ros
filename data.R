# ============================================================
# EDA PRELIMINAR PARA PARAMETRIZACIÓN DE LightGBM
# Dataset: clientes bancarios con variable objetivo clase_ternaria
# ============================================================

# ---- 0. LIBRERÍAS ----
pkgs <- c("data.table", "ggplot2", "corrplot", "lightgbm",
          "caret", "skimr", "moments", "dplyr", "tidyr", "gridExtra")
nuevos <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(nuevos)) install.packages(nuevos, dependencies = TRUE)
lapply(pkgs, library, character.only = TRUE)


# ---- 1. CARGA ----
# Ajustá la ruta si es necesario
df <- fread("Master/LABO1/labo2026ros/datasets_gerencial_competencia_2026.csv", stringsAsFactors = FALSE)
cat("Dimensiones:", nrow(df), "filas x", ncol(df), "columnas\n")


# ---- 2. VISTA GENERAL ----
cat("\n--- ESTRUCTURA ---\n")
str(df)

cat("\n--- RESUMEN SKIMR ---\n")
print(skim(df))


# ---- 3. VARIABLE OBJETIVO ----
cat("\n--- DISTRIBUCIÓN DE clase_ternaria ---\n")
freq <- table(df$clase_ternaria)
prop <- prop.table(freq)
print(cbind(n = freq, pct = round(prop * 100, 2)))

# Desbalanceo: ratio entre clases
cat("\nRatio BAJA/CONTINUA:", round(freq["BAJA+1"] / freq["CONTINUA"], 4),
    " | Ratio BAJA2/CONTINUA:", round(freq["BAJA+2"] / freq["CONTINUA"], 4), "\n")

ggplot(df, aes(x = clase_ternaria, fill = clase_ternaria)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribución de clase_ternaria", x = "", y = "Frecuencia") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("plot_01_target_dist.png", width = 6, height = 4)


# ---- 4. VALORES FALTANTES ----
na_pct <- df[, lapply(.SD, function(x) mean(is.na(x)) * 100)]
na_df  <- data.frame(variable = names(na_pct), pct_na = as.numeric(na_pct))
na_df  <- na_df[order(-na_df$pct_na), ]

cat("\n--- TOP 15 VARIABLES CON MÁS NA ---\n")
print(head(na_df, 15))

ggplot(na_df[na_df$pct_na > 0, ], aes(x = reorder(variable, pct_na), y = pct_na)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "% NAs por variable", x = "", y = "% NA") +
  theme_minimal()
ggsave("plot_02_missing.png", width = 8, height = 5)


# ---- 5. DISTRIBUCIONES NUMÉRICAS ----
nums <- names(df)[sapply(df, is.numeric)]
nums <- setdiff(nums, c("numero_de_cliente", "foto_mes"))

stats_num <- df[, lapply(.SD, function(x) {
  list(
    media   = mean(x, na.rm = TRUE),
    mediana = median(x, na.rm = TRUE),
    sd      = sd(x, na.rm = TRUE),
    skew    = skewness(x, na.rm = TRUE),
    kurt    = kurtosis(x, na.rm = TRUE),
    zeros   = mean(x == 0, na.rm = TRUE) * 100,
    p01     = quantile(x, 0.01, na.rm = TRUE),
    p99     = quantile(x, 0.99, na.rm = TRUE)
  )
}), .SDcols = nums]

cat("\n--- ESTADÍSTICAS NUMÉRICAS CLAVE ---\n")
stats_t <- as.data.frame(t(stats_num))
print(stats_t)

# Señalamos variables muy sesgadas (candidatas a log-transform o winsorizing)
cat("\n--- VARIABLES CON |SKEWNESS| > 3 ---\n")
skews <- sapply(df[, ..nums], skewness, na.rm = TRUE)
print(sort(abs(skews[abs(skews) > 3]), decreasing = TRUE))


# ---- 6. CORRELACIÓN (variables numéricas) ----
cor_mat <- cor(df[, ..nums], use = "pairwise.complete.obs")

png("plot_03_correlacion.png", width = 1200, height = 1000)
corrplot(cor_mat, method = "color", type = "upper",
         tl.cex = 0.6, tl.col = "black",
         title = "Matriz de correlación", mar = c(0,0,2,0))
dev.off()

# Pares altamente correlacionados (|r| > 0.85) → posible feature redundancy
high_cor <- which(abs(cor_mat) > 0.85 & upper.tri(cor_mat), arr.ind = TRUE)
if (nrow(high_cor) > 0) {
  cat("\n--- PARES CON CORRELACIÓN > 0.85 ---\n")
  cat("(Considerar eliminar uno de cada par para reducir redundancia)\n")
  for (i in seq_len(nrow(high_cor))) {
    r <- row.names(cor_mat)[high_cor[i, 1]]
    c <- colnames(cor_mat)[high_cor[i, 2]]
    cat(sprintf("  %s  vs  %s  => r = %.3f\n", r, c, cor_mat[high_cor[i, 1], high_cor[i, 2]]))
  }
}


# ---- 7. SEPARACIÓN TRAIN/TEST ESTRATIFICADA ----
set.seed(42)
idx_train <- createDataPartition(df$clase_ternaria, p = 0.70, list = FALSE)
train <- df[idx_train, ]
test  <- df[-idx_train, ]
cat("\nTrain:", nrow(train), " | Test:", nrow(test), "\n")


# ---- 8. PREPARACIÓN PARA LightGBM ----
# Convertir target a numérico (0=CONTINUA, 1=BAJA+1, 2=BAJA+2)
encode_target <- function(x) {
  case_when(x == "CONTINUA"  ~ 0L,
            x == "BAJA+1"   ~ 1L,
            x == "BAJA+2"   ~ 2L,
            TRUE             ~ NA_integer_)
}

excluir <- c("numero_de_cliente", "foto_mes", "clase_ternaria",
             "Master_fechaalta", "Visa_fechaalta")   # IDs / fechas

features <- setdiff(names(df), excluir)

# Convertir categorías de estado de tarjeta a factor numérico
cat_cols <- c("Master_status", "Visa_status")
for (col in cat_cols) {
  if (col %in% features) {
    train[[col]] <- as.integer(as.factor(train[[col]]))
    test[[col]]  <- as.integer(as.factor(test[[col]]))
  }
}

X_train <- as.matrix(train[, ..features])
y_train <- encode_target(train$clase_ternaria)
X_test  <- as.matrix(test[, ..features])
y_test  <- encode_target(test$clase_ternaria)

dtrain <- lgb.Dataset(X_train, label = y_train)
dtest  <- lgb.Dataset(X_test,  label = y_test, reference = dtrain)


# ---- 9. MODELO BASE LightGBM (para Feature Importance) ----
cat("\n--- ENTRENANDO MODELO BASE (50 rounds) ---\n")

# Pesos de clase para el desbalanceo
# CONTINUA=0 (mayoritaria) peso bajo; BAJA+1=1, BAJA+2=2 peso alto
pesos <- ifelse(y_train == 0, 1,
                ifelse(y_train == 1, 5, 10))

params_base <- list(
  objective        = "multiclass",
  num_class        = 3,
  metric           = "multi_logloss",
  learning_rate    = 0.05,
  num_leaves       = 31,
  min_data_in_leaf = 20,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq     = 5,
  verbose          = -1
)

modelo_base <- lgb.train(
  params   = params_base,
  data     = dtrain,
  nrounds  = 50,
  valids   = list(test = dtest),
  weights  = pesos,
  early_stopping_rounds = 10,
  verbose  = 0
)

# Feature importance
imp <- lgb.importance(modelo_base, percentage = TRUE)
cat("\n--- TOP 20 FEATURES (Gain) ---\n")
print(head(imp, 20))

lgb.plot.importance(imp, top_n = 20, measure = "Gain",
                    main = "Feature Importance (Gain) - modelo base")


# ---- 10. GUÍA DE PARÁMETROS LightGBM A TUNEAR ----
cat("\n
================================================================
  RECOMENDACIONES DE PARÁMETROS LightGBM según el EDA
================================================================

DESBALANCEO DE CLASES
  → Usar `class_weight` o `is_unbalance = TRUE`
  → Alternativamente, asignar `scale_pos_weight` por clase
  → Con multiclase: definir pesos manuales via `weights=` en lgb.train

COMPLEJIDAD DEL ÁRBOL
  → num_leaves: empezar en 31-63; con muchas features probar 127
  → max_depth:  -1 (sin límite) o 6-10 para regularizar
  → min_data_in_leaf: 20-50 (sube si hay overfitting)

REGULARIZACIÓN
  → lambda_l1 (L1): 0, 0.1, 1, 10
  → lambda_l2 (L2): 0, 0.1, 1, 10
  → min_gain_to_split: 0, 0.01, 0.1

VELOCIDAD / SAMPLING
  → feature_fraction: 0.6-0.9 (reduce overfitting y acelera)
  → bagging_fraction: 0.6-0.9
  → bagging_freq:    1 o 5

LEARNING RATE
  → learning_rate: 0.01-0.1; con early stopping se puede dejar 0.05
  → nrounds: usar early_stopping_rounds = 50 con lr bajo

MÉTRICAS RECOMENDADAS
  → multiclase: 'multi_logloss' o 'auc_mu'
  → si priorizás detección de BAJA+2: podés hacer One-vs-Rest con AUC

GRID SUGERIDO INICIAL
  num_leaves      : c(31, 63, 127)
  min_data_in_leaf: c(20, 50, 100)
  feature_fraction: c(0.6, 0.8)
  lambda_l2       : c(0, 1, 5)
  learning_rate   : 0.05 (fijo) + early stopping

================================================================
")


# ---- 11. CROSS-VALIDATION RÁPIDA PARA ESTIMAR nrounds ----
cat("\n--- CV para estimar nrounds óptimo ---\n")
cv_result <- lgb.cv(
  params    = params_base,
  data      = dtrain,
  nrounds   = 300,
  nfold     = 5,
  weights   = pesos,
  early_stopping_rounds = 30,
  verbose   = 0
)
best_iter <- cv_result$best_iter
cat(sprintf("Mejor iteración CV: %d  |  best score: %.5f\n",
            best_iter, cv_result$best_score))


cat("\n✅ EDA completado. Revisá los plots guardados y las recomendaciones de parámetros.\n")