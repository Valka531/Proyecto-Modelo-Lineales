# ============================================================
# PROYECTO MEA: MODELOS ESTADISTICOS ACTIVOS
# ANALISIS COMPLETO DE CARS93
# Modelos: MCO, VIF, BIC, Ridge, Lasso y Elastic Net
# Seleccion de variables: stepAIC con AIC y BIC
# Evaluacion predictiva: conjunto de prueba y 100 particiones
# ============================================================


# ============================================================
# 1. CONFIGURACION Y CARGA DE DATOS
# ============================================================

options(stringsAsFactors = FALSE)

paquetes <- c("MASS", "glmnet", "leaps", "car")

faltantes <- paquetes[
  !vapply(paquetes, requireNamespace, logical(1), quietly = TRUE)
]

if (length(faltantes) > 0) {
  install.packages(faltantes)
}

library(MASS)
library(glmnet)
library(leaps)
library(car)

data("Cars93", package = "MASS")

View(Cars93)

respuesta <- "MPG.city"

predictores <- c(
  "Min.Price",
  "Price",
  "Max.Price",
  "MPG.highway",
  "EngineSize",
  "Horsepower",
  "RPM",
  "Rev.per.mile",
  "Fuel.tank.capacity",
  "Passengers",
  "Length",
  "Wheelbase",
  "Width",
  "Turn.circle",
  "Rear.seat.room",
  "Weight"
)

variables <- c(respuesta, predictores)

stopifnot(all(variables %in% names(Cars93)))

datos_originales <- Cars93[, variables, drop = FALSE]

datos <- na.omit(datos_originales)

cat("\n========================================\n")
cat("1. AUDITORIA DE LOS DATOS CARS93\n")
cat("========================================\n")

cat("Observaciones originales:", nrow(Cars93), "\n")
cat("Variables seleccionadas:", length(variables), "\n")
cat("Observaciones completas:", nrow(datos), "\n")
cat("Observaciones excluidas:",
    nrow(datos_originales) - nrow(datos), "\n")
cat("Valores faltantes en las variables seleccionadas:",
    sum(is.na(datos_originales)), "\n")

cat("\nResumen estadistico de los datos:\n")
print(summary(datos))

cat("\nPrimeras seis observaciones:\n")
print(head(datos))

stopifnot(
  nrow(datos) > 0,
  all(is.finite(as.matrix(datos)))
)


# ============================================================
# 2. ANALISIS EXPLORATORIO Y CORRELACIONES
# ============================================================

cat("\n========================================\n")
cat("2. CORRELACIONES ENTRE PREDICTORES\n")
cat("========================================\n")

matriz_correlaciones <- cor(
  datos[, predictores, drop = FALSE]
)

print(round(matriz_correlaciones, 3))

cat("\nCorrelacion de cada predictor con MPG.city:\n")

correlaciones_respuesta <- sapply(
  predictores,
  function(variable) {
    cor(datos[[respuesta]], datos[[variable]])
  }
)

print(sort(correlaciones_respuesta, decreasing = TRUE))

heatmap(
  matriz_correlaciones,
  symm = TRUE,
  main = "Cars93: correlaciones entre predictores"
)


# ============================================================
# 3. PARTICION DE ENTRENAMIENTO Y PRUEBA: 80/20
# ============================================================

cat("\n========================================\n")
cat("3. PARTICION DE LOS DATOS\n")
cat("========================================\n")

set.seed(20261020)

n <- nrow(datos)

indices_train <- sample(
  seq_len(n),
  size = floor(0.8 * n)
)

train <- datos[indices_train, , drop = FALSE]
test <- datos[-indices_train, , drop = FALSE]

cat("Observaciones de entrenamiento:", nrow(train), "\n")
cat("Observaciones de prueba:", nrow(test), "\n")

formula_completa <- reformulate(
  predictores,
  response = respuesta
)


# ============================================================
# 4. REGRESION LINEAL MULTIPLE POR MCO
# ============================================================

cat("\n========================================\n")
cat("4. REGRESION LINEAL MULTIPLE: MCO\n")
cat("========================================\n")

modelo_mco <- lm(
  formula_completa,
  data = train
)

print(summary(modelo_mco))

cat("\nIntervalos de confianza de los coeficientes:\n")
print(confint(modelo_mco))

cat("\nAnalisis de varianza:\n")
print(anova(modelo_mco))

cat("\nDiagnosticos graficos del modelo MCO:\n")

par(mfrow = c(2, 2))
plot(modelo_mco)
par(mfrow = c(1, 1))


# ============================================================
# 5. ANALISIS DE MULTICOLINEALIDAD MEDIANTE VIF
# ============================================================

cat("\n========================================\n")
cat("5. FACTORES DE INFLACION DE VARIANZA (VIF)\n")
cat("========================================\n")

vif_inicial <- car::vif(modelo_mco)

cat("\nVIF del modelo MCO:\n")
print(sort(vif_inicial, decreasing = TRUE))

cat("\nVariables con VIF superior a 5:\n")
print(vif_inicial[vif_inicial > 5])

# Eliminacion iterativa del predictor con mayor VIF.
# Se utiliza unicamente el conjunto de entrenamiento.

variables_vif <- predictores

historial_vif <- data.frame(
  iteracion = integer(),
  variable_eliminada = character(),
  VIF_maximo = numeric()
)

iteracion <- 1

while (length(variables_vif) >= 2) {
  
  formula_vif <- reformulate(
    variables_vif,
    response = respuesta
  )
  
  modelo_temporal <- lm(
    formula_vif,
    data = train
  )
  
  valores_vif <- car::vif(modelo_temporal)
  
  vif_maximo <- max(valores_vif)
  
  if (vif_maximo <= 5) {
    break
  }
  
  variable_eliminar <- names(
    valores_vif
  )[which.max(valores_vif)]
  
  historial_vif <- rbind(
    historial_vif,
    data.frame(
      iteracion = iteracion,
      variable_eliminada = variable_eliminar,
      VIF_maximo = vif_maximo
    )
  )
  
  variables_vif <- setdiff(
    variables_vif,
    variable_eliminar
  )
  
  iteracion <- iteracion + 1
}

cat("\nHistorial de eliminacion por VIF:\n")
print(historial_vif)

cat("\nVariables conservadas:\n")
print(variables_vif)

modelo_vif <- lm(
  reformulate(
    variables_vif,
    response = respuesta
  ),
  data = train
)

cat("\nResumen del modelo reducido por VIF:\n")
print(summary(modelo_vif))

cat("\nVIF finales:\n")

if (length(variables_vif) >= 2) {
  print(car::vif(modelo_vif))
} else {
  cat("Solo queda un predictor; no se calcula VIF entre predictores.\n")
}


# ============================================================
# 6. SELECCION EXHAUSTIVA DE SUBCONJUNTOS MEDIANTE BIC
# ============================================================

cat("\n========================================\n")
cat("6. SELECCION DE VARIABLES MEDIANTE BIC\n")
cat("========================================\n")

modelo_subconjuntos <- leaps::regsubsets(
  formula_completa,
  data = train,
  nvmax = length(predictores),
  method = "exhaustive"
)

resumen_subconjuntos <- summary(
  modelo_subconjuntos
)

tabla_subconjuntos <- data.frame(
  numero_predictores = seq_along(
    resumen_subconjuntos$bic
  ),
  RSS = resumen_subconjuntos$rss,
  R2_ajustado = resumen_subconjuntos$adjr2,
  Cp = resumen_subconjuntos$cp,
  BIC = resumen_subconjuntos$bic
)

cat("\nResultados de los mejores subconjuntos por tamano:\n")
print(tabla_subconjuntos)

indice_bic <- which.min(
  resumen_subconjuntos$bic
)

variables_bic <- names(
  which(
    resumen_subconjuntos$which[indice_bic, ]
  )
)

variables_bic <- setdiff(
  variables_bic,
  "(Intercept)"
)

cat("\nVariables seleccionadas por BIC:\n")
print(variables_bic)

modelo_bic <- lm(
  reformulate(
    variables_bic,
    response = respuesta
  ),
  data = train
)

cat("\nResumen del modelo seleccionado por BIC:\n")
print(summary(modelo_bic))


# ============================================================
# 7. SELECCION DE VARIABLES CON STEPAIC: EJERCICIOS 5 Y 6
# ============================================================

cat("\n========================================\n")
cat("7. SELECCION STEPWISE CON AIC Y BIC\n")
cat("========================================\n")

# Para este analisis se excluye MPG.highway como predictor.

pred_ej56 <- setdiff(
  predictores,
  "MPG.highway"
)

datos_ej56 <- na.omit(
  Cars93[, c(respuesta, pred_ej56), drop = FALSE]
)

formula_ej56 <- reformulate(
  pred_ej56,
  response = respuesta
)

n_ej56 <- nrow(datos_ej56)

modelo_nulo <- lm(
  reformulate(
    character(0),
    response = respuesta
  ),
  data = datos_ej56
)

modelo_completo_step <- lm(
  formula_ej56,
  data = datos_ej56
)

# Seleccion mediante AIC.

modelo_forward_aic <- stepAIC(
  modelo_nulo,
  scope = list(
    lower = formula(modelo_nulo),
    upper = formula(modelo_completo_step)
  ),
  direction = "forward",
  trace = 0
)

modelo_backward_aic <- stepAIC(
  modelo_completo_step,
  direction = "backward",
  trace = 0
)

modelo_both_aic <- stepAIC(
  modelo_nulo,
  scope = list(
    lower = formula(modelo_nulo),
    upper = formula(modelo_completo_step)
  ),
  direction = "both",
  trace = 0
)

# Seleccion mediante BIC: k = log(n).

modelo_forward_bic <- stepAIC(
  modelo_nulo,
  scope = list(
    lower = formula(modelo_nulo),
    upper = formula(modelo_completo_step)
  ),
  direction = "forward",
  k = log(n_ej56),
  trace = 0
)

modelo_backward_bic <- stepAIC(
  modelo_completo_step,
  direction = "backward",
  k = log(n_ej56),
  trace = 0
)

modelo_both_bic <- stepAIC(
  modelo_nulo,
  scope = list(
    lower = formula(modelo_nulo),
    upper = formula(modelo_completo_step)
  ),
  direction = "both",
  k = log(n_ej56),
  trace = 0
)

modelos_seleccion <- list(
  Forward_AIC = modelo_forward_aic,
  Backward_AIC = modelo_backward_aic,
  Both_AIC = modelo_both_aic,
  Forward_BIC = modelo_forward_bic,
  Backward_BIC = modelo_backward_bic,
  Both_BIC = modelo_both_bic
)

tabla_comparacion_seleccion <- do.call(
  rbind,
  lapply(names(modelos_seleccion), function(nombre) {
    
    ajuste <- modelos_seleccion[[nombre]]
    
    data.frame(
      metodo = nombre,
      formula = paste(
        deparse(formula(ajuste)),
        collapse = ""
      ),
      numero_variables = length(
        attr(terms(ajuste), "term.labels")
      ),
      AIC = AIC(ajuste),
      BIC = BIC(ajuste),
      R2_ajustado = summary(ajuste)$adj.r.squared,
      R2 = summary(ajuste)$r.squared
    )
  })
)

cat("\nComparacion de los modelos stepAIC:\n")
print(tabla_comparacion_seleccion, row.names = FALSE)

cat("\nResumen del modelo Forward AIC:\n")
print(summary(modelo_forward_aic))

cat("\nResumen del modelo Backward AIC:\n")
print(summary(modelo_backward_aic))

cat("\nResumen del modelo Both AIC:\n")
print(summary(modelo_both_aic))

cat("\nResumen del modelo Forward BIC:\n")
print(summary(modelo_forward_bic))

cat("\nResumen del modelo Backward BIC:\n")
print(summary(modelo_backward_bic))

cat("\nResumen del modelo Both BIC:\n")
print(summary(modelo_both_bic))


# ============================================================
# 8. REGULARIZACION: RIDGE, LASSO Y ELASTIC NET
# ============================================================

cat("\n========================================\n")
cat("8. RIDGE, LASSO Y ELASTIC NET\n")
cat("========================================\n")

# Se construyen matrices sin intercepto.
# glmnet estima el intercepto por defecto.

x_train <- model.matrix(
  formula_completa,
  data = train
)[, -1, drop = FALSE]

x_test <- model.matrix(
  formula_completa,
  data = test
)[, -1, drop = FALSE]

y_train <- train[[respuesta]]
y_test <- test[[respuesta]]

stopifnot(
  all(is.finite(x_train)),
  all(is.finite(x_test)),
  all(is.finite(y_train)),
  all(is.finite(y_test))
)

# RIDGE: alpha = 0.

set.seed(20261021)

cv_ridge <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0,
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

# LASSO: alpha = 1.

set.seed(20261022)

cv_lasso <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

# ELASTIC NET: alpha = 0.5.

set.seed(20261023)

cv_en <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0.5,
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

cat("\nLambda minimo y lambda de un error estandar:\n")

tabla_lambda <- data.frame(
  metodo = c("Ridge", "Lasso", "Elastic Net"),
  alpha = c(0, 1, 0.5),
  lambda_min = c(
    cv_ridge$lambda.min,
    cv_lasso$lambda.min,
    cv_en$lambda.min
  ),
  lambda_1se = c(
    cv_ridge$lambda.1se,
    cv_lasso$lambda.1se,
    cv_en$lambda.1se
  ),
  MSE_CV_minimo = c(
    min(cv_ridge$cvm),
    min(cv_lasso$cvm),
    min(cv_en$cvm)
  )
)

print(tabla_lambda, row.names = FALSE)

cat("\nCoeficientes Ridge para lambda.min:\n")
print(coef(cv_ridge, s = "lambda.min"))

cat("\nCoeficientes Ridge para lambda.1se:\n")
print(coef(cv_ridge, s = "lambda.1se"))

cat("\nCoeficientes Lasso para lambda.min:\n")
print(coef(cv_lasso, s = "lambda.min"))

cat("\nCoeficientes Lasso para lambda.1se:\n")
print(coef(cv_lasso, s = "lambda.1se"))

cat("\nCoeficientes Elastic Net para lambda.min:\n")
print(coef(cv_en, s = "lambda.min"))

cat("\nCoeficientes Elastic Net para lambda.1se:\n")
print(coef(cv_en, s = "lambda.1se"))

cat("\nGraficos de validacion cruzada:\n")

par(mfrow = c(1, 3))

plot(cv_ridge, main = "Ridge")
plot(cv_lasso, main = "Lasso")
plot(cv_en, main = "Elastic Net")

par(mfrow = c(1, 1))


# ============================================================
# 9. FUNCIONES PARA EVALUAR LOS MODELOS
# ============================================================

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

mae <- function(y, pred) {
  mean(abs(y - pred))
}


# ============================================================
# 10. COMPARACION PREDICTIVA EN EL CONJUNTO DE PRUEBA
# ============================================================

cat("\n========================================\n")
cat("9. EVALUACION EN EL CONJUNTO DE PRUEBA\n")
cat("========================================\n")

predicciones <- list(
  MCO = predict(
    modelo_mco,
    newdata = test
  ),
  MCO_VIF = predict(
    modelo_vif,
    newdata = test
  ),
  MCO_BIC = predict(
    modelo_bic,
    newdata = test
  ),
  Ridge = as.numeric(
    predict(
      cv_ridge,
      newx = x_test,
      s = "lambda.1se"
    )
  ),
  Lasso = as.numeric(
    predict(
      cv_lasso,
      newx = x_test,
      s = "lambda.1se"
    )
  ),
  ElasticNet = as.numeric(
    predict(
      cv_en,
      newx = x_test,
      s = "lambda.1se"
    )
  )
)

metricas_test <- do.call(
  rbind,
  lapply(names(predicciones), function(nombre) {
    
    pred <- predicciones[[nombre]]
    
    data.frame(
      modelo = nombre,
      RMSE = rmse(y_test, pred),
      MAE = mae(y_test, pred)
    )
  })
)

metricas_test <- metricas_test[
  order(metricas_test$RMSE),
]

cat("\nComparacion de modelos ordenada por RMSE:\n")
print(metricas_test, row.names = FALSE)

cat("\nGrafico comparativo del RMSE:\n")

barplot(
  metricas_test$RMSE,
  names.arg = metricas_test$modelo,
  las = 2,
  col = c("#4391B0", "#BA3D99", "#339E7C"),
  ylab = "RMSE en prueba",
  main = "Cars93: Comparación del RMSE",
  border = "white"
)

cat("\nGrafico comparativo del MAE:\n")

barplot(
  metricas_test$MAE,
  names.arg = metricas_test$modelo,
  las = 2,
  col = c("#4391B0", "#BA3D99", "#339E7C"),
  ylab = "MAE en prueba",
  main = "Cars93: comparacion del MAE"
)


# ============================================================
# 11. VALIDACION REPETIDA: 100 PARTICIONES ALEATORIAS
# ============================================================

cat("\n========================================\n")
cat("10. VALIDACION REPETIDA: 100 PARTICIONES\n")
cat("========================================\n")

set.seed(20261024)

B <- 100

resultados_repetidos <- vector(
  "list",
  B
)

for (b in seq_len(B)) {
  
  indices <- sample(
    seq_len(nrow(datos)),
    size = floor(0.8 * nrow(datos))
  )
  
  tr <- datos[indices, , drop = FALSE]
  te <- datos[-indices, , drop = FALSE]
  
  formula_actual <- reformulate(
    predictores,
    response = respuesta
  )
  
  # Modelo MCO.
  modelo_ols <- lm(
    formula_actual,
    data = tr
  )
  
  # Matrices para glmnet.
  xt <- model.matrix(
    formula_actual,
    data = tr
  )[, -1, drop = FALSE]
  
  xv <- model.matrix(
    formula_actual,
    data = te
  )[, -1, drop = FALSE]
  
  yt <- tr[[respuesta]]
  yv <- te[[respuesta]]
  
  # Ridge.
  set.seed(20261019 + b)
  
  modelo_ridge <- cv.glmnet(
    xt,
    yt,
    alpha = 0,
    nfolds = 10,
    type.measure = "mse",
    standardize = TRUE
  )
  
  # Lasso.
  set.seed(20262019 + b)
  
  modelo_lasso <- cv.glmnet(
    xt,
    yt,
    alpha = 1,
    nfolds = 10,
    type.measure = "mse",
    standardize = TRUE
  )
  
  # Elastic Net.
  set.seed(20263019 + b)
  
  modelo_elastic <- cv.glmnet(
    xt,
    yt,
    alpha = 0.5,
    nfolds = 10,
    type.measure = "mse",
    standardize = TRUE
  )
  
  # Predicciones sobre la particion de prueba.
  pred <- list(
    MCO = predict(
      modelo_ols,
      newdata = te
    ),
    Ridge = as.numeric(
      predict(
        modelo_ridge,
        newx = xv,
        s = "lambda.1se"
      )
    ),
    Lasso = as.numeric(
      predict(
        modelo_lasso,
        newx = xv,
        s = "lambda.1se"
      )
    ),
    ElasticNet = as.numeric(
      predict(
        modelo_elastic,
        newx = xv,
        s = "lambda.1se"
      )
    )
  )
  
  resultados_repetidos[[b]] <- do.call(
    rbind,
    lapply(names(pred), function(nombre) {
      
      data.frame(
        repeticion = b,
        modelo = nombre,
        RMSE = rmse(yv, pred[[nombre]]),
        MAE = mae(yv, pred[[nombre]])
      )
    })
  )
}

resultados_repetidos <- do.call(
  rbind,
  resultados_repetidos
)

cat("\nPrimeros resultados de las 100 particiones:\n")
print(head(resultados_repetidos, 20))

# Resumen de RMSE.
resumen_rmse <- aggregate(
  RMSE ~ modelo,
  data = resultados_repetidos,
  FUN = function(x) {
    c(media = mean(x), desviacion = sd(x))
  }
)

# Resumen de MAE.
resumen_mae <- aggregate(
  MAE ~ modelo,
  data = resultados_repetidos,
  FUN = function(x) {
    c(media = mean(x), desviacion = sd(x))
  }
)

resumen_100 <- data.frame(
  modelo = resumen_rmse$modelo,
  RMSE_media = vapply(
    resumen_rmse$RMSE,
    function(x) x[1],
    numeric(1)
  ),
  RMSE_desviacion = vapply(
    resumen_rmse$RMSE,
    function(x) x[2],
    numeric(1)
  ),
  MAE_media = vapply(
    resumen_mae$MAE,
    function(x) x[1],
    numeric(1)
  ),
  MAE_desviacion = vapply(
    resumen_mae$MAE,
    function(x) x[2],
    numeric(1)
  )
)

resumen_100 <- resumen_100[
  order(resumen_100$RMSE_media),
]

cat("\n========================================\n")
cat("RESUMEN DE LAS 100 PARTICIONES\n")
cat("========================================\n")

print(resumen_100, row.names = FALSE)

cat("\nGrafico del RMSE medio en las 100 particiones:\n")

barplot(
  resumen_100$RMSE_media,
  names.arg = resumen_100$modelo,
  las = 2,
  col = "#4391B0",
  ylab = "RMSE medio",
  main = "Cars93: RMSE medio en 100 particiones"
)

cat("\nGrafico del MAE medio en las 100 particiones:\n")

barplot(
  resumen_100$MAE_media,
  names.arg = resumen_100$modelo,
  las = 2,
  col = "#4391B0",
  ylab = "MAE medio",
  main = "Cars93: MAE medio en 100 particiones"
)


# ============================================================
# 12. CONCLUSION AUTOMATICA DE LA COMPARACION PREDICTIVA
# ============================================================

cat("\n========================================\n")
cat("11. RESUMEN FINAL\n")
cat("========================================\n")

mejor_test <- metricas_test$modelo[
  which.min(metricas_test$RMSE)
]

mejor_repetido <- resumen_100$modelo[
  which.min(resumen_100$RMSE_media)
]

cat(
  "Mejor modelo en la particion de prueba segun RMSE:",
  mejor_test,
  "\n"
)

cat(
  "Mejor modelo segun el RMSE medio de las 100 particiones:",
  mejor_repetido,
  "\n"
)


cat("================================================================\n")
# ============================================================
# PROYECTO MEA: MODELOS ESTADISTICOS ACTIVOS
# ANALISIS DE BOSTONHOUSING Y HALD
# ============================================================

# ============================================================
# 0. CONFIGURACION Y CARGA DE PAQUETES
# ============================================================

options(stringsAsFactors = FALSE)

paquetes <- c("mctest", "mlbench")

faltantes <- paquetes[
  !vapply(paquetes, requireNamespace, logical(1), quietly = TRUE)
]

if (length(faltantes) > 0) {
  install.packages(faltantes)
}


library(mctest) 
library(mlbench) 

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

mae <- function(y, pred) {
  mean(abs(y - pred))
}


# ============================================================
# PARTE A. BOSTONHOUSING
# EJERCICIO 4: SELECCION DE SUBCONJUNTOS Y CRITERIOS
# ============================================================


# ============================================================
# 1. CARGA Y EXPLORACION DE BOSTONHOUSING
# ============================================================

# Cargar los datos
data("BostonHousing", package = "mlbench")
View(BostonHousing)

variables_bh <- c(
  "medv",
  "indus",
  "rm",
  "ptratio",
  "lstat"
)

# Seleccionar las variables y eliminar filas con valores faltantes
bh <- na.omit(
  BostonHousing[, variables_bh, drop = FALSE]
)

cat("\n========================================\n")
cat("BOSTONHOUSING: AUDITORIA DE LOS DATOS\n")
cat("========================================\n")

cat("Observaciones originales:", nrow(BostonHousing), "\n")
cat("Observaciones utilizadas:", nrow(bh), "\n")
cat("Variables utilizadas:", ncol(bh), "\n")
cat("Valores faltantes restantes:", sum(is.na(bh)), "\n")

cat("\nResumen estadistico:\n")
print(summary(bh))

cat("\nPrimeras seis observaciones:\n")
print(head(bh))

cat("\nMatriz de correlaciones:\n")
print(round(cor(bh), 3))

cat("\nGrafico de correlaciones:\n")

heatmap(
  cor(bh),
  symm = TRUE,
  main = "BostonHousing: matriz de correlaciones"
)


# ============================================================
# 2. MODELO DE REGRESION LINEAL MULTIPLE COMPLETO
# ============================================================

cat("\n========================================\n")
cat("BOSTONHOUSING: MODELO COMPLETO\n")
cat("========================================\n")

modelo_bh_completo <- lm(
  medv ~ indus + rm + ptratio + lstat,
  data = bh
)

print(summary(modelo_bh_completo))

cat("\nIntervalos de confianza:\n")
print(confint(modelo_bh_completo))

cat("\nAnalisis de varianza:\n")
print(anova(modelo_bh_completo))

cat("\nVIF del modelo completo:\n")
print(car::vif(modelo_bh_completo))

cat("\nDiagnosticos graficos del modelo completo:\n")

par(mfrow = c(2, 2))
plot(modelo_bh_completo)
par(mfrow = c(1, 1))


# ============================================================
# 3. SELECCION EXHAUSTIVA DE SUBCONJUNTOS
# ============================================================

cat("\n========================================\n")
cat("BOSTONHOUSING: SELECCION DE SUBCONJUNTOS\n")
cat("========================================\n")

modelo_bh_subconjuntos <- leaps::regsubsets(
  medv ~ indus + rm + ptratio + lstat,
  data = bh,
  nvmax = 4,
  method = "exhaustive"
)

resumen_bh <- summary(modelo_bh_subconjuntos)

tabla_bh <- data.frame(
  numero_predictores = seq_along(resumen_bh$rss),
  RSS = resumen_bh$rss,
  R2 = resumen_bh$rsq,
  R2_ajustado = resumen_bh$adjr2,
  Cp = resumen_bh$cp,
  BIC = resumen_bh$bic
)

cat("\nComparacion de los mejores subconjuntos por tamano:\n")
print(tabla_bh, row.names = FALSE)

cat("\nVariables seleccionadas para cada tamano:\n")

variables_por_tamano_bh <- lapply(
  seq_len(nrow(resumen_bh$which)),
  function(i) {
    nombres <- colnames(resumen_bh$which)
    
    nombres[
      resumen_bh$which[i, ] &
        nombres != "(Intercept)"
    ]
  }
)

for (i in seq_along(variables_por_tamano_bh)) {
  cat("\nModelo con", i, "predictor(es):\n")
  print(variables_por_tamano_bh[[i]])
}


# ============================================================
# 4. COMPARACION DE MODELOS MEDIANTE AIC, BIC, Cp Y R2
# ============================================================

cat("\n========================================\n")
cat("BOSTONHOUSING: COMPARACION DE CRITERIOS\n")
cat("========================================\n")

modelos_bh <- lapply(
  variables_por_tamano_bh,
  function(vars) {
    lm(
      reformulate(vars, response = "medv"),
      data = bh
    )
  }
)

tabla_criterios_bh <- data.frame(
  numero_predictores = seq_along(modelos_bh),
  variables = vapply(
    variables_por_tamano_bh,
    function(vars) paste(vars, collapse = " + "),
    character(1)
  ),
  AIC = vapply(modelos_bh, AIC, numeric(1)),
  BIC = vapply(modelos_bh, BIC, numeric(1)),
  Cp = resumen_bh$cp,
  R2 = resumen_bh$rsq,
  R2_ajustado = resumen_bh$adjr2
)

cat("\nTabla comparativa de modelos:\n")
print(tabla_criterios_bh, row.names = FALSE)

indice_cp_bh <- which.min(tabla_criterios_bh$Cp)
indice_aic_bh <- which.min(tabla_criterios_bh$AIC)
indice_bic_bh <- which.min(tabla_criterios_bh$BIC)

modelo_bh_cp <- modelos_bh[[indice_cp_bh]]
modelo_bh_aic <- modelos_bh[[indice_aic_bh]]
modelo_bh_bic <- modelos_bh[[indice_bic_bh]]

cat("\nMejor modelo segun Cp:\n")
print(formula(modelo_bh_cp))
print(summary(modelo_bh_cp))

cat("\nMejor modelo segun AIC:\n")
print(formula(modelo_bh_aic))
print(summary(modelo_bh_aic))

cat("\nMejor modelo segun BIC:\n")
print(formula(modelo_bh_bic))
print(summary(modelo_bh_bic))


# ============================================================
# 5. SELECCION STEPWISE PARA BOSTONHOUSING
# ============================================================

cat("\n========================================\n")
cat("BOSTONHOUSING: SELECCION STEPWISE\n")
cat("========================================\n")

modelo_bh_nulo <- lm(
  medv ~ 1,
  data = bh
)

modelo_bh_completo_step <- lm(
  medv ~ indus + rm + ptratio + lstat,
  data = bh
)

modelo_bh_forward <- stepAIC(
  modelo_bh_nulo,
  scope = list(
    lower = formula(modelo_bh_nulo),
    upper = formula(modelo_bh_completo_step)
  ),
  direction = "forward",
  trace = 0
)

modelo_bh_backward <- stepAIC(
  modelo_bh_completo_step,
  direction = "backward",
  trace = 0
)

modelo_bh_both <- stepAIC(
  modelo_bh_nulo,
  scope = list(
    lower = formula(modelo_bh_nulo),
    upper = formula(modelo_bh_completo_step)
  ),
  direction = "both",
  trace = 0
)

modelos_step_bh <- list(
  Forward = modelo_bh_forward,
  Backward = modelo_bh_backward,
  Both = modelo_bh_both
)

tabla_step_bh <- do.call(
  rbind,
  lapply(names(modelos_step_bh), function(nombre) {
    
    ajuste <- modelos_step_bh[[nombre]]
    
    data.frame(
      metodo = nombre,
      formula = paste(
        deparse(formula(ajuste)),
        collapse = ""
      ),
      numero_variables = length(
        attr(terms(ajuste), "term.labels")
      ),
      AIC = AIC(ajuste),
      BIC = BIC(ajuste),
      R2 = summary(ajuste)$r.squared,
      R2_ajustado = summary(ajuste)$adj.r.squared
    )
  })
)

cat("\nComparacion stepwise:\n")
print(tabla_step_bh, row.names = FALSE)

for (nombre in names(modelos_step_bh)) {
  cat("\nResumen del modelo", nombre, ":\n")
  print(summary(modelos_step_bh[[nombre]]))
}


# ============================================================
# PARTE B. HALD
# EJERCICIOS 7 Y 8: SUBCONJUNTOS Y MULTICOLINEALIDAD
# ============================================================


# ============================================================
# 6. CARGA Y EXPLORACION DE HALD
# ============================================================

data("Hald", package = "mctest")

hald <- as.data.frame(mctest::Hald)

stopifnot(
  all(c("y", "X1", "X2", "X3", "X4") %in% names(hald))
)

hald <- hald[, c("y", "X1", "X2", "X3", "X4")]

hald <- na.omit(hald)

cat("\n========================================\n")
cat("HALD: AUDITORIA DE LOS DATOS\n")
cat("========================================\n")

cat("Numero de observaciones:", nrow(hald), "\n")
cat("Numero de variables:", ncol(hald), "\n")
cat("Valores faltantes:", sum(is.na(hald)), "\n")

cat("\nDatos completos:\n")
print(hald)

View(hald)

cat("\nResumen estadistico:\n")
print(summary(hald))

cat("\nMatriz de correlaciones:\n")
print(round(cor(hald), 3))

cat("\nCorrelaciones entre los predictores:\n")
print(round(cor(hald[, c("X1", "X2", "X3", "X4")]), 3))

cat("\nGrafico de correlaciones entre predictores:\n")

heatmap(
  cor(hald[, c("X1", "X2", "X3", "X4")]),
  symm = TRUE,
  main = "Hald: correlaciones entre predictores"
)


# ============================================================
# 7. MODELO COMPLETO DE HALD
# ============================================================

cat("\n========================================\n")
cat("HALD: MODELO DE REGRESION COMPLETO\n")
cat("========================================\n")

modelo_hald_completo <- lm(
  y ~ X1 + X2 + X3 + X4,
  data = hald
)

print(summary(modelo_hald_completo))

cat("\nIntervalos de confianza:\n")
print(confint(modelo_hald_completo))

cat("\nAnalisis de varianza:\n")
print(anova(modelo_hald_completo))

cat("\nDiagnosticos graficos del modelo completo:\n")

par(mfrow = c(2, 2))
plot(modelo_hald_completo)
par(mfrow = c(1, 1))


# ============================================================
# 8. SELECCION EXHAUSTIVA DE SUBCONJUNTOS PARA HALD
# ============================================================

cat("\n========================================\n")
cat("HALD: SELECCION DE SUBCONJUNTOS\n")
cat("========================================\n")

modelo_hald_subconjuntos <- leaps::regsubsets(
  y ~ X1 + X2 + X3 + X4,
  data = hald,
  nvmax = 4,
  method = "exhaustive"
)

resumen_hald <- summary(modelo_hald_subconjuntos)

tabla_hald <- data.frame(
  numero_predictores = seq_along(resumen_hald$rss),
  RSS = resumen_hald$rss,
  R2 = resumen_hald$rsq,
  R2_ajustado = resumen_hald$adjr2,
  Cp = resumen_hald$cp,
  BIC = resumen_hald$bic
)

cat("\nResultados por tamano de subconjunto:\n")
print(tabla_hald, row.names = FALSE)

variables_por_tamano_hald <- lapply(
  seq_len(nrow(resumen_hald$which)),
  function(i) {
    nombres <- colnames(resumen_hald$which)
    
    nombres[
      resumen_hald$which[i, ] &
        nombres != "(Intercept)"
    ]
  }
)

cat("\nVariables seleccionadas por tamano:\n")

for (i in seq_along(variables_por_tamano_hald)) {
  cat("\nModelo con", i, "predictor(es):\n")
  print(variables_por_tamano_hald[[i]])
}


# ============================================================
# 9. COMPARACION DE CRITERIOS PARA HALD
# ============================================================

cat("\n========================================\n")
cat("HALD: COMPARACION DE CRITERIOS\n")
cat("========================================\n")

modelos_hald <- lapply(
  variables_por_tamano_hald,
  function(vars) {
    lm(
      reformulate(vars, response = "y"),
      data = hald
    )
  }
)

tabla_criterios_hald <- data.frame(
  numero_predictores = seq_along(modelos_hald),
  variables = vapply(
    variables_por_tamano_hald,
    function(vars) paste(vars, collapse = " + "),
    character(1)
  ),
  AIC = vapply(modelos_hald, AIC, numeric(1)),
  BIC = vapply(modelos_hald, BIC, numeric(1)),
  Cp = resumen_hald$cp,
  R2 = resumen_hald$rsq,
  R2_ajustado = resumen_hald$adjr2
)

cat("\nTabla comparativa:\n")
print(tabla_criterios_hald, row.names = FALSE)

indice_r2_hald <- which.max(
  tabla_criterios_hald$R2_ajustado
)

indice_cp_hald <- which.min(
  tabla_criterios_hald$Cp
)

indice_bic_hald <- which.min(
  tabla_criterios_hald$BIC
)

modelo_hald_r2 <- modelos_hald[[indice_r2_hald]]
modelo_hald_cp <- modelos_hald[[indice_cp_hald]]
modelo_hald_bic <- modelos_hald[[indice_bic_hald]]

cat("\nMejor modelo segun R2 ajustado:\n")
print(formula(modelo_hald_r2))
print(summary(modelo_hald_r2))

cat("\nMejor modelo segun Cp:\n")
print(formula(modelo_hald_cp))
print(summary(modelo_hald_cp))

cat("\nMejor modelo segun BIC:\n")
print(formula(modelo_hald_bic))
print(summary(modelo_hald_bic))


# ============================================================
# 10. SELECCION STEPWISE PARA HALD
# ============================================================

cat("\n========================================\n")
cat("HALD: SELECCION STEPWISE\n")
cat("========================================\n")

modelo_hald_nulo <- lm(
  y ~ 1,
  data = hald
)

modelo_hald_completo_step <- lm(
  y ~ X1 + X2 + X3 + X4,
  data = hald
)

modelo_hald_forward <- step(
  modelo_hald_nulo,
  scope = list(
    lower = formula(modelo_hald_nulo),
    upper = formula(modelo_hald_completo_step)
  ),
  direction = "forward",
  trace = 0
)

modelo_hald_backward <- step(
  modelo_hald_completo_step,
  direction = "backward",
  trace = 0
)

modelo_hald_both <- step(
  modelo_hald_nulo,
  scope = list(
    lower = formula(modelo_hald_nulo),
    upper = formula(modelo_hald_completo_step)
  ),
  direction = "both",
  trace = 0
)

modelos_step_hald <- list(
  Forward = modelo_hald_forward,
  Backward = modelo_hald_backward,
  Stepwise = modelo_hald_both
)

tabla_step_hald <- do.call(
  rbind,
  lapply(names(modelos_step_hald), function(nombre) {
    
    ajuste <- modelos_step_hald[[nombre]]
    
    data.frame(
      metodo = nombre,
      formula = paste(
        deparse(formula(ajuste)),
        collapse = ""
      ),
      numero_variables = length(
        attr(terms(ajuste), "term.labels")
      ),
      AIC = AIC(ajuste),
      BIC = BIC(ajuste),
      R2_ajustado = summary(ajuste)$adj.r.squared
    )
  })
)

cat("\nComparacion de los modelos stepwise:\n")
print(tabla_step_hald, row.names = FALSE)

for (nombre in names(modelos_step_hald)) {
  cat("\nResumen del modelo", nombre, ":\n")
  print(summary(modelos_step_hald[[nombre]]))
}


# ============================================================
# 11. MULTICOLINEALIDAD EN HALD: VIF
# ============================================================

cat("\n========================================\n")
cat("HALD: ANALISIS DE MULTICOLINEALIDAD\n")
cat("========================================\n")

cat("\nVIF del modelo completo:\n")
print(car::vif(modelo_hald_completo))

cat("\nVIF del modelo seleccionado por R2 ajustado:\n")

if (length(attr(terms(modelo_hald_r2), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_r2))
} else {
  cat("El modelo tiene menos de dos predictores.\n")
}

cat("\nVIF del modelo seleccionado por Cp:\n")

if (length(attr(terms(modelo_hald_cp), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_cp))
} else {
  cat("El modelo tiene menos de dos predictores.\n")
}

cat("\nVIF del modelo seleccionado por BIC:\n")

if (length(attr(terms(modelo_hald_bic), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_bic))
} else {
  cat("El modelo tiene menos de dos predictores.\n")
}


# ============================================================
# 12. INDICES DE CONDICION PARA HALD
# ============================================================

cat("\n========================================\n")
cat("HALD: INDICES DE CONDICION\n")
cat("========================================\n")

X_hald <- as.matrix(
  hald[, c("X1", "X2", "X3", "X4")]
)

# Estandarizar los predictores para comparar sus escalas.

X_hald_estandarizada <- scale(
  X_hald,
  center = TRUE,
  scale = TRUE
)

valores_propios <- eigen(
  crossprod(X_hald_estandarizada),
  symmetric = TRUE,
  only.values = TRUE
)$values

valores_propios <- pmax(
  valores_propios,
  .Machine$double.eps
)

indices_condicion <- sqrt(
  max(valores_propios) / valores_propios
)

tabla_condicion <- data.frame(
  dimension = seq_along(valores_propios),
  valor_propio = valores_propios,
  indice_condicion = indices_condicion
)

cat("\nValores propios e indices de condicion:\n")
print(tabla_condicion, row.names = FALSE)

cat(
  "\nIndice de condicion maximo:",
  max(indices_condicion),
  "\n"
)

# Numero de condicion de la matriz de diseno original con intercepto.

matriz_diseno_hald <- model.matrix(
  y ~ X1 + X2 + X3 + X4,
  data = hald
)

valores_propios_originales <- eigen(
  crossprod(matriz_diseno_hald),
  symmetric = TRUE,
  only.values = TRUE
)$values

numero_condicion_original <- sqrt(
  max(valores_propios_originales) /
    min(valores_propios_originales)
)

cat(
  "Numero de condicion de la matriz de diseno original:",
  format(numero_condicion_original, scientific = TRUE),
  "\n"
)


# ============================================================
# PARTE C. HALD
# EJERCICIO 9: RIDGE, LASSO Y ELASTIC NET CON LOOCV
# ============================================================


# ============================================================
# 13. PREPARACION PARA REGULARIZACION
# ============================================================

cat("\n========================================\n")
cat("HALD: REGULARIZACION CON LOOCV\n")
cat("========================================\n")

X <- as.matrix(
  hald[, c("X1", "X2", "X3", "X4")]
)

y <- hald$y

stopifnot(
  nrow(X) == length(y),
  all(is.finite(X)),
  all(is.finite(y))
)

n_hald <- nrow(X)

cat("Numero de observaciones:", n_hald, "\n")
cat("Numero de predictores:", ncol(X), "\n")

cat("\nSe utilizara validacion cruzada leave-one-out (LOOCV).\n")


# ============================================================
# 14. RIDGE CON LOOCV
# ============================================================

set.seed(20261022)

cv_ridge_hald <- cv.glmnet(
  X,
  y,
  alpha = 0,
  family = "gaussian",
  type.measure = "mse",
  nfolds = n_hald,
  standardize = TRUE
)

cat("\n========================================\n")
cat("HALD: RIDGE\n")
cat("========================================\n")

cat("Lambda minimo:", cv_ridge_hald$lambda.min, "\n")
cat("Lambda 1se:", cv_ridge_hald$lambda.1se, "\n")
cat("MSE minimo en LOOCV:", min(cv_ridge_hald$cvm), "\n")

cat("\nCoeficientes Ridge para lambda.min:\n")
print(coef(cv_ridge_hald, s = "lambda.min"))

cat("\nCoeficientes Ridge para lambda.1se:\n")
print(coef(cv_ridge_hald, s = "lambda.1se"))

plot(
  cv_ridge_hald,
  main = "Hald: Ridge con LOOCV"
)


# ============================================================
# 15. LASSO CON LOOCV
# ============================================================

set.seed(20261022)

cv_lasso_hald <- cv.glmnet(
  X,
  y,
  alpha = 1,
  family = "gaussian",
  type.measure = "mse",
  nfolds = n_hald,
  standardize = TRUE
)

cat("\n========================================\n")
cat("HALD: LASSO\n")
cat("========================================\n")

cat("Lambda minimo:", cv_lasso_hald$lambda.min, "\n")
cat("Lambda 1se:", cv_lasso_hald$lambda.1se, "\n")
cat("MSE minimo en LOOCV:", min(cv_lasso_hald$cvm), "\n")

cat("\nCoeficientes Lasso para lambda.min:\n")
print(coef(cv_lasso_hald, s = "lambda.min"))

cat("\nCoeficientes Lasso para lambda.1se:\n")
print(coef(cv_lasso_hald, s = "lambda.1se"))

plot(
  cv_lasso_hald,
  main = "Hald: Lasso con LOOCV"
)


# ============================================================
# 16. ELASTIC NET CON LOOCV
# ============================================================

set.seed(20261022)

cv_en_hald <- cv.glmnet(
  X,
  y,
  alpha = 0.5,
  family = "gaussian",
  type.measure = "mse",
  nfolds = n_hald,
  standardize = TRUE
)

cat("\n========================================\n")
cat("HALD: ELASTIC NET\n")
cat("========================================\n")

cat("Lambda minimo:", cv_en_hald$lambda.min, "\n")
cat("Lambda 1se:", cv_en_hald$lambda.1se, "\n")
cat("MSE minimo en LOOCV:", min(cv_en_hald$cvm), "\n")

cat("\nCoeficientes Elastic Net para lambda.min:\n")
print(coef(cv_en_hald, s = "lambda.min"))

cat("\nCoeficientes Elastic Net para lambda.1se:\n")
print(coef(cv_en_hald, s = "lambda.1se"))

plot(
  cv_en_hald,
  main = "Hald: Elastic Net con LOOCV"
)


# ============================================================
# 17. COMPARACION DE LAMBDAS Y ERROR DE VALIDACION
# ============================================================

cat("\n========================================\n")
cat("HALD: COMPARACION DE REGULARIZACION\n")
cat("========================================\n")

tabla_regularizacion_hald <- data.frame(
  metodo = c("Ridge", "Lasso", "Elastic Net"),
  alpha = c(0, 1, 0.5),
  lambda_min = c(
    cv_ridge_hald$lambda.min,
    cv_lasso_hald$lambda.min,
    cv_en_hald$lambda.min
  ),
  lambda_1se = c(
    cv_ridge_hald$lambda.1se,
    cv_lasso_hald$lambda.1se,
    cv_en_hald$lambda.1se
  ),
  MSE_CV_minimo = c(
    min(cv_ridge_hald$cvm),
    min(cv_lasso_hald$cvm),
    min(cv_en_hald$cvm)
  )
)

cat("\nTabla de comparacion:\n")
print(tabla_regularizacion_hald, row.names = FALSE)

cat("\nRMSE LOOCV estimado para cada metodo:\n")

tabla_regularizacion_hald$RMSE_CV <- sqrt(
  tabla_regularizacion_hald$MSE_CV_minimo
)

print(
  tabla_regularizacion_hald[
    order(tabla_regularizacion_hald$RMSE_CV),
  ],
  row.names = FALSE
)


# ============================================================
# 18. COMPARACION DE COEFICIENTES
# ============================================================

cat("\n========================================\n")
cat("HALD: COMPARACION DE COEFICIENTES\n")
cat("========================================\n")

coeficientes_ridge <- as.matrix(
  coef(cv_ridge_hald, s = "lambda.min")
)

coeficientes_lasso <- as.matrix(
  coef(cv_lasso_hald, s = "lambda.min")
)

coeficientes_en <- as.matrix(
  coef(cv_en_hald, s = "lambda.min")
)

tabla_coeficientes_hald <- data.frame(
  termino = rownames(coeficientes_ridge),
  Ridge = as.numeric(coeficientes_ridge[, 1]),
  Lasso = as.numeric(coeficientes_lasso[, 1]),
  ElasticNet = as.numeric(coeficientes_en[, 1])
)

cat("\nCoeficientes para lambda.min:\n")
print(tabla_coeficientes_hald, row.names = FALSE)

cat("\nNumero de coeficientes distintos de cero, sin intercepto:\n")

esparsidad_hald <- data.frame(
  metodo = c("Ridge", "Lasso", "Elastic Net"),
  coeficientes_no_cero = c(
    sum(abs(coeficientes_ridge[-1, 1]) > 1e-8),
    sum(abs(coeficientes_lasso[-1, 1]) > 1e-8),
    sum(abs(coeficientes_en[-1, 1]) > 1e-8)
  )
)

print(esparsidad_hald, row.names = FALSE)


# ============================================================
# 19. TRAYECTORIAS DE COEFICIENTES
# ============================================================

cat("\n========================================\n")
cat("HALD: TRAYECTORIAS DE COEFICIENTES\n")
cat("========================================\n")

par(mfrow = c(1, 3))

plot(
  cv_ridge_hald$glmnet.fit,
  xvar = "lambda",
  main = "Ridge: trayectorias"
)

plot(
  cv_lasso_hald$glmnet.fit,
  xvar = "lambda",
  main = "Lasso: trayectorias"
)

plot(
  cv_en_hald$glmnet.fit,
  xvar = "lambda",
  main = "Elastic Net: trayectorias"
)

par(mfrow = c(1, 1))


# ============================================================
# 20. RESUMEN FINAL
# ============================================================

cat("\n========================================\n")
cat("RESUMEN FINAL DEL ANALISIS\n")
cat("========================================\n")

cat("\nBOSTONHOUSING:\n")

cat("Modelo seleccionado por Cp:\n")
print(formula(modelo_bh_cp))

cat("Modelo seleccionado por AIC:\n")
print(formula(modelo_bh_aic))

cat("Modelo seleccionado por BIC:\n")
print(formula(modelo_bh_bic))

cat("\nHALD:\n")

cat("Modelo seleccionado por R2 ajustado:\n")
print(formula(modelo_hald_r2))

cat("Modelo seleccionado por Cp:\n")
print(formula(modelo_hald_cp))

cat("Modelo seleccionado por BIC:\n")
print(formula(modelo_hald_bic))

cat("\nMejor metodo de regularizacion en Hald segun MSE LOOCV:\n")

mejor_regularizacion_hald <- tabla_regularizacion_hald$metodo[
  which.min(tabla_regularizacion_hald$MSE_CV_minimo)
]

print(mejor_regularizacion_hald)



