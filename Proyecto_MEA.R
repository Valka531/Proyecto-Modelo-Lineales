# ============================================================
# lABORATORIO DE MODELOS LINEALES
# Modelos: MCO, VIF, BIC, Ridge, Lasso y Elastic Net
# ============================================================

# ============================================================
# 1. CONFIGURACION Y CARGA DE DATOS
# ============================================================

options(stringsAsFactors = FALSE)

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


# ========================================
#     AUDITORIA DE LOS DATOS CARS93 
# ========================================

# Observaciones originales
nrow(Cars93)
# Variables seleccionadas
length(variables)
# Observaciones completas
nrow(datos)
# Observaciones excluidas
nrow(datos_originales) - nrow(datos)
# Valores faltantes en las variables seleccionada
sum(is.na(datos_originales))

# Resumen estadísticos de los datos
print(summary(datos))

# Primeras seis observaciones
print(head(datos))

stopifnot( nrow(datos) > 0, all(is.finite(as.matrix(datos))))


# ============================================================
# 2. ANALISIS EXPLORATORIO Y CORRELACIONES
# ============================================================
matriz_correlaciones <- cor(datos[, predictores, drop = FALSE])

print(round(matriz_correlaciones, 3))

# Correlacion de cada predictor con MPG.city

correlaciones_respuesta <- sapply(predictores, 
                                  function(variable) {
                                    cor(datos[[respuesta]], datos[[variable]])
                                    }
                                  )

print(sort(correlaciones_respuesta, decreasing = TRUE))

heatmap(matriz_correlaciones,
  symm = TRUE,
  main = "Cars93: correlaciones entre predictores"
)


# ============================================================
# 3. PARTICION DE ENTRENAMIENTO Y PRUEBA: 80/20
# ============================================================

set.seed(20261020)

n <- nrow(datos)

indices_train <- sample(seq_len(n), size = floor(0.8 * n))

train <- datos[indices_train, , drop = FALSE]
test <- datos[-indices_train, , drop = FALSE]

# Observaciones de entrenamiento
nrow(train)
# Observaciones de prueba
nrow(test)

formula_completa <- reformulate(
  predictores,
  response = respuesta
)


# ============================================================
# 4. REGRESION LINEAL MULTIPLE POR MCO
# ============================================================

modelo_mco <- lm(formula_completa, data = train)

print(summary(modelo_mco))

# IC de los coeficientes
print(confint(modelo_mco))
# Analisis de varianza
print(anova(modelo_mco))

# ============================================================
# 5. ANALISIS DE MULTICOLINEALIDAD MEDIANTE VIF
# ============================================================

vif_inicial <- car::vif(modelo_mco)

#VIF del modelo MCO
print(sort(vif_inicial, decreasing = TRUE))

#Variables con VIF superior a 5
print(vif_inicial[vif_inicial > 5])

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

# Historial de eliminacion por VIF
print(historial_vif)
# Variables conservadas
print(variables_vif)

modelo_vif <- lm(reformulate(variables_vif, response = respuesta), data = train)

# Resumen del modelo reducido por VIF
print(summary(modelo_vif))
# VIF finales
if (length(variables_vif) >= 2) {
  print(car::vif(modelo_vif))
} else {
  print("Solo queda un predictor; no se calcula VIF entre predictores.")
}

# ============================================================
# 6. SELECCION EXHAUSTIVA DE SUBCONJUNTOS MEDIANTE BIC
# ============================================================

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

# Resultados de los mejores subconjuntos por tamano
print(tabla_subconjuntos)

indice_bic <- which.min(resumen_subconjuntos$bic)

variables_bic <- names(which(resumen_subconjuntos$which[indice_bic, ]))

variables_bic <- setdiff(
  variables_bic,
  "(Intercept)"
)

# Variables seleccionadas por BIC:
print(variables_bic)

modelo_bic <- lm(reformulate(variables_bic, response = respuesta), data = train)

# Resumen del modelo seleccionado por BIC:
print(summary(modelo_bic))


# ============================================================
# 7. SELECCION DE VARIABLES CON STEPWISE CON AIC Y BIC: EJERCICIOS 5 Y 6
# ============================================================

# Para este analisis se excluye MPG.highway como predictor.
pred_ej56 <- setdiff(predictores,"MPG.highway")

datos_ej56 <- na.omit(Cars93[, c(respuesta, pred_ej56), drop = FALSE])

formula_ej56 <- reformulate(pred_ej56, response = respuesta)

n_ej56 <- nrow(datos_ej56)

modelo_nulo <- lm(reformulate(character(0), response = respuesta), data = datos_ej56)

modelo_completo_step <- lm(formula_ej56, data = datos_ej56)

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

# Comparacion de los modelos stepAIC:
print(tabla_comparacion_seleccion, row.names = FALSE)
# Resumen del modelo Forward AIC
summary(modelo_forward_aic)
# Resumen del modelo Backward AIC:
summary(modelo_backward_aic)
#Resumen del modelo Both AIC:
summary(modelo_both_aic)
# Resumen del modelo Forward BIC:
summary(modelo_forward_bic)
# Resumen del modelo Backward BIC
summary(modelo_backward_bic)
# Resumen del modelo Both BIC:
summary(modelo_both_bic)


# ============================================================
# 8. REGULARIZACION: RIDGE, LASSO Y ELASTIC NET
# ============================================================

# Se construyen matrices sin intercepto.glmnet estima el intercepto por defecto.

x_train <- model.matrix(formula_completa, data = train)[, -1, drop = FALSE]
x_test <- model.matrix(formula_completa, data = test)[, -1, drop = FALSE]

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

# Lambda minimo y lambda de un error estandar:

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

# Coeficientes Ridge para lambda.min:
print(coef(cv_ridge, s = "lambda.min"))
# Coeficientes Ridge para lambda.1se:
print(coef(cv_ridge, s = "lambda.1se"))
# Coeficientes Lasso para lambda.min:
print(coef(cv_lasso, s = "lambda.min"))
# Coeficientes Lasso para lambda.1se:
print(coef(cv_lasso, s = "lambda.1se"))
# Coeficientes Elastic Net para lambda.min:
print(coef(cv_en, s = "lambda.min"))
# Coeficientes Elastic Net para lambda.1se:
print(coef(cv_en, s = "lambda.1se"))

# Graficos de validacion cruzada:

par(mfrow = c(1, 3))
plot(cv_ridge, main = "Ridge")
plot(cv_lasso, main = "Lasso")
plot(cv_en, main = "Elastic Net")
par(mfrow = c(1, 1))


# ============================================================
# 9. FUNCIONES PARA EVALUAR LOS MODELOS
# ============================================================

rmse <- function(y, pred) {sqrt(mean((y - pred)^2))}
mae <- function(y, pred) {mean(abs(y - pred))}


# ============================================================
# 10. COMPARACION PREDICTIVA EN EL CONJUNTO DE PRUEBA
# ============================================================

# 9. Evaluación del conjunto de prueba

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

# Comparacion de modelos ordenada por RMSE:
print(metricas_test, row.names = FALSE)

# ============================================================
# 11. VALIDACION REPETIDA: 100 PARTICIONES ALEATORIAS
# ============================================================

set.seed(20261024)

B <- 100

resultados_repetidos <- vector("list", B)

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

# Primeros resultados de las 100 particiones:
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

# RESUMEN DE LAS 100 PARTICIONES
print(resumen_100, row.names = FALSE)

# ============================================================
# 12. CONCLUSION AUTOMATICA DE LA COMPARACION PREDICTIVA
# ============================================================

mejor_test <- metricas_test$modelo[which.min(metricas_test$RMSE)]
mejor_repetido <- resumen_100$modelo[which.min(resumen_100$RMSE_media)]

# Mejor modelo en la particion de prueba segun RMSE
mejor_test
# Mejor modelo segun el RMSE medio de las 100 particiones:
mejor_repetido

# ============================================================
# ANALISIS DE BOSTONHOUSING Y HALD
# ============================================================

# ============================================================
# 0. CONFIGURACION Y CARGA DE PAQUETES
# ============================================================

library(mctest) 
library(mlbench) 

rmse <- function(y, pred) {sqrt(mean((y - pred)^2))}
mae <- function(y, pred) {mean(abs(y - pred))}


# ============================================================
# PARTE A. BOSTONHOUSING
# EJERCICIO 4: SELECCION DE SUBCONJUNTOS Y CRITERIOS
# ============================================================

# 1. CARGA Y EXPLORACION DE BOSTONHOUSING

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
bh <- na.omit(BostonHousing[, variables_bh, drop = FALSE])

# BOSTONHOUSING: AUDITORIA DE LOS DATOS

# Observaciones originales
nrow(BostonHousing)
# Observaciones utilizadas:
nrow(bh)
# Variables utilizadas:
ncol(bh)
# Valores faltantes restantes:
sum(is.na(bh))

# Resumen estadistico:
summary(bh)
# Matriz de correlaciones:
print(round(cor(bh), 3))

# Grafico de correlaciones:

heatmap(
  cor(bh),
  symm = TRUE,
  main = "BostonHousing: matriz de correlaciones"
)

# ============================================================
# 2. MODELO DE REGRESION LINEAL MULTIPLE COMPLETO
# ============================================================

# BOSTONHOUSING: MODELO COMPLETO

modelo_bh_completo <- lm(medv ~ indus + rm + ptratio + lstat, data = bh)
summary(modelo_bh_completo)

# IC
confint(modelo_bh_completo)
# Analisis de varianza:
anova(modelo_bh_completo)
# VIF del modelo completo:
print(car::vif(modelo_bh_completo))

# ============================================================
# 3. SELECCION EXHAUSTIVA DE SUBCONJUNTOS
# ============================================================

# BOSTONHOUSING: SELECCION DE SUBCONJUNTOS
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

# Comparacion de los mejores subconjuntos por tamano:
print(tabla_bh, row.names = FALSE)
# Variables seleccionadas para cada tamano:
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

# OSTONHOUSING: COMPARACION DE CRITERIOS

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

# Tabla comparativa de modelos:
print(tabla_criterios_bh, row.names = FALSE)

indice_cp_bh <- which.min(tabla_criterios_bh$Cp)
indice_aic_bh <- which.min(tabla_criterios_bh$AIC)
indice_bic_bh <- which.min(tabla_criterios_bh$BIC)

modelo_bh_cp <- modelos_bh[[indice_cp_bh]]
modelo_bh_aic <- modelos_bh[[indice_aic_bh]]
modelo_bh_bic <- modelos_bh[[indice_bic_bh]]

# Mejor modelo segun Cp:
formula(modelo_bh_cp)
summary(modelo_bh_cp)

# Mejor modelo segun AIC:
formula(modelo_bh_aic)
summary(modelo_bh_aic)

# nMejor modelo segun BIC:
formula(modelo_bh_bic)
summary(modelo_bh_bic)

# ============================================================
# 5. SELECCION STEPWISE PARA BOSTONHOUSING
# ============================================================

# BOSTONHOUSING: SELECCION STEPWISE\n")

modelo_bh_nulo <- lm(medv ~ 1, data = bh)
modelo_bh_completo_step <- lm(medv ~ indus + rm + ptratio + lstat, data = bh)

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

# Comparacion stepwise:
print(tabla_step_bh, row.names = FALSE)

# ============================================================
# PARTE B. HALD
# EJERCICIOS 7 Y 8: SUBCONJUNTOS Y MULTICOLINEALIDAD
# ============================================================

# ============================================================
# 6. CARGA Y EXPLORACION DE HALD
# ============================================================

data("Hald", package = "mctest")
hald <- as.data.frame(mctest::Hald)
stopifnot(all(c("y", "X1", "X2", "X3", "X4") %in% names(hald)))
hald <- hald[, c("y", "X1", "X2", "X3", "X4")]
hald <- na.omit(hald)

# HALD: AUDITORIA DE LOS DATOS
# N° de obervaciones 
nrow(hald)
# N° de variables
ncol(hald)
# Valores faltantes:
sum(is.na(hald))

View(hald)
# Resumen estadistico:
summary(hald)
# Matriz de correlaciones:
round(cor(hald), 3)
# Correlaciones entre los predictores:
print(round(cor(hald[, c("X1", "X2", "X3", "X4")]), 3))
# Grafico de correlaciones entre predictores:
heatmap(
  cor(hald[, c("X1", "X2", "X3", "X4")]),
  symm = TRUE,
  main = "Hald: correlaciones entre predictores"
)

# ============================================================
# 7. MODELO COMPLETO DE HALD
# ============================================================

modelo_hald_completo <- lm(y ~ X1 + X2 + X3 + X4, data = hald)
summary(modelo_hald_completo)

#IC
confint(modelo_hald_completo)
# Analisis de varianza:
anova(modelo_hald_completo)

# ============================================================
# 8. SELECCION DE SUBCONJUNTOS PARA HALD
# ============================================================

modelo_hald_subconjuntos <- leaps::regsubsets(y ~ X1 + X2 + X3 + X4, data = hald, nvmax = 4, method = "exhaustive")

resumen_hald <- summary(modelo_hald_subconjuntos)

tabla_hald <- data.frame(
  numero_predictores = seq_along(resumen_hald$rss),
  RSS = resumen_hald$rss,
  R2 = resumen_hald$rsq,
  R2_ajustado = resumen_hald$adjr2,
  Cp = resumen_hald$cp,
  BIC = resumen_hald$bic
)

# Resultados por tamano de subconjunto:
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

# Variables seleccionadas por tamano:

for (i in seq_along(variables_por_tamano_hald)) {
  print(paste("\nModelo con", i, "predictor(es):"))
  print(variables_por_tamano_hald[[i]])
}

# ============================================================
# 9. COMPARACION DE CRITERIOS PARA HALD
# ============================================================

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

# Tabla comparativa:\
print(tabla_criterios_hald, row.names = FALSE)

indice_r2_hald <- which.max(tabla_criterios_hald$R2_ajustado)
indice_cp_hald <- which.min(tabla_criterios_hald$Cp)
indice_bic_hald <- which.min(tabla_criterios_hald$BIC)

modelo_hald_r2 <- modelos_hald[[indice_r2_hald]]
modelo_hald_cp <- modelos_hald[[indice_cp_hald]]
modelo_hald_bic <- modelos_hald[[indice_bic_hald]]

# Mejor modelo segun R2 ajustado:
formula(modelo_hald_r2)
summary(modelo_hald_r2)

cat("\nMejor modelo segun Cp:\n")
print(formula(modelo_hald_cp))
print(summary(modelo_hald_cp))

# Mejor modelo segun BIC:
print(formula(modelo_hald_bic))
print(summary(modelo_hald_bic))


# ============================================================
# 10. SELECCION STEPWISE PARA HALD
# ============================================================

modelo_hald_nulo <- lm(y ~ 1,data = hald)
modelo_hald_completo_step <- lm(y ~ X1 + X2 + X3 + X4, data = hald)

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

# Comparacion de los modelos stepwise:
print(tabla_step_hald, row.names = FALSE)


# ============================================================
# 11. MULTICOLINEALIDAD EN HALD: VIF
# ============================================================

# HALD: ANALISIS DE MULTICOLINEALIDAD
# VIF del modelo completo:
print(car::vif(modelo_hald_completo))

# VIF del modelo seleccionado por R2 ajustado:
if (length(attr(terms(modelo_hald_r2), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_r2))
} else {
  print("El modelo tiene menos de dos predictores.")
}

# VIF del modelo seleccionado por Cp:
if (length(attr(terms(modelo_hald_cp), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_cp))
} else {
  print("El modelo tiene menos de dos predictores.")
}

# VIF del modelo seleccionado por BIC:

if (length(attr(terms(modelo_hald_bic), "term.labels")) >= 2) {
  print(car::vif(modelo_hald_bic))
} else {
  print("El modelo tiene menos de dos predictores.")
}

# ============================================================
# 12. INDICES DE CONDICION PARA HALD
# ============================================================

X_hald <- as.matrix(hald[, c("X1", "X2", "X3", "X4")])

# Estandarizar los predictores para comparar sus escalas.
X_hald_estandarizada <- scale(X_hald, center = TRUE, scale = TRUE)

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

# Valores propios e indices de condicion:
print(tabla_condicion, row.names = FALSE)

# indice3e de condicion maximo
max(indices_condicion)

# Numero de condicion de la matriz de diseno original con intercepto.
matriz_diseno_hald <- model.matrix(y ~ X1 + X2 + X3 + X4,data = hald)

valores_propios_originales <- eigen(
  crossprod(matriz_diseno_hald),
  symmetric = TRUE,
  only.values = TRUE
)$values

numero_condicion_original <- sqrt(
  max(valores_propios_originales) /
    min(valores_propios_originales)
)

# N° de condicion de la matriz de diseño original
format(numero_condicion_original, scientific = TRUE)

# ============================================================
# PARTE C. HALD
# EJERCICIO 9: RIDGE, LASSO Y ELASTIC NET CON LOOCV
# ============================================================

# ============================================================
# 13. PREPARACION PARA REGULARIZACION
# ============================================================

X <- as.matrix(hald[, c("X1", "X2", "X3", "X4")])
y <- hald$y

stopifnot(
  nrow(X) == length(y),
  all(is.finite(X)),
  all(is.finite(y))
)

n_hald <- nrow(X)

# N° de observaciones:
n_hald
#N° de predictores:
ncol(X)

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

# HALD: RIDGE

# Lambda minimo:
cv_ridge_hald$lambda.min
# Lambda 1se:
cv_ridge_hald$lambda.1se
# MSE minimo en LOOCV:
min(cv_ridge_hald$cvm)

# Coeficientes Ridge para lambda.min:
coef(cv_ridge_hald, s = "lambda.min")
# Coeficientes Ridge para lambda.1se:
coef(cv_ridge_hald, s = "lambda.1se")

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

# HALD: LASSO\n")
# Lambda minimo:
cv_lasso_hald$lambda.min
# Lambda 1se:
cv_lasso_hald$lambda.1se
# MSE minimo en LOOCV:
min(cv_lasso_hald$cvm)

# Coeficientes Lasso para lambda.min:
coef(cv_lasso_hald, s = "lambda.min")
# Coeficientes Lasso para lambda.1se:
coef(cv_lasso_hald, s = "lambda.1se")

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

# HALD: ELASTIC NET\n")
# Lambda minimo:
cv_en_hald$lambda.min
# Lambda 1se:
cv_en_hald$lambda.1se
# MSE minimo en LOOCV:
min(cv_en_hald$cvm)

# Coeficientes Elastic Net para lambda.min:
coef(cv_en_hald, s = "lambda.min")
# Coeficientes Elastic Net para lambda.1se:
coef(cv_en_hald, s = "lambda.1se")

plot(
  cv_en_hald,
  main = "Hald: Elastic Net con LOOCV"
)

# ============================================================
# 17. COMPARACION DE LAMBDAS Y ERROR DE VALIDACION
# ============================================================

# HALD: COMPARACION DE REGULARIZACION

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

# Tabla de comparacion:
print(tabla_regularizacion_hald, row.names = FALSE)

# RMSE LOOCV estimado para cada metodo:

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
# 18. COMPARACION DE COEFICIENTES DE HALD
# ============================================================

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

# Coeficientes para lambda.min:
print(tabla_coeficientes_hald, row.names = FALSE)

# N° de coeficientes distintos de cero, sin intercepto
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
