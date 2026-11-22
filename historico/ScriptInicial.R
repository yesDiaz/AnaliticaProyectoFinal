# ============================================================
# PROYECTO 3: CARGAR, UNIFICAR Y ORDENAR DATASETS
# ============================================================



library(data.table)
library(readxl)


# 2. Rutas de entrada y salida
carpeta_datos <- "C:/Descargas3/Proyecto3Analitica/Datos"
carpeta_salida <- "C:/Descargas3/Proyecto3Analitica/Resultados"


# 3. Buscar archivos CSV y Excel
archivos <- list.files(
  carpeta_datos,
  pattern = "\\.(csv|xlsx|xls)$",
  full.names = TRUE,
  ignore.case = TRUE
)



# 4. Cargar los datasets
columnas <- c("STATION_ID", "YEAR", "MONTH", "DAY", "VALUE")
tablas <- list()

for (archivo in archivos) {
  
  es_csv <- grepl("\\.csv$", archivo, ignore.case = TRUE)
  hojas <- if (es_csv) "CSV" else excel_sheets(archivo)
  
  for (hoja in hojas) {
    
    # Leer como texto para conservar identificadores con ceros iniciales
    tabla <- if (es_csv) {
      fread(
        archivo,
        colClasses = "character",
        na.strings = c("", "NA")
      )
    } else {
      as.data.table(
        read_excel(archivo, sheet = hoja, col_types = "text")
      )
    }
    
    setnames(tabla, toupper(trimws(names(tabla))))
    
    # Verificar las cinco columnas esperadas
    if (anyDuplicated(names(tabla)) > 0 ||
        !setequal(names(tabla), columnas)) {
      stop(
        "Revisa las columnas de: ", basename(archivo),
        " | ", hoja,
        ". Deben ser STATION_ID, YEAR, MONTH, DAY y VALUE."
      )
    }
    
    setcolorder(tabla, columnas)
    tablas[[length(tablas) + 1L]] <- tabla
    
    message(
      "Cargado: ", basename(archivo),
      " | ", hoja, " | Filas: ", nrow(tabla)
    )
  }
}


# 5. Unificar todos los registros
datos_unificados <- rbindlist(tablas, use.names = TRUE)

if (nrow(datos_unificados) == 0) {
  stop("Los datasets no contienen registros.")
}


# 6. Convertir año, mes y día a enteros para ordenar correctamente
convertir_entero <- function(x, columna) {
  numero <- suppressWarnings(as.numeric(trimws(x)))
  
  invalido <- !is.finite(numero) |
    numero != trunc(numero) |
    numero < 1 |
    numero > 9999
  
  if (any(invalido)) {
    stop("Hay valores inválidos o vacíos en la columna ", columna, ".")
  }
  
  as.integer(numero)
}

datos_unificados[, YEAR := convertir_entero(YEAR, "YEAR")]
datos_unificados[, MONTH := convertir_entero(MONTH, "MONTH")]
datos_unificados[, DAY := convertir_entero(DAY, "DAY")]

# Validar las fechas sin agregar columnas al dataset
texto_fecha <- with(
  datos_unificados,
  sprintf("%04d-%02d-%02d", YEAR, MONTH, DAY)
)

fechas <- suppressWarnings(
  as.Date(texto_fecha, format = "%Y-%m-%d")
)

if (any(is.na(fechas) |
        format(fechas, "%Y-%m-%d") != texto_fecha)) {
  stop("Hay fechas inválidas. Revisa YEAR, MONTH y DAY.")
}


# 7. Ordenar cronológicamente
setorder(datos_unificados, YEAR, MONTH, DAY, STATION_ID)


# 8. Guardar el dataset unificado
archivo_salida <- file.path(
  carpeta_salida,
  "datos_unificados_ordenados.csv"
)

fwrite(
  datos_unificados,
  archivo_salida,
  bom = TRUE,
  na = ""
)


# 9. Mostrar resumen y primeros registros
cat("\nTotal de registros:", nrow(datos_unificados))
cat("\nPrimer año:", min(datos_unificados$YEAR))
cat("\nÚltimo año:", max(datos_unificados$YEAR))
cat("\nGuardado en:", archivo_salida, "\n\n")

print(head(datos_unificados, 10))

# Visualizar una muestra en RStudio
View(head(datos_unificados, 1000))

////////////////////////////////////////////////////////////////////////////////////
  
  # ============================================================
# EDA: ESTADÍSTICAS DESCRIPTIVAS DE ESTACIONES
# ============================================================

library(data.table)

# Copia para el análisis
datos_eda <- copy(datos_unificados)

# Conservar los identificadores como texto
datos_eda[, STATION_ID := trimws(as.character(STATION_ID))]
datos_eda[STATION_ID == "", STATION_ID := NA_character_]

# 1. Número de estaciones distintas, excluyendo identificadores faltantes
cantidad_estaciones <- uniqueN(
  datos_eda$STATION_ID,
  na.rm = TRUE
)

cat("Cantidad de estaciones:", cantidad_estaciones, "\n")

# 2. Registros sin identificador de estación
cat(
  "Registros sin STATION_ID:",
  sum(is.na(datos_eda$STATION_ID)),
  "\n"
)

# 3. Listado de estaciones
estaciones <- datos_eda[
  !is.na(STATION_ID),
  .(STATION_ID = sort(unique(STATION_ID)))
]

print(estaciones)

# 4. Cantidad de registros y años registrados por estación
resumen_estaciones <- datos_eda[
  !is.na(STATION_ID),
  .(
    REGISTROS = .N,
    PRIMER_ANIO = if (all(is.na(YEAR))) NA_integer_
    else min(YEAR, na.rm = TRUE),
    ULTIMO_ANIO = if (all(is.na(YEAR))) NA_integer_
    else max(YEAR, na.rm = TRUE),
    ANIOS_CON_REGISTROS = uniqueN(YEAR, na.rm = TRUE)
  ),
  by = STATION_ID
]

setorder(resumen_estaciones, -REGISTROS)

print(resumen_estaciones)

# 5. Cantidad de estaciones con registros en cada año
estaciones_por_anio <- datos_eda[
  !is.na(STATION_ID) & !is.na(YEAR),
  .(ESTACIONES = uniqueN(STATION_ID)),
  by = YEAR
]

setorder(estaciones_por_anio, YEAR)

print(estaciones_por_anio)

# 6. Gráfico de estaciones por año
if (nrow(estaciones_por_anio) > 0) {
  with(
    estaciones_por_anio,
    plot(
      YEAR,
      ESTACIONES,
      type = "b",
      pch = 19,
      col = "steelblue",
      xlab = "Año",
      ylab = "Número de estaciones con registros",
      main = "Estaciones registradas por año"
    )
  )
}

# ============================================================
# EDA TEMPORAL: YEAR, MONTH Y DAY
# ============================================================


library(data.table)
options(scipen = 999)


# 1. COMPROBAR LOS DATOS DE ENTRADA -----------------------------

if (!exists("datos_unificados")) {
  stop("Primero carga y unifica los datasets.")
}

columnas <- c("STATION_ID", "YEAR", "MONTH", "DAY", "VALUE")

if (!all(columnas %in% names(datos_unificados))) {
  stop("Faltan columnas requeridas en datos_unificados.")
}

temporal <- as.data.table(copy(datos_unificados))
temporal <- temporal[, ..columnas]
temporal[, FILA := .I]

temporal[, STATION_ID := trimws(as.character(STATION_ID))]
temporal[STATION_ID == "", STATION_ID := NA_character_]

# Conservar las fechas originales para revisar errores
temporal[, `:=`(
  YEAR_ORIGINAL = as.character(YEAR),
  MONTH_ORIGINAL = as.character(MONTH),
  DAY_ORIGINAL = as.character(DAY)
)]


# 2. CONVERTIR Y VALIDAR LAS FECHAS -----------------------------

a_entero <- function(x, minimo, maximo) {
  
  numero <- suppressWarnings(
    as.numeric(trimws(as.character(x)))
  )
  
  valido <- is.finite(numero) &
    numero == trunc(numero) &
    numero >= minimo &
    numero <= maximo
  
  resultado <- rep(NA_integer_, length(numero))
  resultado[valido] <- as.integer(numero[valido])
  
  resultado
}

temporal[, `:=`(
  YEAR = a_entero(YEAR, 1, 9999),
  MONTH = a_entero(MONTH, 1, 12),
  DAY = a_entero(DAY, 1, 31)
)]

temporal[, FECHA_TEXTO := sprintf(
  "%04d-%02d-%02d", YEAR, MONTH, DAY
)]

temporal[, FECHA := suppressWarnings(
  as.Date(FECHA_TEXTO, format = "%Y-%m-%d")
)]

# Comprobar que la fecha convertida coincide exactamente.
# Esto permite detectar fechas como el 31 de febrero.
temporal[, FECHA_VALIDA :=
           !is.na(YEAR) &
           !is.na(MONTH) &
           !is.na(DAY) &
           !is.na(FECHA)
]

temporal[
  FECHA_VALIDA == TRUE,
  FECHA_VALIDA := format(FECHA, "%Y-%m-%d") == FECHA_TEXTO
]

temporal[is.na(FECHA_VALIDA), FECHA_VALIDA := FALSE]
temporal[FECHA_VALIDA == FALSE, FECHA := as.Date(NA)]

fechas_invalidas <- temporal[
  FECHA_VALIDA == FALSE,
  .(
    FILA, STATION_ID,
    YEAR_ORIGINAL, MONTH_ORIGINAL, DAY_ORIGINAL,
    VALUE
  )
]

sin_estacion <- temporal[is.na(STATION_ID)]

# Solo esta copia de análisis excluye fechas o estaciones inválidas
base <- temporal[
  FECHA_VALIDA == TRUE & !is.na(STATION_ID)
]

if (nrow(base) == 0) {
  stop("No hay registros con estación y fecha válidas.")
}


# 3. IDENTIFICAR MEDICIONES NUMÉRICAS ---------------------------

base[, VALOR_NUM := suppressWarnings(
  as.numeric(trimws(as.character(VALUE)))
)]

base[, MEDICION_NUMERICA := is.finite(VALOR_NUM)]

# Aquí un número finito cuenta como medición numérica.
# Los códigos especiales, unidades y valores extremos
# deben validarse por separado.


# 4. REVISAR DUPLICADOS POR ESTACIÓN Y FECHA --------------------

diario <- base[
  , .(
    FILAS = .N,
    HAY_MEDICION = any(MEDICION_NUMERICA),
    VALORES_DISTINTOS = uniqueN(
      VALOR_NUM[MEDICION_NUMERICA]
    )
  ),
  by = .(STATION_ID, FECHA)
]

diario[, CONFLICTO := VALORES_DISTINTOS > 1L]
diario[, DISPONIBLE := HAY_MEDICION & !CONFLICTO]

duplicados <- diario[FILAS > 1L]

# No se eliminan filas originales.
# Para contar cobertura, cada estación-fecha cuenta una vez.


# 5. DEFINIR EL PERIODO DE ANÁLISIS -----------------------------

# Por defecto se usa todo el periodo observado
inicio <- min(base$FECHA)
fin <- max(base$FECHA)

# Si necesitas un periodo específico, cambia las líneas anteriores:
# inicio <- as.Date("1981-01-01")
# fin <- as.Date("2013-12-31")

if (is.na(inicio) || is.na(fin) || inicio > fin) {
  stop("El periodo de análisis no es válido.")
}

calendario <- data.table(
  FECHA = seq(inicio, fin, by = "day")
)

calendario[, `:=`(
  YEAR = as.integer(format(FECHA, "%Y")),
  MONTH = as.integer(format(FECHA, "%m"))
)]

estaciones <- sort(unique(base$STATION_ID))

observado <- copy(
  diario[FECHA >= inicio & FECHA <= fin]
)

observado[, `:=`(
  YEAR = as.integer(format(FECHA, "%Y")),
  MONTH = as.integer(format(FECHA, "%m"))
)]


# 6. COBERTURA MENSUAL ----------------------------------------

# Días esperados: respeta meses y años bisiestos
esperados <- calendario[
  , .(DIAS_ESPERADOS = .N),
  by = .(YEAR, MONTH)
]

# Combinaciones de todas las estaciones con cada año-mes.
# Esta parte reemplaza el merge con by = NULL que fallaba.
rejilla <- esperados[
  , .(STATION_ID = estaciones),
  by = .(YEAR, MONTH, DIAS_ESPERADOS)
]

setcolorder(
  rejilla,
  c("STATION_ID", "YEAR", "MONTH", "DIAS_ESPERADOS")
)

conteos <- observado[
  , .(
    DIAS_CON_FILA = .N,
    DIAS_DISPONIBLES = sum(DISPONIBLE),
    DIAS_SIN_MEDICION_NUMERICA = sum(!HAY_MEDICION),
    DIAS_CON_CONFLICTO = sum(CONFLICTO)
  ),
  by = .(STATION_ID, YEAR, MONTH)
]

cobertura_mensual <- merge(
  rejilla,
  conteos,
  by = c("STATION_ID", "YEAR", "MONTH"),
  all.x = TRUE
)

contadores <- c(
  "DIAS_CON_FILA",
  "DIAS_DISPONIBLES",
  "DIAS_SIN_MEDICION_NUMERICA",
  "DIAS_CON_CONFLICTO"
)

# Estos ceros son conteos de registros, no valores de lluvia
for (columna in contadores) {
  set(
    cobertura_mensual,
    i = which(is.na(cobertura_mensual[[columna]])),
    j = columna,
    value = 0L
  )
}

cobertura_mensual[, `:=`(
  DIAS_SIN_FILA = DIAS_ESPERADOS - DIAS_CON_FILA,
  COBERTURA_PCT = round(
    100 * DIAS_DISPONIBLES / DIAS_ESPERADOS,
    2
  )
)]

setorder(cobertura_mensual, STATION_ID, YEAR, MONTH)

# Verificar que todos los días queden clasificados
stopifnot(
  all(
    cobertura_mensual$DIAS_ESPERADOS ==
      cobertura_mensual$DIAS_SIN_FILA +
      cobertura_mensual$DIAS_DISPONIBLES +
      cobertura_mensual$DIAS_SIN_MEDICION_NUMERICA +
      cobertura_mensual$DIAS_CON_CONFLICTO
  )
)


# 7. COBERTURA ANUAL ------------------------------------------

cobertura_anual <- cobertura_mensual[
  , .(
    DIAS_ESPERADOS = sum(DIAS_ESPERADOS),
    DIAS_CON_FILA = sum(DIAS_CON_FILA),
    DIAS_DISPONIBLES = sum(DIAS_DISPONIBLES),
    DIAS_SIN_FILA = sum(DIAS_SIN_FILA),
    DIAS_SIN_MEDICION_NUMERICA =
      sum(DIAS_SIN_MEDICION_NUMERICA),
    DIAS_CON_CONFLICTO = sum(DIAS_CON_CONFLICTO)
  ),
  by = .(STATION_ID, YEAR)
]

cobertura_anual[, COBERTURA_PCT := round(
  100 * DIAS_DISPONIBLES / DIAS_ESPERADOS,
  2
)]

setorder(cobertura_anual, STATION_ID, YEAR)


# 8. HUECOS ENTRE REGISTROS -----------------------------------

# Detecta días sin ninguna fila entre dos registros.
# No incluye ausencias antes del primer registro ni después del último.
setorder(observado, STATION_ID, FECHA)

observado[
  , FECHA_ANTERIOR := shift(FECHA),
  by = STATION_ID
]

observado[
  , DIAS_AUSENTES := as.integer(FECHA - FECHA_ANTERIOR) - 1L
]

huecos <- observado[
  !is.na(DIAS_AUSENTES) & DIAS_AUSENTES > 0,
  .(
    STATION_ID,
    DESDE = FECHA_ANTERIOR + 1,
    HASTA = FECHA - 1,
    DIAS_AUSENTES
  )
]

setorder(huecos, -DIAS_AUSENTES)


# 9. RESUMEN DE ESTACIONES POR AÑO -----------------------------

estaciones_por_anio <- cobertura_anual[
  , .(
    ESTACIONES_CON_REGISTROS = sum(DIAS_CON_FILA > 0),
    ESTACIONES_CON_MEDICIONES = sum(DIAS_DISPONIBLES > 0),
    ESTACIONES_CON_COBERTURA_COMPLETA =
      sum(DIAS_DISPONIBLES == DIAS_ESPERADOS)
  ),
  by = YEAR
]

setorder(estaciones_por_anio, YEAR)


# 10. MOSTRAR RESULTADOS --------------------------------------

cat("\n========== RESUMEN GENERAL ==========\n")
cat("Filas originales:", nrow(temporal), "\n")
cat("Filas con fecha inválida:", nrow(fechas_invalidas), "\n")
cat("Filas sin estación:", nrow(sin_estacion), "\n")
cat("Estaciones con fecha válida:", length(estaciones), "\n")
cat("Primera fecha observada:",
    as.character(min(base$FECHA)), "\n")
cat("Última fecha observada:",
    as.character(max(base$FECHA)), "\n")
cat("Periodo evaluado:", as.character(inicio),
    "a", as.character(fin), "\n")
cat("Pares estación-fecha duplicados:", nrow(duplicados), "\n")
cat("Pares con conflicto:",
    diario[CONFLICTO == TRUE, .N], "\n")

cat("\n--- Estaciones por año ---\n")
print(estaciones_por_anio, nrows = Inf)

cat("\n--- 20 combinaciones estación-año con menor cobertura ---\n")
print(head(cobertura_anual[order(COBERTURA_PCT)], 20))

cat("\n--- Primeros 20 meses sin registros ---\n")
print(head(cobertura_mensual[DIAS_CON_FILA == 0], 20))

cat("\n--- 20 huecos más largos entre registros ---\n")
print(head(huecos, 20))

cat("\n--- Primeras 20 filas con fechas inválidas ---\n")
print(head(fechas_invalidas, 20))


# 11. CONSULTAR TABLAS EN RSTUDIO ------------------------------

if (interactive()) {
  View(cobertura_anual)
  View(cobertura_mensual)
  View(huecos)
}

resumen_huecos <- huecos[
  , .(
    PERIODOS_SIN_REGISTROS = .N,
    TOTAL_DIAS_AUSENTES = sum(DIAS_AUSENTES),
    MAYOR_HUECO_DIAS = max(DIAS_AUSENTES)
  ),
  by = STATION_ID
]

setorder(resumen_huecos, -TOTAL_DIAS_AUSENTES)

View(resumen_huecos)


# Ejecutar solo si no tienes el paquete:
# install.packages("ggplot2")

library(ggplot2)

ggplot(
  cobertura_anual,
  aes(
    x = YEAR,
    y = factor(STATION_ID),
    fill = COBERTURA_PCT
  )
) +
  geom_tile() +
  scale_fill_gradient(
    low = "white",
    high = "steelblue4",
    limits = c(0, 100),
    name = "Cobertura (%)"
  ) +
  labs(
    title = "Cobertura diaria por estación y año",
    subtitle = "Porcentaje de días con medición numérica sin conflicto",
    x = "Año",
    y = "Estación"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_text(size = 6)
  )





# 1. Copiar la cobertura anual
cobertura_rangos <- copy(cobertura_anual)

# Para intervalos de 5 años, cambia 10 por 5
amplitud <- 10L

cobertura_rangos[
  , INICIO_RANGO := as.integer(floor(YEAR / amplitud) * amplitud)
]

# 2. Calcular cobertura por estación y rango
resumen_rangos <- cobertura_rangos[
  , .(
    PRIMER_ANIO = min(YEAR),
    ULTIMO_ANIO = max(YEAR),
    DIAS_DISPONIBLES = sum(DIAS_DISPONIBLES),
    DIAS_ESPERADOS = sum(DIAS_ESPERADOS)
  ),
  by = .(STATION_ID, INICIO_RANGO)
]

resumen_rangos[, COBERTURA_PCT := 
                 100 * DIAS_DISPONIBLES / DIAS_ESPERADOS
]

# Etiquetas con los años realmente evaluados:
# por ejemplo, el último rango puede ser 2000–2003
resumen_rangos[, RANGO := paste(
  PRIMER_ANIO, ULTIMO_ANIO, sep = "–"
)]

# Ordenar los rangos cronológicamente
etiquetas <- unique(
  resumen_rangos[order(INICIO_RANGO), RANGO]
)

resumen_rangos[, RANGO := factor(RANGO, levels = etiquetas)]

# 3. Mapa de calor
ggplot(
  resumen_rangos,
  aes(
    x = RANGO,
    y = factor(STATION_ID),
    fill = COBERTURA_PCT
  )
) +
  geom_tile(color = "grey85", linewidth = 0.2) +
  scale_fill_gradient(
    low = "white",
    high = "steelblue4",
    limits = c(0, 100),
    name = "Cobertura (%)"
  ) +
  labs(
    title = "Cobertura diaria por estación y rango de años",
    subtitle = paste("Intervalos de", amplitud, "años"),
    x = "Rango de años evaluados",
    y = "Estación"
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 6)
  )




# ============================================================
# EDA DE LA COLUMNA VALUE
# ============================================================



# 1. PREPARAR LOS DATOS ---------------------------------------

eda_valores <- as.data.table(copy(datos_unificados))

eda_valores[, VALOR_TEXTO := trimws(as.character(VALUE))]

eda_valores[, ES_FALTANTE :=
              is.na(VALOR_TEXTO) | VALOR_TEXTO == ""
]

eda_valores[, VALOR_NUM := suppressWarnings(
  as.numeric(VALOR_TEXTO)
)]

eda_valores[, ES_NUMERICO :=
              !ES_FALTANTE & is.finite(VALOR_NUM)
]




# 2. REVISAR FALTANTES Y CONVERSIONES --------------------------

calidad_valores <- eda_valores[
  , .(
    TOTAL_REGISTROS = .N,
    VALORES_NUMERICOS = sum(ES_NUMERICO),
    DATOS_FALTANTES = sum(ES_FALTANTE),
    VALORES_NO_CONVERTIBLES = sum(!ES_FALTANTE & !ES_NUMERICO),
    PORCENTAJE_FALTANTES = round(100 * mean(ES_FALTANTE), 2)
  )
]

cat("\n--- CALIDAD DE VALUE ---\n")
print(calidad_valores)

# Mostrar textos que no pudieron interpretarse como números
problemas_conversion <- eda_valores[
  !ES_FALTANTE & !ES_NUMERICO,
  .(REGISTROS = .N),
  by = VALUE
]

if (nrow(problemas_conversion) > 0) {
  cat("\n--- VALORES QUE REQUIEREN REVISIÓN ---\n")
  print(problemas_conversion)
}


# 3. ESTADÍSTICAS GENERALES -----------------------------------

valores <- eda_valores[ES_NUMERICO == TRUE, VALOR_NUM]

if (length(valores) == 0) {
  stop("No hay valores numéricos disponibles para el análisis.")
}

q1 <- as.numeric(quantile(valores, 0.25))
q3 <- as.numeric(quantile(valores, 0.75))
ric <- q3 - q1

resumen_valores <- data.table(
  ESTADISTICA = c(
    "Cantidad de valores numéricos",
    "Mínimo",
    "Cuartil 1 (25 %)",
    "Mediana (50 %)",
    "Media",
    "Cuartil 3 (75 %)",
    "Percentil 95",
    "Percentil 99",
    "Máximo",
    "Amplitud del rango (máximo - mínimo)",
    "Rango intercuartílico",
    "Desviación estándar",
    "Varianza",
    "Cantidad de ceros",
    "Porcentaje de ceros",
    "Cantidad de negativos",
    "Cantidad de positivos"
  ),
  VALOR = c(
    length(valores),
    min(valores),
    q1,
    median(valores),
    mean(valores),
    q3,
    as.numeric(quantile(valores, 0.95)),
    as.numeric(quantile(valores, 0.99)),
    max(valores),
    max(valores) - min(valores),
    ric,
    sd(valores),
    var(valores),
    sum(valores == 0),
    100 * mean(valores == 0),
    sum(valores < 0),
    sum(valores > 0)
  )
)

cat("\n--- ESTADÍSTICAS GENERALES ---\n")
print(resumen_valores, digits = 10)


# 4. REVISAR LOS VALORES MÁS ALTOS -----------------------------

# Mostrar estación y fecha para localizar los registros originales
valores_mas_altos <- eda_valores[
  ES_NUMERICO == TRUE
][
  order(-VALOR_NUM),
  .(STATION_ID, YEAR, MONTH, DAY, VALUE, VALOR_NUM)
]

cat("\n--- LOS 20 REGISTROS CON MAYOR VALUE ---\n")
print(head(valores_mas_altos, 20))


# 5. SEÑALAR POSIBLES VALORES ATÍPICOS -------------------------

limite_inferior <- q1 - 1.5 * ric
limite_superior <- q3 + 1.5 * ric

posibles_atipicos <- eda_valores[
  ES_NUMERICO == TRUE &
    (VALOR_NUM < limite_inferior | VALOR_NUM > limite_superior),
  .(STATION_ID, YEAR, MONTH, DAY, VALUE, VALOR_NUM)
]

cat("\n--- REGLA DEL RANGO INTERCUARTÍLICO ---\n")
cat("Límite inferior:", limite_inferior, "\n")
cat("Límite superior:", limite_superior, "\n")
cat("Registros señalados:", nrow(posibles_atipicos), "\n")

# Se señalan para revisión, no se eliminan.


# 6. ESTADÍSTICAS POR ESTACIÓN --------------------------------

resumen_por_estacion <- eda_valores[
  !is.na(STATION_ID) & trimws(as.character(STATION_ID)) != "",
  {
    x <- VALOR_NUM[ES_NUMERICO]
    
    .(
      REGISTROS = .N,
      NUMERICOS = length(x),
      FALTANTES = sum(ES_FALTANTE),
      NO_CONVERTIBLES = sum(!ES_FALTANTE & !ES_NUMERICO),
      MINIMO = if (length(x)) min(x) else NA_real_,
      MEDIA = if (length(x)) mean(x) else NA_real_,
      MEDIANA = if (length(x)) median(x) else NA_real_,
      MAXIMO = if (length(x)) max(x) else NA_real_,
      DESVIACION_ESTANDAR = if (length(x) > 1) sd(x) else NA_real_,
      PORCENTAJE_CEROS = if (length(x)) {
        100 * mean(x == 0)
      } else {
        NA_real_
      }
    )
  },
  by = STATION_ID
]

setorder(resumen_por_estacion, -MAXIMO, na.last = TRUE)

cat("\n--- RESUMEN POR ESTACIÓN ---\n")
print(head(resumen_por_estacion, 20), digits = 8)


# 1. Resumen anual de las mediciones numéricas
resumen_anual <- eda_valores[
  ES_NUMERICO == TRUE & !is.na(YEAR),
  .(
    REGISTROS = .N,
    ESTACIONES = uniqueN(STATION_ID, na.rm = TRUE),
    MEDIA = mean(VALOR_NUM),
    MEDIANA = median(VALOR_NUM),
    MAXIMO = max(VALOR_NUM)
  ),
  by = YEAR
]

setorder(resumen_anual, YEAR)

# 2. Preparar las dos series
datos_grafica <- melt(
  resumen_anual,
  id.vars = "YEAR",
  measure.vars = c("MEDIA", "MEDIANA"),
  variable.name = "ESTADISTICA",
  value.name = "VALOR"
)

# 3. Graficar
ggplot(
  datos_grafica,
  aes(x = YEAR, y = VALOR, color = ESTADISTICA)
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2) +
  scale_color_manual(
    values = c("MEDIA" = "steelblue", "MEDIANA" = "darkorange"),
    labels = c("MEDIA" = "Media", "MEDIANA" = "Mediana"),
    name = NULL
  ) +
  labs(
    title = "Comportamiento de VALUE por año",
    subtitle = "Resumen de las mediciones disponibles de todas las estaciones",
    x = "Año",
    y = "VALUE (unidad del dataset)"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  )

# Consultar los resultados numéricos
print(resumen_anual, nrows = Inf)





