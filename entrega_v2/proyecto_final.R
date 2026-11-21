#Script: Proyecto final Analítica de Datos 2026-2S (UAO) - Fases 0, 1, 2 y 3 en R
#Precipitación en el Valle del Cauca: estaciones en tierra (diario) vs producto satelital (péntada)
#
#Ejecución (desde la carpeta entrega/):   Rscript proyecto_final.R            (todas las fases)
#                                          Rscript proyecto_final.R fase1      (una fase; usa cache/)
#Rutas relativas: ../datos (datos del curso), ./datos_externos (ONI NOAA-CPC), ./cache, ./tablas_R, ./figuras
#Semilla fija: 2026.
#Versiones usadas: R 4.3.3; data.table 1.14.10; readxl 1.4.3; terra 1.7-65 (lectura GeoTIFF);
#                  fda 6.3.0 y fda.usc 2.2.0 (solo verificación en fase 1 y cálculo en fases 2-3).
#Fase 1 (construcción de datos funcionales) se programa A MANO: base B-spline por Cox-de Boor,
#Cholesky, mínimos cuadrados, VC, VCG y matriz de suavizado. fda se usa únicamente para COMPROBAR.
#Este script replica la metodología del script de Python (proyecto_final.py) y, al final de cada
#fase, compara sus resultados con los de Python (tablas_R/TabR_comparacion_*.csv).

rm(list = ls())
set.seed(2026)
options(datatable.integer64 = "double")     #los códigos de estación (10 dígitos) se leen como double

suppressMessages({
  library(data.table)
  library(readxl)
})

#Rutas (el script se ejecuta desde entrega/)
DATOS <- file.path("..", "datos")
EXT   <- "datos_externos"
CACHE <- "cache"
TABR  <- "tablas_R"
TABPY <- "tablas"
FIG   <- "figuras"
for (d in c(CACHE, TABR, FIG)) dir.create(d, showWarnings = FALSE)

writeLines(c(R.version.string,
             paste("data.table", packageVersion("data.table")),
             paste("readxl", packageVersion("readxl")),
             paste("fda", tryCatch(as.character(packageVersion("fda")), error = function(e) "NO")),
             paste("fda.usc", tryCatch(as.character(packageVersion("fda.usc")), error = function(e) "NO")),
             paste("terra", tryCatch(as.character(packageVersion("terra")), error = function(e) "NO"))),
           file.path(TABR, "versiones_R.txt"))

ANIO_INI <- 1981; ANIO_FIN <- 2013; ANIO_REF_FIN <- 1995
DIAS     <- 365
T_FINO   <- (0:364) + 0.5
DIAS_MES <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
INICIO_MES <- c(0, cumsum(DIAS_MES)[-12])
doy365 <- function(mes, dia) INICIO_MES[mes] + dia

#Péntadas en calendario de 365 días: inicio, longitud y punto medio
PENT_INI <- PENT_LON <- numeric(72)
k <- 0
for (m in 1:12) for (j in 0:5) {
  k <- k + 1
  a <- 5 * j; b <- if (j < 5) 5 * j + 5 else DIAS_MES[m]
  PENT_INI[k] <- INICIO_MES[m] + a; PENT_LON[k] <- b - a
}
T_PENT <- PENT_INI + PENT_LON / 2

fecha_txt <- function(t) {
  d <- pmin(pmax(floor(t), 0), 364)
  m <- findInterval(d, INICIO_MES)
  sprintf("%02d-%s", d - INICIO_MES[m] + 1, c("ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic")[m])
}

guarda <- function(df, nombre) { fwrite(as.data.table(df), file.path(TABR, nombre)); cat("tabla", nombre, "\n") }

#Racha más larga de TRUE (como racha_max() de los scripts del docente)
rachas <- function(x) { r <- rle(x); list(ini = cumsum(c(1, head(r$lengths, -1)))[r$values], lon = r$lengths[r$values]) }

args <- commandArgs(trailingOnly = TRUE)
FASES <- if (length(args)) args else c("fase0", "fase0b", "fase1", "fase2", "fase3")



#===================================================================================
#FASE 0. Preprocesamiento de las estaciones (mismas reglas que en Python)
#===================================================================================
fase0 <- function() {
  cat("FASE 0\n")
  ruta <- file.path(CACHE, "R_estaciones_crudo.rds")
  if (file.exists(ruta)) crudo <- readRDS(ruta) else {
    x <- as.data.table(read_excel(file.path(DATOS, "P_1930_1979.xlsx"), sheet = "Hoja1"))
    x <- x[, .(STATION_ID = as.numeric(STATION_ID), YEAR = as.integer(YEAR), MONTH = as.integer(MONTH),
               DAY = as.integer(DAY), VALUE = as.numeric(VALUE))]
    y <- fread(file.path(DATOS, "P_1980_1990.csv")); z <- fread(file.path(DATOS, "P_1991_2003.csv"))
    crudo <- rbind(x, y[, .(STATION_ID = as.numeric(STATION_ID), YEAR, MONTH, DAY, VALUE = as.numeric(VALUE))],
                   z[, .(STATION_ID = as.numeric(STATION_ID), YEAR, MONTH, DAY, VALUE = as.numeric(VALUE))])
    saveRDS(crudo, ruta)
  }
  d <- crudo[!(MONTH == 2 & DAY == 29)]; rm(crudo); gc()
  d[, doy := doy365(MONTH, DAY)]
  setorder(d, STATION_ID, YEAR, doy)
  d[, bandera := ""]
  traz <- list()
  registra <- function(paso, afect) {
    v <- d[YEAR >= ANIO_INI & YEAR <= ANIO_FIN & !is.na(VALUE)]
    traz[[length(traz) + 1]] <<- data.table(paso = paso, estaciones = uniqueN(v$STATION_ID),
                                             pares_estacion_anio = nrow(unique(v[, .(STATION_ID, YEAR)])),
                                             dias_validos = nrow(v), valores_afectados = afect)
    print(traz[[length(traz)]])
  }
  per <- function() d$YEAR >= ANIO_INI & d$YEAR <= ANIO_FIN
  registra("P0 crudo", NA)

  #P1 valores imposibles
  imp <- is.na(d$VALUE) | d$VALUE < 0 | d$VALUE > 1000
  d[imp, `:=`(bandera = "imposible", VALUE = NA)]
  registra("P1 valores imposibles", sum(imp & per()))

  #P2 totales mensuales en un solo día
  mes <- d[, .(n = sum(!is.na(VALUE)), tot = sum(VALUE, na.rm = TRUE), humedos = sum(VALUE > 0, na.rm = TRUE),
               vmax = suppressWarnings(max(VALUE, na.rm = TRUE))), by = .(STATION_ID, YEAR, MONTH)]
  clim <- mes[YEAR <= ANIO_REF_FIN & n >= 25, .(tot_med = median(tot), hum_med = median(humedos)), by = .(STATION_ID, MONTH)]
  mes <- merge(mes, clim, by = c("STATION_ID", "MONTH"), all.x = TRUE)
  tm <- mes[humedos == 1 & vmax >= 50 & vmax >= 0.5 * tot_med & hum_med >= 5]
  guarda(tm, "TabR03_totales_mensuales_un_dia.csv")
  d[tm, on = .(STATION_ID, YEAR, MONTH), `:=`(bandera = "total_mensual", marca = TRUE)]
  afect <- sum(!is.na(d$marca) & per())
  d[marca == TRUE, VALUE := NA]; d[, marca := NULL]
  registra("P2 totales mensuales en un día", afect)

  #P3 errores de digitación
  q <- d[YEAR <= ANIO_REF_FIN & VALUE > 0, .(q999 = quantile(VALUE, 0.999, type = 7)), by = STATION_ID]
  d[q, on = "STATION_ID", q999 := i.q999]
  dig <- !is.na(d$VALUE) & !is.na(d$q999) & d$VALUE > 3 * d$q999 & d$VALUE > 150
  d[dig, `:=`(bandera = "digitacion", VALUE = NA)]; d[, q999 := NULL]
  registra("P3 errores de digitación", sum(dig & per()))

  #P3b meses con escala errónea
  mt <- d[, .(tot = sum(VALUE, na.rm = TRUE), n = sum(!is.na(VALUE))), by = .(STATION_ID, YEAR, MONTH)]
  p95 <- mt[YEAR <= ANIO_REF_FIN & n >= 25, .(p95m = quantile(tot, 0.95, type = 7)), by = STATION_ID]
  mt <- merge(mt, p95, by = "STATION_ID", all.x = TRUE)
  mal <- mt[!is.na(p95m) & tot > 5 * p95m]
  d[mal, on = .(STATION_ID, YEAR, MONTH), marca := TRUE]
  afect <- sum(!is.na(d$marca) & per())
  d[marca == TRUE, `:=`(bandera = "mes_escala_erronea", VALUE = NA)]; d[, marca := NULL]
  registra("P3b meses con escala errónea", afect)

  #P3c bloques de 10 días copiados (ventana idéntica en otra fecha de la misma estación)
  VENT <- 10
  dup <- logical(nrow(d))
  idx_est <- split(seq_len(nrow(d)), d$STATION_ID)
  for (idx in idx_est) {
    v <- d$VALUE[idx]; nv <- length(v)
    if (nv < 2 * VENT) next
    S <- embed(v, VENT)[, VENT:1, drop = FALSE]          #fila i = ventana que empieza en i
    ok <- rowSums(!is.na(S) & S > 0) >= 6
    if (!any(ok)) next
    cl <- apply(S[ok, , drop = FALSE], 1, function(r) paste(sprintf("%.4f", r), collapse = "|"))
    ii <- which(ok)
    primero <- ii[match(cl, cl)]
    es <- (ii - primero) >= VENT
    for (k in which(es)) { dup[idx[ii[k]:(ii[k] + VENT - 1)]] <- TRUE; dup[idx[primero[k]:(primero[k] + VENT - 1)]] <- TRUE }
  }
  d[dup, `:=`(bandera = "bloque_copiado", VALUE = NA)]
  registra("P3c bloques copiados", sum(dup & per()))

  #P4 ceros falsos: Markov-1 (años de referencia) + verificación con vecinas de la subzona
  d[, seco := !is.na(VALUE) & VALUE == 0]
  d[, prev_seco := shift(seco), by = STATION_ID]
  d[, prev_ok := !is.na(shift(VALUE)) & shift(doy) == doy - 1, by = STATION_ID]
  d[is.na(prev_ok), prev_ok := FALSE]
  p11 <- d[YEAR <= ANIO_REF_FIN & prev_ok & !is.na(VALUE) & prev_seco == TRUE, .(p11 = mean(seco)), by = .(STATION_ID, MONTH)]
  p11[, p11 := pmin(p11, 0.995)]
  d[, p11 := NA_real_]
  d[p11, on = c("STATION_ID", "MONTH"), p11 := i.p11]
  d[is.na(p11), p11 := 0.9]
  d[, logp := log(pmax(p11, 0.05))]
  d[, humedo := as.integer(!is.na(VALUE) & VALUE > 0)]
  d[, pref := floor(STATION_ID / 1e6)]
  reg <- d[!is.na(VALUE), .(nv = .N, nw = sum(humedo)), by = .(pref, YEAR, doy)]
  d[, `:=`(nv = 0L, nw = 0L)]
  d[reg, on = c("pref", "YEAR", "doy"), `:=`(nv = i.nv, nw = i.nw)]
  rm(reg); gc()
  d[, n_vec := nv - as.integer(!is.na(VALUE))]
  d[, lluvia_reg := ifelse(n_vec > 0, (nw - humedo) / n_vec >= 0.5, FALSE)]
  falsos <- logical(nrow(d)); filas <- list()
  idx_est <- split(seq_len(nrow(d)), d$STATION_ID)
  for (e in names(idx_est)) {
    idx <- idx_est[[e]]
    dd <- d$doy[idx] + 365 * d$YEAR[idx]
    cont <- c(TRUE, diff(dd) == 1)
    es0 <- d$seco[idx]
    nuevo <- !(es0 & c(FALSE, head(es0, -1)) & cont)
    grupo <- cumsum(nuevo)
    lp <- ifelse(nuevo, 0, d$logp[idx])
    largo <- tapply(es0, grupo, sum); slp <- tapply(lp * es0, grupo, sum)
    cand <- as.integer(names(largo))[largo >= 10 & slp < log(1e-4)]
    for (g in cand) {
      pos <- which(grupo == g & es0)
      info <- d$n_vec[idx[pos]] >= 2
      if (sum(info) >= 0.5 * length(pos)) {
        frac <- mean(d$lluvia_reg[idx[pos]][info]); dec <- if (frac >= 0.2) "cero_falso" else "sequia_regional_se_conserva"
      } else { frac <- NA; dec <- "cero_falso_sin_vecinas" }
      if (dec != "sequia_regional_se_conserva") falsos[idx[pos]] <- TRUE
      filas[[length(filas) + 1]] <- data.table(STATION_ID = e, YEAR = d$YEAR[idx[pos[1]]], longitud_dias = length(pos),
                                               log10_prob = slp[as.character(g)] / log(10), frac_dias_lluvia_regional = frac, decision = dec)
    }
  }
  guarda(rbindlist(filas), "TabR05_ceros_falsos.csv")
  d[falsos, `:=`(bandera = "cero_falso", VALUE = NA)]
  d[, c("seco", "prev_seco", "prev_ok", "p11", "logp", "humedo", "pref", "nv", "nw", "n_vec", "lluvia_reg") := NULL]; gc()
  registra("P4 ceros falsos", sum(falsos & per()))

  #P5 días faltantes (calendario completo) y P6 cobertura
  d <- d[YEAR >= ANIO_INI & YEAR <= ANIO_FIN]
  pares <- unique(d[, .(STATION_ID, YEAR)]); setorder(pares, STATION_ID, YEAR)
  pares[, fila := .I]
  d[pares, on = c("STATION_ID", "YEAR"), fila := i.fila]
  M <- matrix(NA_real_, nrow(pares), DIAS)
  M[cbind(d$fila, d$doy)] <- d$VALUE
  val <- !is.na(M)
  pares[, dias_validos := rowSums(val)]
  pares[, hueco_max := apply(val, 1, function(v) { r <- rachas(!v); if (length(r$lon)) max(r$lon) else 0 })]
  pares[, borde := apply(val, 1, function(v) {
    r <- rachas(!v); if (!length(r$lon)) return(0)
    b0 <- if (any(r$ini == 1)) max(r$lon[r$ini == 1]) else 0
    b1 <- if (any(r$ini + r$lon - 1 == DIAS)) max(r$lon[r$ini + r$lon - 1 == DIAS]) else 0
    max(b0, b1) })]
  traz[[length(traz) + 1]] <- data.table(paso = "P5 días faltantes", estaciones = uniqueN(pares[dias_validos > 0]$STATION_ID),
                                         pares_estacion_anio = sum(pares$dias_validos > 0), dias_validos = sum(val),
                                         valores_afectados = sum(!val))
  pares[, cumple := dias_validos >= 330 & hueco_max <= 30 & borde <= 7]
  ok <- pares[cumple == TRUE]
  traz[[length(traz) + 1]] <- data.table(paso = "P6 cobertura", estaciones = uniqueN(ok$STATION_ID), pares_estacion_anio = nrow(ok),
                                         dias_validos = sum(ok$dias_validos), valores_afectados = sum(!pares$cumple))
  #P7 estaciones con >= 5 años
  na <- ok[, .N, by = STATION_ID]
  ok7 <- ok[STATION_ID %in% na[N >= 5]$STATION_ID]
  traz[[length(traz) + 1]] <- data.table(paso = "P7 estaciones >= 5 años", estaciones = uniqueN(ok7$STATION_ID),
                                         pares_estacion_anio = nrow(ok7), dias_validos = sum(ok7$dias_validos),
                                         valores_afectados = nrow(ok) - nrow(ok7))
  cvc <- fread(file.path(DATOS, "CVC_Wheather_Stations_Long_Lat.csv"), encoding = "Latin-1")
  geo <- cvc[Lon > -77.55 & Lon < -75.70 & lat > 3.05 & lat < 5.00]
  ok8 <- ok7[STATION_ID %in% geo$CODIGO]
  traz[[length(traz) + 1]] <- data.table(paso = "P8 georreferenciadas en la malla", estaciones = uniqueN(ok8$STATION_ID),
                                         pares_estacion_anio = nrow(ok8), dias_validos = sum(ok8$dias_validos),
                                         valores_afectados = nrow(ok7) - nrow(ok8))
  T <- rbindlist(traz, fill = TRUE)
  guarda(T, "TabR02_trazabilidad.csv")
  sel <- pares$cumple & pares$STATION_ID %in% ok7$STATION_ID
  saveRDS(list(Y = M[sel, ], station = pares$STATION_ID[sel], year = pares$YEAR[sel]), file.path(CACHE, "R_fase0_matriz.rds"))
  #Comparación con Python
  if (file.exists(file.path(TABPY, "Tab02_trazabilidad.csv"))) {
    py <- fread(file.path(TABPY, "Tab02_trazabilidad.csv"))
    k <- 1:6   # P0-P3c son comunes; desde P4 Python usa la versión 2 (rachas 30 d + vecinas, reglas, imputación)
    cmp <- data.table(paso_R = T$paso[k], estaciones_R = T$estaciones[k], estaciones_Py = py$estaciones[k],
                      pares_R = T$pares_estacion_anio[k], pares_Py = py$pares_estacion_anio[k],
                      dias_R = T$dias_validos[k], dias_Py = py$dias_validos[k])
    guarda(cmp, "TabR_comparacion_trazabilidad.csv"); print(cmp)
  }
  invisible(T)
}



#===================================================================================
#FASE 0b (versión 2, asesorías 19-20 nov). La imputación espacial (IDW / CSW / kriging) se hace en
#Python (proyecto_final.py, fase0). Aquí: (1) se reproduce en R la decisión de reglas a partir del
#inventario de % de faltantes por estación y año (Tab06), (2) se verifica la matriz imputada y
#(3) se deja como entrada de las fases 1-3 de R (mismos datos que Python).
#===================================================================================
fase0b <- function() {
  cat("FASE 0b\n")
  inv <- fread(file.path(TABPY, "Tab06_faltantes_pct_estacion_anio.csv"))
  anios <- as.character(1981:2003)
  P <- as.matrix(inv[, ..anios])
  pct_datos <- 100 - rowMeans(P)
  elegida <- pct_datos > 45 & inv$con_coordenadas
  reglaR <- data.table(regla = c("Nacional >45 % (con coordenadas)", "  de ellas Pacífico", "pares con <=25 % faltantes"),
                       R = c(sum(elegida), sum(elegida & inv$region == "Pacífico"), sum((P <= 25) & elegida)))
  py <- fread(file.path(TABPY, "Tab06c_comparacion_reglas_estaciones.csv"))[regla == "Nacional (IDEAM)"]
  reglaR[, Python := c(py$estaciones_con_coord, py$Pacifico_con_coord, py$pares_estacion_anio_utilizables)]
  guarda(reglaR, "TabR06_reglas_desde_inventario.csv"); print(reglaR)
  m <- fread(file.path(CACHE, "fase0_matriz_diaria.csv.gz"))
  Y <- as.matrix(m[, paste0("d", 1:DIAS), with = FALSE])
  chk <- data.table(curvas = nrow(Y), estaciones = uniqueN(m$station), min_valor = min(Y, na.rm = TRUE),
                    pct_NaN = 100 * mean(is.na(Y)), pct_ceros = 100 * mean(Y == 0, na.rm = TRUE))
  guarda(chk, "TabR06b_verificacion_matriz_imputada.csv"); print(chk)
  saveRDS(list(Y = Y, station = m$station, year = m$year), file.path(CACHE, "R_fase0_matriz.rds"))
  invisible(chk)
}


#===================================================================================
#FASE 1. Construcción de datos funcionales PASO A PASO (sin librerías)
#===================================================================================
ORDEN <- 4

nodos_bspline <- function(K, a = 0, b = DIAS, orden = ORDEN) {
  interiores <- seq(a, b, length.out = K - orden + 2)[-c(1, K - orden + 2)]
  c(rep(a, orden), interiores, rep(b, orden))
}

#Matriz Φ por la recursión de Cox-de Boor (deriv = 0, 1 o 2)
base_bspline <- function(t, K, orden = ORDEN, deriv = 0) {
  tau <- nodos_bspline(K, orden = orden); b <- tau[length(tau)]
  n_int <- length(tau) - 1
  B <- matrix(0, length(t), n_int)
  for (i in 1:n_int) if (tau[i] < tau[i + 1]) B[, i] <- as.numeric(t >= tau[i] & t < tau[i + 1])
  ult <- max(which(tau[-length(tau)] < tau[-1]))
  B[t == b, ult] <- 1
  if (orden - deriv >= 2) for (k in 2:(orden - deriv)) {
    Bn <- matrix(0, length(t), n_int - k + 1)
    for (i in 1:(n_int - k + 1)) {
      d1 <- tau[i + k - 1] - tau[i]; d2 <- tau[i + k] - tau[i + 1]
      izq <- if (d1 > 0) (t - tau[i]) / d1 * B[, i] else 0
      der <- if (d2 > 0) (tau[i + k] - t) / d2 * B[, i + 1] else 0
      Bn[, i] <- izq + der
    }
    B <- Bn
  }
  if (deriv > 0) for (k in (orden - deriv + 1):orden) {
    Bn <- matrix(0, length(t), n_int - k + 1)
    for (i in 1:(n_int - k + 1)) {
      d1 <- tau[i + k - 1] - tau[i]; d2 <- tau[i + k] - tau[i + 1]
      izq <- if (d1 > 0) B[, i] / d1 else 0
      der <- if (d2 > 0) B[, i + 1] / d2 else 0
      Bn[, i] <- (k - 1) * (izq - der)
    }
    B <- Bn
  }
  stopifnot(ncol(B) == K)
  B
}

cholesky_mano <- function(A) {
  n <- nrow(A); L <- matrix(0, n, n)
  for (j in 1:n) {
    s <- A[j, j] - sum(L[j, seq_len(j - 1)]^2)
    if (s <= 0) stop("no definida positiva")
    L[j, j] <- sqrt(s)
    if (j < n) for (i in (j + 1):n) L[i, j] <- (A[i, j] - sum(L[i, seq_len(j - 1)] * L[j, seq_len(j - 1)])) / L[j, j]
  }
  L
}
adelante <- function(L, B) { B <- as.matrix(B); X <- B * 0
  for (i in 1:nrow(L)) X[i, ] <- (B[i, ] - crossprod(L[i, seq_len(i - 1)], X[seq_len(i - 1), , drop = FALSE])) / L[i, i]; X }
atras <- function(L, B) { B <- as.matrix(B); X <- B * 0; n <- nrow(L)
  for (i in n:1) { j <- if (i < n) (i + 1):n else integer(0)
    X[i, ] <- (B[i, ] - crossprod(L[j, i], X[j, , drop = FALSE])) / L[i, i] }; X }

#Mínimos cuadrados: c = (Φ'Φ)^{-1} Φ'y con Cholesky a mano; h = diag(S)
mco_mano <- function(Phi, Y) {
  L <- cholesky_mano(crossprod(Phi))
  C <- atras(L, adelante(L, crossprod(Phi, Y)))
  Z <- adelante(L, t(Phi))
  list(C = C, h = colSums(Z^2))
}

lee_satelite <- function() {
  ruta <- file.path(CACHE, "R_satelite.rds")
  if (file.exists(ruta)) return(readRDS(ruta))
  suppressMessages(library(terra))
  carpeta <- file.path(CACHE, "sat_tif", "ImagenesSatelitePentadal")
  if (!dir.exists(carpeta)) {
    l <- unzip(file.path(DATOS, "ImagenesSatelitePentadal.zip"), list = TRUE)$Name
    unzip(file.path(DATOS, "ImagenesSatelitePentadal.zip"), files = l[!grepl("/\\._", l) & grepl("\\.tif$", l)],
          exdir = file.path(CACHE, "sat_tif"))
  }
  arch <- sort(list.files(carpeta, pattern = "^ppt.*\\.tif$", full.names = TRUE))
  anio <- as.integer(substr(basename(arch), 4, 7)); pent <- as.integer(substr(basename(arch), 8, 9))
  r <- rast(arch)
  V <- values(r)                                   #celdas (por filas) x archivos
  V[V == -9999] <- NA
  xy <- xyFromCell(r[[1]], 1:ncell(r))
  out <- list(V = V, anio = anio, pent = pent, lon = xy[, 1], lat = xy[, 2])
  saveRDS(out, ruta); out
}

matriz_satelite <- function() {
  s <- lee_satelite()
  anios <- as.integer(names(which(table(s$anio) == 72)))
  X <- NULL; meta <- NULL
  for (a in anios) {
    j <- which(s$anio == a); j <- j[order(s$pent[j])]
    lp <- PENT_LON; if (a %% 4 == 0) lp[12] <- lp[12] + 1
    X <- rbind(X, sweep(s$V[, j], 2, lp, "/"))
    meta <- rbind(meta, data.table(pixel = seq_len(nrow(s$V)) - 1, lon = s$lon, lat = s$lat, year = a))
  }
  list(X = X, meta = meta)
}

fase1 <- function() {
  cat("FASE 1\n")
  z <- readRDS(file.path(CACHE, "R_fase0_matriz.rds"))
  Yd <- z$Y
  sat <- matriz_satelite(); Xs <- sat$X
  cat("curvas estaciones", nrow(Yd), " curvas satélite", nrow(Xs), "\n")

  #Verificación de Φ contra fda::eval.basis (solo comprobación)
  if (requireNamespace("fda", quietly = TRUE)) {
    ver <- rbindlist(lapply(c(6, 12, 20), function(K) {
      bf <- fda::create.bspline.basis(c(0, DIAS), nbasis = K, norder = ORDEN)
      data.table(K = K, max_dif_Phi = max(abs(base_bspline(T_FINO, K) - fda::eval.basis(T_FINO, bf))),
                 max_dif_DPhi = max(abs(base_bspline(T_FINO, K, deriv = 1) - fda::eval.basis(T_FINO, bf, 1))))
    }))
    guarda(ver, "TabR07_verificacion_base_fda.csv")
  }

  #Selección de K: ECM, VC y VCG
  K_grid <- 5:40
  res_s <- rbindlist(lapply(K_grid, function(K) {
    Phi <- base_bspline(T_PENT, K); f <- mco_mano(Phi, t(Xs))
    R <- t(Xs) - Phi %*% f$C; sse <- colSums(R^2); n <- 72
    data.table(K = K, ECM = mean(sse / n), VC = mean(colMeans((R / (1 - f$h))^2)), VCG = mean((sse / n) / ((n - K) / n)^2))
  }))
  res_e <- rbindlist(lapply(K_grid, function(K) {
    P <- base_bspline(T_FINO, K); acum <- c(0, 0, 0); noid <- 0
    for (i in seq_len(nrow(Yd))) {
      obs <- !is.na(Yd[i, ]); Phi <- P[obs, , drop = FALSE]; y <- Yd[i, obs]
      f <- tryCatch(mco_mano(Phi, y), error = function(e) NULL)
      if (is.null(f)) { noid <- noid + 1; next }
      r <- y - Phi %*% f$C; n <- length(y)
      acum <- acum + c(sum(r^2) / n, mean((r / (1 - f$h))^2), (sum(r^2) / n) / ((n - K) / n)^2)
    }
    if (noid > 0) data.table(K = K, ECM = NA, VC = NA, VCG = NA, no_identificables = noid)
    else data.table(K = K, ECM = acum[1] / nrow(Yd), VC = acum[2] / nrow(Yd), VCG = acum[3] / nrow(Yd), no_identificables = 0)
  }))
  K_est <- res_e$K[which.min(res_e$VCG)]; K_sat <- res_s$K[which.min(res_s$VCG)]
  guarda(rbind(res_e[, fuente := "estaciones"], res_s[, `:=`(fuente = "satelite", no_identificables = 0)], fill = TRUE), "TabR08_seleccion_K.csv")
  elec <- data.table(fuente = c("estaciones", "satelite"), K_VCG = c(K_est, K_sat),
                     K_VC = c(res_e$K[which.min(res_e$VC)], res_s$K[which.min(res_s$VC)]),
                     VCG_min = c(min(res_e$VCG, na.rm = TRUE), min(res_s$VCG)))
  guarda(elec, "TabR09_K_elegido.csv"); print(elec)
  png(file.path(FIG, "R_Fig04_seleccion_K.png"), width = 1400, height = 700, res = 130)
  par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))
  for (rr in list(res_e, res_s)) for (cr in c("ECM", "VC", "VCG")) {
    plot(rr$K, rr[[cr]], type = "b", pch = 19, cex = 0.5, xlab = "K", ylab = cr, main = paste(rr$fuente[1], cr))
    abline(v = rr$K[which.min(rr$VCG)], lty = 2)
  }
  dev.off()

  #Ajuste final y verificación de S
  Pe <- base_bspline(T_FINO, K_est)
  C_e <- t(sapply(seq_len(nrow(Yd)), function(i) { obs <- !is.na(Yd[i, ]); mco_mano(Pe[obs, , drop = FALSE], Yd[i, obs])$C }))
  Ps <- base_bspline(T_PENT, K_sat)
  C_s <- t(mco_mano(Ps, t(Xs))$C)
  Ps_c <- base_bspline(T_PENT, K_est)
  C_s_c <- t(mco_mano(Ps_c, t(Xs))$C)
  verS <- function(Phi, nom) {
    L <- cholesky_mano(crossprod(Phi)); Ainv <- atras(L, adelante(L, diag(ncol(Phi))))
    S <- Phi %*% Ainv %*% t(Phi)
    data.table(caso = nom, n = nrow(Phi), K = ncol(Phi), traza = sum(diag(S)), max_SS_menos_S = max(abs(S %*% S - S)),
               max_S_menos_St = max(abs(S - t(S))))
  }
  guarda(rbind(verS(Ps, "satélite"), verS(Pe, "estación completa")), "TabR10_verificacion_S.csv")
  #Comparación con smooth.basis() de fda: la VCG por curva coincide con la calculada a mano
  if (requireNamespace("fda", quietly = TRUE)) {
    bf <- fda::create.bspline.basis(c(0, DIAS), nbasis = K_sat, norder = ORDEN)
    sb <- fda::smooth.basis(T_PENT, t(Xs[1:50, ]), bf)
    Phi <- base_bspline(T_PENT, K_sat); R <- t(Xs[1:50, ]) - Phi %*% t(C_s[1:50, ])
    gcv_mano <- (colSums(R^2) / 72) / ((72 - K_sat) / 72)^2
    guarda(data.table(max_dif_coef = max(abs(sb$fd$coefs - t(C_s[1:50, ]))), max_dif_gcv = max(abs(sb$gcv - gcv_mano))),
           "TabR10b_verificacion_smooth_basis.csv")
  }
  saveRDS(list(C_e = C_e, K_e = K_est, station = z$station, year = z$year, C_s = C_s, K_s = K_sat, C_s_c = C_s_c,
               meta_s = sat$meta), file.path(CACHE, "R_fase1_coef.rds"))
  if (file.exists(file.path(TABPY, "Tab09_K_elegido.csv"))) {
    py <- fread(file.path(TABPY, "Tab09_K_elegido.csv"))
    cmp <- data.table(fuente = elec$fuente, K_R = elec$K_VCG, K_Py = py$K_VCG, VCG_R = elec$VCG_min, VCG_Py = py$VCG_min)
    guarda(cmp, "TabR_comparacion_K.csv"); print(cmp)
  }
  invisible(elec)
}



#===================================================================================
#FASE 2. Análisis exploratorio funcional (fda y fda.usc, como en Script_AnalisisExploratorio.R)
#===================================================================================
lee_oni <- function() {
  o <- fread(file.path(EXT, "oni.ascii.txt"))
  a <- o[, .(oni_anual = mean(ANOM), n = .N), by = .(year = YR)][n == 12]
  a[, fase_enso := ifelse(oni_anual >= 0.5, "El Niño", ifelse(oni_anual <= -0.5, "La Niña", "Neutral"))]
  a[, n := NULL]; a
}

fase2 <- function() {
  cat("FASE 2\n")
  suppressMessages({ library(fda); library(fda.usc) })
  z <- readRDS(file.path(CACHE, "R_fase1_coef.rds"))
  be <- create.bspline.basis(c(0, DIAS), nbasis = z$K_e, norder = ORDEN)
  bs <- create.bspline.basis(c(0, DIAS), nbasis = z$K_s, norder = ORDEN)
  Y_fd <- fd(t(z$C_e), be); X_fd <- fd(t(z$C_s), bs)
  Pe <- base_bspline(T_FINO, z$K_e); Ps <- base_bspline(T_FINO, z$K_s)

  #Media paso a paso vs mean.fd(); varianza vs var.fd()
  media_paso_e <- Pe %*% colMeans(z$C_e); media_paso_s <- Ps %*% colMeans(z$C_s)
  Sc_e <- cov(z$C_e); Sc_s <- cov(z$C_s)
  ver <- data.table(fuente = c("estaciones", "satelite"),
                    max_dif_media = c(max(abs(media_paso_e - eval.fd(T_FINO, mean.fd(Y_fd)))),
                                      max(abs(media_paso_s - eval.fd(T_FINO, mean.fd(X_fd))))),
                    max_dif_cov = c(max(abs(Pe %*% Sc_e %*% t(Pe) - eval.bifd(T_FINO, T_FINO, var.fd(Y_fd)))),
                                    max(abs(Ps %*% Sc_s %*% t(Ps) - eval.bifd(T_FINO, T_FINO, var.fd(X_fd))))))
  guarda(ver, "TabR13_verificacion_media_cov.csv"); print(ver)
  DE_e <- sqrt(diag(Pe %*% Sc_e %*% t(Pe))); DE_s <- sqrt(diag(Ps %*% Sc_s %*% t(Ps)))
  Cor_e <- cor.fd(T_FINO, Y_fd); Cor_s <- cor.fd(T_FINO, X_fd)

  #Derivadas: inicio/fin de temporada con la derivada de la media
  D_e <- eval.fd(T_FINO, deriv.fd(mean.fd(Y_fd), 1)); D_s <- eval.fd(T_FINO, deriv.fd(mean.fd(X_fd), 1))
  temporadas <- function(dm, nom) {
    ceros <- which(diff(sign(dm)) != 0); out <- NULL
    for (c0 in ceros) out <- rbind(out, data.table(fuente = nom, evento = ifelse(dm[c0] > 0, "pico", "valle"), fecha = fecha_txt(T_FINO[c0])))
    for (pk in ceros[dm[ceros] > 0 & ceros > 40 & ceros < 345]) {
      a0 <- max(c(1, ceros[ceros < pk])); b0 <- min(c(DIAS, ceros[ceros > pk]))
      out <- rbind(out, data.table(fuente = nom, evento = c("inicio", "fin"),
                                   fecha = fecha_txt(T_FINO[c(a0 - 1 + which.max(dm[a0:pk]), pk - 1 + which.min(dm[pk:b0]))])))
    }
    out
  }
  guarda(rbind(temporadas(D_e, "estaciones"), temporadas(D_s, "satelite")), "TabR16_temporadas.csv")

  #Profundidad FM (fda.usc) y boxplot funcional MBD (fda::fbplot) en malla de 5 días
  t5 <- seq(2.5, DIAS, by = 5)
  Y5 <- eval.fd(t5, Y_fd)                                            #73 x N
  FM <- depth.FM(fdata(t(Y5), t5, c(0, DIAS)), trim = 0.25)
  BX <- fbplot(Y5, t5, method = "MBD", plot = FALSE)
  An <- Y5 - sapply(seq_len(ncol(Y5)), function(i) rowMeans(Y5[, z$station == z$station[i], drop = FALSE]))
  BXa <- fbplot(An, t5, method = "MBD", plot = FALSE)
  oni <- lee_oni()
  prof <- data.table(station = z$station, year = z$year, FM = FM$dep, MBD = BX$depth, MBD_anom = BXa$depth,
                     atipica_bruta = seq_len(ncol(Y5)) %in% BX$outpoint, atipica_anomalia = seq_len(ncol(Y5)) %in% BXa$outpoint)
  prof <- merge(prof, oni, by = "year", all.x = TRUE)
  guarda(prof[atipica_bruta | atipica_anomalia], "TabR18_atipicas_estaciones.csv")
  #Comparación con la profundidad calculada a mano en Python
  ruta_py <- file.path(CACHE, "fase2_profundidades_estaciones.csv")
  if (file.exists(ruta_py)) {
    py <- fread(ruta_py)
    m <- merge(prof, py[, .(station, year, FM_py = FM, MBD_py = MBD, atip_py = atipica_bruta, atipa_py = atipica_anomalia)], by = c("station", "year"))
    ab <- coef(lm(FM ~ FM_py, data = m))   #fda.usc escala la FM de otra forma: relación lineal exacta
    cmp <- data.table(curvas = nrow(m), cor_FM = cor(m$FM, m$FM_py), FM_R_intercepto = ab[1], FM_R_pendiente = ab[2],
                      max_dif_FM_tras_ajuste_lineal = max(abs(m$FM - (ab[1] + ab[2] * m$FM_py))),
                      cor_MBD = cor(m$MBD, m$MBD_py), max_dif_MBD = max(abs(m$MBD - m$MBD_py)),
                      atipicas_brutas_R = sum(m$atipica_bruta), atipicas_brutas_Py = sum(m$atip_py),
                      atipicas_anom_R = sum(m$atipica_anomalia), atipicas_anom_Py = sum(m$atipa_py))
    guarda(cmp, "TabR_comparacion_profundidad.csv"); print(cmp)
  }
  png(file.path(FIG, "R_Fig08_12_media_boxplot.png"), width = 1500, height = 1000, res = 130)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  plot(T_FINO, media_paso_e, type = "l", lwd = 3, col = "#1f4e79", ylim = c(0, 16), xlab = "Día del año", ylab = "mm/día",
       main = "Media funcional (paso a paso)")
  lines(T_FINO, media_paso_s, lwd = 3, lty = 2, col = "#c8691c")
  legend("topleft", c("Estaciones", "Satélite"), col = c("#1f4e79", "#c8691c"), lty = 1:2, lwd = 3, bty = "n")
  plot(T_FINO, DE_e, type = "l", lwd = 3, col = "#1f4e79", ylim = c(0, max(DE_e, DE_s)), xlab = "Día del año", ylab = "mm/día",
       main = "Desviación estándar funcional"); lines(T_FINO, DE_s, lwd = 3, lty = 2, col = "#c8691c")
  image(T_FINO, T_FINO, Cor_e, zlim = c(-1, 1), col = hcl.colors(40, "Blue-Red 3"), xlab = "s", ylab = "t", main = "Correlación estaciones")
  fbplot(An, t5, method = "MBD", xlab = "Día del año", ylab = "Anomalía (mm/día)", main = "Boxplot funcional (anomalías)")
  dev.off()
  invisible(prof)
}



#===================================================================================
#FASE 3. Comparación estación - satélite
#===================================================================================
fase3 <- function() {
  cat("FASE 3\n")
  z <- readRDS(file.path(CACHE, "R_fase1_coef.rds"))
  cvc <- fread(file.path(DATOS, "CVC_Wheather_Stations_Long_Lat.csv"), encoding = "Latin-1")
  el  <- as.data.table(read_excel(file.path(DATOS, "CatalogoEstacionesElevacion.xlsx")))
  m <- merge(cvc[CODIGO %in% z$station, .(station = CODIGO, lon = Lon, lat)], el[, .(station = as.numeric(codigo), alt)], by = "station", all.x = TRUE)
  m <- m[lon > -77.55 & lon < -75.70 & lat < 5.00 & lat > 3.05]
  m[, `:=`(col = floor((lon + 77.55) / 0.05), fila = floor((5.00 - lat) / 0.05))]
  m[, pixel := fila * 37 + col]
  m[, estaciones_en_pixel := .N, by = pixel]
  m[, zona := ifelse(station >= 5e9, "Pacífico (53/54)", "Cauca (26)")]
  guarda(m, "TabR24_emparejamiento.csv")
  #terra numera las celdas por filas desde la esquina NO: celda (base 0) = fila*37 + col, igual que en Python
  pe <- data.table(station = z$station, year = z$year, ie = seq_along(z$year))
  ps <- data.table(pixel = z$meta_s$pixel, year = z$meta_s$year, is = seq_len(nrow(z$meta_s)))
  pares <- merge(merge(pe, m, by = "station"), ps, by = c("pixel", "year"))
  pares <- merge(pares, lee_oni(), by = "year", all.x = TRUE)
  setorder(pares, station, year)
  Pe <- base_bspline(T_FINO, z$K_e)
  Y <- z$C_e[pares$ie, ] %*% t(Pe); X <- z$C_s_c[pares$is, ] %*% t(Pe)   #misma base K = K_est
  D <- Y - X
  Dst <- rowsum(D, pares$station) / as.vector(table(pares$station))
  ns <- nrow(Dst); dbar <- colMeans(Dst); ee <- apply(Dst, 2, sd) / sqrt(ns)
  intervalos <- function(ind) { r <- rachas(ind); if (!length(r$lon)) return("ninguno")
    paste(fecha_txt(T_FINO[r$ini]), "a", fecha_txt(T_FINO[r$ini + r$lon - 1]), collapse = "; ") }
  #Prueba pareada por cambio de signo (Script_t-Funcional.R), unidad = estación
  B <- 2000
  Tobs <- abs(dbar) / ee
  Tmax <- replicate(B, { s <- sample(c(-1, 1), ns, replace = TRUE); Db <- Dst * s
    max(abs(colMeans(Db)) / (apply(Db, 2, sd) / sqrt(ns))) })
  cmax <- quantile(Tmax, 0.95)
  res <- data.table(pares = nrow(pares), estaciones = ns, pixeles = uniqueN(pares$pixel),
                    d_integrado_mm = sum(dbar), banda_est_mayor = intervalos(dbar - 2 * ee > 0),
                    banda_sat_mayor = intervalos(dbar + 2 * ee < 0), c_max = cmax,
                    p_global = (sum(Tmax >= max(Tobs)) + 1) / (B + 1), intervalos_prueba = intervalos(Tobs > cmax))
  guarda(res, "TabR26_diferencia.csv"); print(res)
  #Correlación cruzada (cor.fd sobre los pares) y su diagonal
  suppressMessages(library(fda))
  be <- create.bspline.basis(c(0, DIAS), nbasis = z$K_e, norder = ORDEN)
  CC <- cor.fd(T_FINO, fd(t(z$C_s_c[pares$is, ]), be), T_FINO, fd(t(z$C_e[pares$ie, ]), be))
  #Comparación por grupo
  grp <- rbindlist(lapply(list(c("zona", "Cauca (26)"), c("zona", "Pacífico (53/54)"), c("fase_enso", "El Niño"),
                               c("fase_enso", "Neutral"), c("fase_enso", "La Niña")), function(g) {
    i <- pares[[g[1]]] == g[2]
    data.table(variable = g[1], grupo = g[2], pares = sum(i), total_est = mean(rowSums(Y[i, ])), total_sat = mean(rowSums(X[i, ])),
               sesgo_rel_mediano = median(100 * (rowSums(X[i, ]) - rowSums(Y[i, ])) / rowSums(Y[i, ])))
  }))
  guarda(grp, "TabR30_por_grupo.csv"); print(grp)
  guarda(data.table(r_diag_media = mean(diag(CC)), r_diag_min = min(diag(CC)), r_diag_max = max(diag(CC)),
                    var_total_est = sum(apply(Y, 2, var)), var_total_sat = sum(apply(X, 2, var))), "TabR28_29_correlacion_covarianza.csv")
  png(file.path(FIG, "R_Fig16_17_diferencia_correlacion.png"), width = 1500, height = 600, res = 130)
  par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
  plot(T_FINO, dbar, type = "l", lwd = 3, col = "navy", ylim = range(c(dbar - 2 * ee, dbar + 2 * ee)),
       xlab = "Día del año", ylab = "mm/día", main = "d(t) = estación - satélite, ± 2 EE (unidad = estación)")
  lines(T_FINO, dbar + 2 * ee, lty = 2, col = "navy"); lines(T_FINO, dbar - 2 * ee, lty = 2, col = "navy"); abline(h = 0, lty = 3)
  plot(T_FINO, diag(CC), type = "l", lwd = 3, ylim = c(0, 1), xlab = "Día del año", ylab = "r(t,t)", main = "Diagonal de la correlación cruzada")
  dev.off()
  if (file.exists(file.path(TABPY, "Tab32_resumen_comparacion.csv"))) {
    py <- fread(file.path(TABPY, "Tab32_resumen_comparacion.csv"))
    guarda(data.table(pares_R = nrow(pares), pares_Py = py$pares, estaciones_R = ns, estaciones_Py = py$estaciones,
                      total_sat_R = mean(rowSums(X)), total_sat_Py = py$total_sat_medio_mm), "TabR_comparacion_fase3.csv")
  }
  invisible(res)
}

for (f in FASES) get(f)()
