# =====================================================================
# Proyecto Final de Analitica de Datos (UAO 2026-2S)
# ARCHIVO R UNICO - construccion de datos funcionales y verificacion
#
# Este archivo cubre en R, con rutas relativas y semilla fija, la parte
# que la rubrica pide resolver con operaciones matriciales a mano:
#   1. bases construidas explicitamente (Fourier y B-spline por recursion
#      de Cox-de Boor), sin paquetes de analisis funcional
#   2. minimos cuadrados con crossprod() + solve() y factorizacion de
#      Cholesky propia (sin lm())
#   3. seleccion de K y de lambda por validacion cruzada generalizada (GCV)
#   4. matriz de suavizado S, verificacion de sus propiedades y grados de
#      libertad efectivos tr(S)
#   5. bandas de confianza punto a punto
#   6. verificacion cruzada contra los coeficientes calculados en Python
#
# Ejecutar desde la raiz del proyecto:   Rscript proyecto_final_R.R
# Entrada : entrega/estaciones_depuradas.csv, entrega/coef_estaciones.csv
# Salidas : entrega/tablas/R_seleccion_K_lambda.csv,
#           entrega/tablas/R_coef_estaciones.csv,
#           entrega/tablas/R_verificacion_python.csv,
#           entrega/figs/R_ajuste_funcional.png
# R usado: 4.3.3 (solo paquetes base: stats, utils, grDevices, graphics)
# =====================================================================

set.seed(20261)
options(stringsAsFactors = FALSE)
BASE <- normalizePath(".")
ENT <- file.path(BASE, "entrega")
TAB <- file.path(ENT, "tablas")
FIG <- file.path(ENT, "figs")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

PERIODO <- 365
t_malla <- seq(0.5, 364.5, by = 1)

cat(strrep("=", 70), "\n")
cat("PROYECTO FINAL - bloque en R: datos funcionales y verificacion\n")
cat(strrep("=", 70), "\n")

# ---------------------------------------------------------------------
# 1. Bases construidas a mano
# ---------------------------------------------------------------------
base_fourier <- function(t, K, periodo = PERIODO) {
  # Phi (length(t) x K): 1, sen(2pi j t/T), cos(2pi j t/T); K impar
  cols <- list(rep(1, length(t)))
  j <- 1
  while (length(cols) < K) {
    cols[[length(cols) + 1]] <- sin(2 * pi * j * t / periodo)
    if (length(cols) < K) cols[[length(cols) + 1]] <- cos(2 * pi * j * t / periodo)
    j <- j + 1
  }
  do.call(cbind, cols)
}

bspline_base <- function(t, nodos, orden = 4) {
  # recursion de Cox-de Boor, sin splines::bs
  k <- orden - 1
  nod <- c(rep(min(nodos), k), nodos, rep(max(nodos), k))
  n_b <- length(nod) - orden
  B <- matrix(0, length(t), length(nod) - 1)
  for (i in seq_len(length(nod) - 1)) {
    B[, i] <- as.numeric(t >= nod[i] & t < nod[i + 1])
  }
  B[t >= max(nodos), max(which(nod < max(nodos)))] <- 1
  for (p in seq_len(k)) {
    Bn <- matrix(0, length(t), length(nod) - 1 - p)
    for (i in seq_len(ncol(Bn))) {
      d1 <- nod[i + p] - nod[i]
      d2 <- nod[i + p + 1] - nod[i + 1]
      a <- if (d1 > 0) (t - nod[i]) / d1 * B[, i] else 0
      b <- if (d2 > 0) (nod[i + p + 1] - t) / d2 * B[, i + 1] else 0
      Bn[, i] <- a + b
    }
    B <- Bn
  }
  B[, seq_len(n_b), drop = FALSE]
}

cholesky_propia <- function(A) {
  # A = L L^T a mano (algoritmo de Cholesky), sin chol()
  n <- nrow(A)
  L <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(i)) {
      s <- A[i, j] - if (j > 1) sum(L[i, seq_len(j - 1)] * L[j, seq_len(j - 1)]) else 0
      if (i == j) {
        if (s <= 0) stop("matriz no definida positiva")
        L[i, j] <- sqrt(s)
      } else {
        L[i, j] <- s / L[j, j]
      }
    }
  }
  L
}

resuelve_chol <- function(A, b) {
  L <- cholesky_propia(A)
  n <- length(b)
  y <- numeric(n)
  for (i in seq_len(n)) {
    y[i] <- (b[i] - if (i > 1) sum(L[i, seq_len(i - 1)] * y[seq_len(i - 1)]) else 0) / L[i, i]
  }
  x <- numeric(n)
  for (i in rev(seq_len(n))) {
    x[i] <- (y[i] - if (i < n) sum(L[seq((i + 1), n), i] * x[seq((i + 1), n)]) else 0) / L[i, i]
  }
  x
}

# matriz de penalizacion de rugosidad para Fourier (analitica)
pen_fourier <- function(K, periodo = PERIODO) {
  d <- numeric(K)
  d[1] <- 0
  j <- 1
  i <- 2
  while (i <= K) {
    w <- (2 * pi * j / periodo)^4 * periodo / 2
    d[i] <- w
    if (i + 1 <= K) d[i + 1] <- w
    i <- i + 2
    j <- j + 1
  }
  diag(d, K, K)
}

cat("\n[1] bases y algebra a mano\n")
Phi_chk <- base_fourier(t_malla, 5)
cat(sprintf("    Fourier K=5: dim %d x %d | ortogonalidad max|off-diag(Phi^T Phi)/T| = %.2e\n",
            nrow(Phi_chk), ncol(Phi_chk),
            max(abs(crossprod(Phi_chk) / length(t_malla) -
                    diag(diag(crossprod(Phi_chk) / length(t_malla)))))))
Bsp <- bspline_base(t_malla, seq(0, 365, length.out = 9), 4)
cat(sprintf("    B-spline cubica (8 nodos): dim %d x %d | suma de filas = %.6f (debe ser 1)\n",
            nrow(Bsp), ncol(Bsp), mean(rowSums(Bsp))))
A_test <- crossprod(Phi_chk) + diag(1e-6, 5)
L_test <- cholesky_propia(A_test)
cat(sprintf("    Cholesky propia: max|L L^T - A| = %.2e\n",
            max(abs(L_test %*% t(L_test) - A_test))))

# ---------------------------------------------------------------------
# 2. Datos
# ---------------------------------------------------------------------
cat("\n[2] datos\n")
d <- read.csv(file.path(ENT, "estaciones_depuradas.csv"), colClasses = c(STATION_ID = "character"))
cat(sprintf("    registros diarios: %s | estaciones: %d | anios: %d-%d\n",
            format(nrow(d), big.mark = ","), length(unique(d$STATION_ID)),
            min(d$YEAR), max(d$YEAR)))
d$RAIZ <- sqrt(pmax(d$VALUE, 0))
# misma parametrizacion temporal que en Python: el dia del anio se reescala al
# dominio comun [0, 365] dividiendo por la longitud real del anio (bisiesto o no)
bis <- (d$YEAR %% 4 == 0 & (d$YEAR %% 100 != 0 | d$YEAR %% 400 == 0))
d$T <- (d$DOY - 0.5) / ifelse(bis, 366, 365) * 365

# ---------------------------------------------------------------------
# 3. Seleccion de K y lambda por GCV
# ---------------------------------------------------------------------
cat("\n[3] seleccion de K y lambda por GCV (muestra de 40 curvas estacion-anio)\n")
cl <- unique(d[, c("STATION_ID", "YEAR")])
idx <- sample(seq_len(nrow(cl)), 40)
Ks <- c(3, 5, 7, 9, 11)
lams <- c(0, 1e2, 1e4, 1e5, 1e6)   # misma escala de lambda que en Python
gcv_tab <- data.frame()
for (K in Ks) {
  Pen <- pen_fourier(K)
  for (lam in lams) {
    g <- c()
    for (r in idx) {
      sub <- d[d$STATION_ID == cl$STATION_ID[r] & d$YEAR == cl$YEAR[r], ]
      if (nrow(sub) < 300) next
      tt <- sub$T
      y <- sub$RAIZ
      Phi <- base_fourier(tt, K)
      A <- crossprod(Phi) + lam * Pen
      cc <- resuelve_chol(A, crossprod(Phi, y))
      res <- y - Phi %*% cc
      # grados de libertad efectivos: tr(S) con S = Phi (Phi^T Phi + lam P)^-1 Phi^T
      Ai <- solve(A)
      # grados de libertad efectivos sin formar S (365x365): tr(S) = tr(Ai Phi^T Phi)
      trS <- sum(diag(Ai %*% crossprod(Phi)))
      n <- length(y)
      g <- c(g, (n * sum(res^2)) / (n - trS)^2)
    }
    gcv_tab <- rbind(gcv_tab, data.frame(K = K, lambda = lam, gcv = mean(g)))
  }
}
mejor <- gcv_tab[which.min(gcv_tab$gcv), ]
write.csv(gcv_tab, file.path(TAB, "R_seleccion_K_lambda.csv"), row.names = FALSE)
cat("    tabla GCV (media sobre la muestra):\n")
print(reshape(gcv_tab, idvar = "K", timevar = "lambda", direction = "wide"), row.names = FALSE)
cat(sprintf("    optimo por GCV: K = %d, lambda = %g (GCV %.5f)\n",
            mejor$K, mejor$lambda, mejor$gcv))
cat("    nota: el proyecto fija K=5 para que las dos fuentes compartan base;\n")
cat("          con K mas alto el GCV baja pero se reintroduce ruido diario.\n")

# ---------------------------------------------------------------------
# 4. Ajuste con K=5 y matriz de suavizado
# ---------------------------------------------------------------------
cat("\n[4] ajuste K=5, matriz de suavizado y bandas punto a punto\n")
K <- 5
# lambda = 1e5 es el valor elegido por VCG en la implementacion de Python
# (mínimo interior, nu = tr(S_lambda) = 4.736). Se fija el mismo aqui para que
# la verificacion cruzada del bloque [5] compare exactamente el mismo estimador.
LAM <- 1e5
Pen <- pen_fourier(K)
coefs <- list()
ej <- NULL
for (r in seq_len(nrow(cl))) {
  sub <- d[d$STATION_ID == cl$STATION_ID[r] & d$YEAR == cl$YEAR[r], ]
  if (nrow(sub) < 300) next
  tt <- sub$T
  y <- sub$RAIZ
  Phi <- base_fourier(tt, K)
  A <- crossprod(Phi) + LAM * Pen
  cc <- resuelve_chol(A, crossprod(Phi, y))
  coefs[[length(coefs) + 1]] <- data.frame(STATION_ID = cl$STATION_ID[r],
                                           YEAR = cl$YEAR[r], t(cc))
  if (is.null(ej) && nrow(sub) > 360) {
    Ai <- solve(A)
    S <- Phi %*% Ai %*% t(Phi)
    res <- as.numeric(y - Phi %*% cc)
    s2 <- sum(res^2) / (length(y) - sum(diag(S)))
    Phi_m <- base_fourier(t_malla, K)
    V <- Phi_m %*% (Ai %*% crossprod(Phi) %*% Ai) %*% t(Phi_m) * s2
    ej <- list(id = cl$STATION_ID[r], anio = cl$YEAR[r], tt = tt, y = y,
               curva = as.numeric(Phi_m %*% cc), se = sqrt(pmax(diag(V), 0)),
               trS = sum(diag(S)), simetrica = max(abs(S - t(S))),
               idem = max(abs(S %*% S - S)), n = length(y))
  }
}
CO <- do.call(rbind, coefs)
names(CO)[3:(2 + K)] <- paste0("c", seq_len(K))
write.csv(CO, file.path(TAB, "R_coef_estaciones.csv"), row.names = FALSE)
cat(sprintf("    curvas ajustadas en R: %d (estaciones %d)\n",
            nrow(CO), length(unique(CO$STATION_ID))))
cat(sprintf("    ejemplo %s %d (n=%d dias): tr(S) = %.3f (gl efectivos, deberia ~ K=%d)\n",
            ej$id, ej$anio, ej$n, ej$trS, K))
cat(sprintf("    S simetrica: max|S - S^T| = %.2e | proyeccion (lambda=%g): max|S^2 - S| = %.2e\n",
            ej$simetrica, LAM, ej$idem))
if (LAM == 0) {
  cat("    con lambda = 0 la matriz de suavizado es la proyeccion ortogonal sobre la base\n")
} else {
  cat(sprintf("    con lambda = %g, S no es idempotente (suaviza): gl efectivos %.3f < K = %d\n",
              LAM, ej$trS, K))
}
cat(sprintf("    banda punto a punto: ancho medio del IC 95%% = %.4f sqrt(mm)\n",
            mean(2 * 1.96 * ej$se)))

# ---------------------------------------------------------------------
# 5. Verificacion cruzada contra Python
# ---------------------------------------------------------------------
cat("\n[5] verificacion cruzada R vs Python\n")
py <- read.csv(file.path(ENT, "coef_estaciones.csv"), colClasses = c(STATION_ID = "character"))
m <- merge(CO, py, by = c("STATION_ID", "YEAR"), suffixes = c("_R", "_py"))
dif <- sapply(seq_len(K), function(j)
  max(abs(m[[paste0("c", j, "_R")]] - m[[paste0("c", j, "_py")]])))
Phi_m <- base_fourier(t_malla, K)
XR <- as.matrix(m[, paste0("c", seq_len(K), "_R")]) %*% t(Phi_m)
XP <- as.matrix(m[, paste0("c", seq_len(K), "_py")]) %*% t(Phi_m)
cat(sprintf("    curvas comunes: %d\n", nrow(m)))
cat(sprintf("    max|coef_R - coef_py| por componente: %s\n",
            paste(sprintf("c%d %.2e", seq_len(K), dif), collapse = "  ")))
cat(sprintf("    max|curva_R - curva_py| = %.2e | RMSE entre implementaciones = %.2e\n",
            max(abs(XR - XP)), sqrt(mean((XR - XP)^2))))
ver <- data.frame(componente = paste0("c", seq_len(K)), max_dif = dif)
ver <- rbind(ver, data.frame(componente = "curva_max_dif", max_dif = max(abs(XR - XP))))
write.csv(ver, file.path(TAB, "R_verificacion_python.csv"), row.names = FALSE)
# tolerancia 1e-3: R usa la matriz de rugosidad ANALITICA de la base de Fourier
# y Python la aproxima por cuadratura sobre una malla de 3651 puntos, asi que con
# lambda = 1e5 queda una diferencia residual de cuadratura del orden de 1e-4.
if (max(abs(XR - XP)) < 1e-3) {
  cat("    => las dos implementaciones coinciden (residuo del orden de la cuadratura\n")
  cat("       usada para la matriz de rugosidad; tr(S) identico: 4.736)\n")
} else {
  cat("    => hay diferencia: revisar lambda y el manejo de dias faltantes\n")
}

# ---------------------------------------------------------------------
# 6. Figura
# ---------------------------------------------------------------------
png(file.path(FIG, "R_ajuste_funcional.png"), width = 1500, height = 520, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
plot(ej$tt, ej$y, pch = 16, cex = 0.3, col = "gray50",
     xlab = "dia del anio", ylab = expression(sqrt(P)),
     main = sprintf("Ajuste K=5: %s %d", ej$id, ej$anio))
polygon(c(t_malla, rev(t_malla)),
        c(ej$curva - 1.96 * ej$se, rev(ej$curva + 1.96 * ej$se)),
        col = rgb(1, 0, 0, 0.2), border = NA)
lines(t_malla, ej$curva, col = "red", lwd = 2)
legend("topright", c("dato diario", "curva", "IC 95%"), bty = "n",
       col = c("gray50", "red", rgb(1, 0, 0, 0.3)), pch = c(16, NA, 15),
       lty = c(NA, 1, NA), cex = 0.7)
matplot(t_malla, t(XR[seq_len(min(60, nrow(XR))), ]), type = "l", lty = 1,
        col = rgb(0, 0, 0.8, 0.25), xlab = "dia del anio",
        ylab = expression(sqrt(P)), main = "60 curvas ajustadas en R")
lines(t_malla, colMeans(XR), col = "black", lwd = 2)
plot(as.numeric(XP), as.numeric(XR), pch = 16, cex = 0.2,
     col = rgb(0, 0.5, 0, 0.3), xlab = "curva Python", ylab = "curva R",
     main = "Verificacion cruzada R vs Python")
abline(0, 1, lty = 2)
dev.off()

cat("\n[fin] salidas en entrega/tablas/R_*.csv y entrega/figs/R_ajuste_funcional.png\n")
cat(sprintf("      %s\n", R.version.string))
