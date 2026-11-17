#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto Final Analitica de Datos (UAO) 2026-2S
FASE 2 - Analisis exploratorio funcional (15 %)

Entra:  entrega/coef_estaciones.csv  (3.567 curvas estacion-anio, base Fourier K=5)
        entrega/coef_satelite.csv    (46.176 curvas pixel-anio,   base Fourier K=5)
        aux/oni.txt                  (anomalia Nino 3.4 mensual, NOAA/PSL)

Sale:   entrega/figs/fase2_*.png
        entrega/tablas/fase2_*.csv
        entrega/log_fase2.txt   (via tee al ejecutar)

Reglas de la rubrica respetadas:
  * media funcional calculada PASO A PASO (acumulador explicito, sin np.mean)
    y verificada contra: (a) la identidad analitica Phi(t) c_barra,
    (b) numpy.mean, (c) skfda.representation.grid.FDataGrid.mean() si skfda esta.
  * varianza / desviacion / covarianza / correlacion funcional a mano.
  * derivadas analiticas de la base de Fourier (no diferencias finitas).
  * profundidad Fraiman-Muniz y MBD implementadas a mano y verificadas contra
    skfda (ModifiedBandDepth / IntegratedDepth) cuando esta disponible.
  * boxplot funcional (Sun & Genton 2011) con atipicas listadas por estacion y anio.
  * todo el conjunto 1981-2003 (estaciones) y 1981-2012 (satelite): sin submuestras.

Ejecutar:  ./venv/bin/python fase2_eda_funcional.py | tee entrega/log_fase2.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMILLA = 20262
np.random.seed(SEMILLA)

BASE = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(BASE, "entrega")
FIG = os.path.join(ENT, "figs")
TAB = os.path.join(ENT, "tablas")
AUX = os.path.join(BASE, "aux")
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

PERIODO = 365.0
K = 5                      # base elegida en la Fase 1 por VCG (Fourier, 2 armonicos)
LIM_MES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

print("=" * 78)
print("FASE 2 - ANALISIS EXPLORATORIO FUNCIONAL")
print(f"numpy {np.__version__} | pandas {pd.__version__} | matplotlib {matplotlib.__version__}")
try:
    import skfda
    HAY_SKFDA = True
    print(f"skfda {skfda.__version__} (solo para VERIFICAR lo calculado a mano)")
except Exception as e:                                        # pragma: no cover
    HAY_SKFDA = False
    print(f"skfda no disponible ({e}); se verifica contra numpy y contra la identidad analitica")
print("=" * 78)


# =====================================================================
# 0. Base de Fourier y su derivada (analiticas)
# =====================================================================
def base_fourier(t, K=K, periodo=PERIODO):
    """Phi (len(t) x K): 1, sen(2pi j t/T), cos(2pi j t/T). Igual que en la Fase 1."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for j in range(1, (K - 1) // 2 + 1):
        w = 2.0 * np.pi * j * t / periodo
        cols.append(np.sin(w))
        cols.append(np.cos(w))
    return np.column_stack(cols[:K])


def base_fourier_d1(t, K=K, periodo=PERIODO):
    """Derivada primera de la base (analitica)."""
    t = np.asarray(t, dtype=float)
    cols = [np.zeros_like(t)]
    for j in range(1, (K - 1) // 2 + 1):
        wj = 2.0 * np.pi * j / periodo
        cols.append(wj * np.cos(wj * t))
        cols.append(-wj * np.sin(wj * t))
    return np.column_stack(cols[:K])


def integra(y, x):
    """Regla del trapecio (compatible con numpy 1.x y 2.x)."""
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def eje_meses(ax, ylab=None):
    ax.set_xticks([(LIM_MES[i] + LIM_MES[i + 1]) / 2 for i in range(12)])
    ax.set_xticklabels(MESES, fontsize=8)
    for x in LIM_MES[1:-1]:
        ax.axvline(x, color="0.9", lw=0.6, zorder=0)
    ax.set_xlim(0, 365)
    if ylab:
        ax.set_ylabel(ylab)


# =====================================================================
# 1. Carga de los datos funcionales de la Fase 1
# =====================================================================
ce = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"))
cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
COLS = [f"c{i+1}" for i in range(K)]

t_dia = np.arange(0.5, 365.0, 1.0)            # 365 dias, centros de dia
t_pen = np.arange(2.5, 365.0, 5.0)            # 73 pentadas (malla nativa del satelite)
Phi_dia = base_fourier(t_dia)
Phi_pen = base_fourier(t_pen)
dPhi_dia = base_fourier_d1(t_dia)

Ce = ce[COLS].to_numpy()
Cs = cs[COLS].to_numpy()
Xe = Ce @ Phi_dia.T                            # 3.567 x 365   (escala sqrt mm/dia)
Xs_p = Cs @ Phi_pen.T                          # 46.176 x 73
print(f"\n[1] curvas estacion-anio: {Xe.shape[0]} ({ce.STATION_ID.nunique()} estaciones, "
      f"{ce.YEAR.min()}-{ce.YEAR.max()})")
print(f"    curvas pixel-anio:    {Xs_p.shape[0]} ({cs.groupby(['FILA','COL']).ngroups} pixeles, "
      f"{cs.YEAR.min()}-{cs.YEAR.max()})")
print("    las dos fuentes comparten base (Fourier K=5) y dominio t en [0,365]")

# acumulado anual OBSERVADO (dato depurado de la fase 4.1), para validar y para el
# analisis interanual: la curva suavizada en escala sqrt(P) subestima el acumulado.
obs = (pd.read_csv(os.path.join(ENT, "estaciones_depuradas.csv"),
                   usecols=["STATION_ID", "YEAR", "VALUE"])
       .groupby(["STATION_ID", "YEAR"], as_index=False)
       .agg(acum_obs_mm=("VALUE", "sum"), dias=("VALUE", "size")))
ce = ce.merge(obs, on=["STATION_ID", "YEAR"], how="left")
acum_rec = (Xe ** 2).sum(axis=1)                      # integral de la curva ajustada al cuadrado
ce["acum_rec_mm"] = acum_rec
sesgo = float(np.mean(acum_rec - ce.acum_obs_mm))
fac = float(ce.acum_obs_mm.sum() / acum_rec.sum())
print(f"    acumulado observado medio {ce.acum_obs_mm.mean():.0f} mm/anio vs "
      f"reconstruido {acum_rec.mean():.0f} mm/anio (sesgo {sesgo:.0f} mm, "
      f"{100 * sesgo / ce.acum_obs_mm.mean():.1f} %; factor de escala {fac:.2f})")
print("    -> el suavizado con K=5 elimina la varianza diaria de alta frecuencia; la curva")
print("       describe la FORMA estacional, no el total. Para totales se usa el observado.")


# =====================================================================
# 2. Media funcional PASO A PASO + verificaciones
# =====================================================================
def media_funcional_paso_a_paso(X):
    """Media punto a punto con acumulador explicito. Sin np.mean."""
    n, m = X.shape
    acum = np.zeros(m, dtype=float)
    for i in range(n):                      # recorre curva por curva
        fila = X[i]
        for k in range(m):                  # acumula punto a punto
            acum[k] += fila[k]
    return acum / n


print("\n[2] media funcional")
mu_e = media_funcional_paso_a_paso(Xe)
# (a) identidad analitica: la media de las curvas es la curva de los coeficientes medios
c_barra = Ce.sum(axis=0) / Ce.shape[0]
mu_e_analitica = Phi_dia @ c_barra
d_a = float(np.max(np.abs(mu_e - mu_e_analitica)))
# (b) numpy
mu_e_np = Xe.mean(axis=0)
d_b = float(np.max(np.abs(mu_e - mu_e_np)))
print(f"    estaciones: max|manual - Phi(t)c_barra| = {d_a:.3e}")
print(f"    estaciones: max|manual - numpy.mean|    = {d_b:.3e}")

d_c = np.nan
if HAY_SKFDA:
    from skfda.representation.grid import FDataGrid
    fd_e = FDataGrid(data_matrix=Xe, grid_points=t_dia)
    mu_e_sk = np.asarray(fd_e.mean().data_matrix).ravel()
    d_c = float(np.max(np.abs(mu_e - mu_e_sk)))
    print(f"    estaciones: max|manual - skfda.mean|    = {d_c:.3e}")

mu_s = media_funcional_paso_a_paso(Xs_p)
d_s = float(np.max(np.abs(mu_s - (Phi_pen @ (Cs.sum(axis=0) / Cs.shape[0])))))
print(f"    satelite:   max|manual - Phi(t)c_barra| = {d_s:.3e}")

pd.DataFrame([dict(fuente="estaciones", n=Xe.shape[0], dif_vs_analitica=d_a,
                   dif_vs_numpy=d_b, dif_vs_skfda=d_c),
              dict(fuente="satelite", n=Xs_p.shape[0], dif_vs_analitica=d_s,
                   dif_vs_numpy=float(np.max(np.abs(mu_s - Xs_p.mean(axis=0)))),
                   dif_vs_skfda=np.nan)]
             ).to_csv(os.path.join(TAB, "fase2_verificacion_media.csv"), index=False)


# =====================================================================
# 3. Varianza, desviacion, covarianza y correlacion funcional (a mano)
# =====================================================================
def var_funcional(X, mu):
    n, m = X.shape
    acum = np.zeros(m)
    for i in range(n):
        d = X[i] - mu
        acum += d * d
    return acum / (n - 1)


def cov_funcional(X, mu):
    """C(s,t) = 1/(n-1) sum_i (x_i(s)-mu(s))(x_i(t)-mu(t)) por producto matricial."""
    D = X - mu
    return (D.T @ D) / (X.shape[0] - 1)


print("\n[3] varianza, covarianza y correlacion funcional")
var_e = var_funcional(Xe, mu_e)
sd_e = np.sqrt(var_e)
var_s = var_funcional(Xs_p, mu_s)
sd_s = np.sqrt(var_s)

# covarianza en malla pentadal (73x73) para que sea legible y comparable entre fuentes
Xe_p = Ce @ Phi_pen.T
mu_e_p = media_funcional_paso_a_paso(Xe_p)
Ce_cov = cov_funcional(Xe_p, mu_e_p)
Cs_cov = cov_funcional(Xs_p, mu_s)
dg_e = np.sqrt(np.diag(Ce_cov))
dg_s = np.sqrt(np.diag(Cs_cov))
Re = Ce_cov / np.outer(dg_e, dg_e)
Rs = Cs_cov / np.outer(dg_s, dg_s)
print(f"    traza de la covarianza (varianza total integrada): estaciones "
      f"{np.trace(Ce_cov) * 5:.2f} | satelite {np.trace(Cs_cov) * 5:.2f}  (escala sqrt mm)^2")
print(f"    desv. estandar funcional: estaciones min {sd_e.min():.3f} (dia "
      f"{int(t_dia[sd_e.argmin()])}) max {sd_e.max():.3f} (dia {int(t_dia[sd_e.argmax()])})")
print(f"    correlacion ene-abr (dia 15 vs 105): estaciones {Re[3, 21]:.3f} | satelite {Rs[3, 21]:.3f}")
print(f"    correlacion abr-oct (dia 105 vs 300): estaciones {Re[21, 60]:.3f} | satelite {Rs[21, 60]:.3f}")

pd.DataFrame(dict(dia=t_dia, media_est=mu_e, sd_est=sd_e,
                  media_sat=np.interp(t_dia, t_pen, mu_s),
                  sd_sat=np.interp(t_dia, t_pen, sd_s))
             ).to_csv(os.path.join(TAB, "fase2_media_sd.csv"), index=False)
pd.DataFrame(Re, index=t_pen.astype(int), columns=t_pen.astype(int)
             ).to_csv(os.path.join(TAB, "fase2_correlacion_estaciones.csv"))
pd.DataFrame(Rs, index=t_pen.astype(int), columns=t_pen.astype(int)
             ).to_csv(os.path.join(TAB, "fase2_correlacion_satelite.csv"))

# --- figura: media +- sd de las dos fuentes
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(t_dia, mu_e - sd_e, mu_e + sd_e, color="tab:blue", alpha=0.18,
                label="estaciones: media $\\pm$ 1 sd")
ax.plot(t_dia, mu_e, color="tab:blue", lw=2, label="media funcional estaciones")
ax.fill_between(t_pen, mu_s - sd_s, mu_s + sd_s, color="tab:orange", alpha=0.18,
                label="satelite: media $\\pm$ 1 sd")
ax.plot(t_pen, mu_s, color="tab:orange", lw=2, label="media funcional satelite")
eje_meses(ax, "$\\sqrt{P}$ [$\\sqrt{mm}$/dia]")
ax.set_title("Fase 2 - media y desviacion funcional por fuente (1981-2003 / 1981-2012)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_media_sd.png"), dpi=130)
plt.close(fig)

# --- figura: covarianza y correlacion
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for j, (M, nom) in enumerate([(Ce_cov, "estaciones"), (Cs_cov, "satelite")]):
    im = axs[0, j].imshow(M, origin="lower", extent=[0, 365, 0, 365], cmap="viridis")
    axs[0, j].set_title(f"covarianza funcional - {nom}")
    fig.colorbar(im, ax=axs[0, j], fraction=0.046)
for j, (M, nom) in enumerate([(Re, "estaciones"), (Rs, "satelite")]):
    im = axs[1, j].imshow(M, origin="lower", extent=[0, 365, 0, 365], cmap="coolwarm",
                          vmin=-1, vmax=1)
    axs[1, j].set_title(f"correlacion funcional - {nom}")
    fig.colorbar(im, ax=axs[1, j], fraction=0.046)
for a in axs.ravel():
    a.set_xlabel("dia del anio")
    a.set_ylabel("dia del anio")
fig.suptitle("Fase 2 - estructura de segundo orden (malla pentadal, 73 x 73)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_covarianza_correlacion.png"), dpi=130)
plt.close(fig)


# =====================================================================
# 4. Derivadas: inicio y fin de las temporadas de lluvia
# =====================================================================
print("\n[4] derivadas e inicio/fin de temporadas")
dmu_e = dPhi_dia @ c_barra
dmu_s = base_fourier_d1(t_dia) @ (Cs.sum(axis=0) / Cs.shape[0])


def cruces(f, t, subida=True):
    """Dias donde f cruza el cero (interpolacion lineal entre muestras)."""
    s = np.sign(f)
    out = []
    for i in range(len(f) - 1):
        if s[i] == 0 or s[i] * s[i + 1] < 0:
            if (f[i + 1] > f[i]) == subida:
                x = t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])
                out.append(float(x))
    return out


picos_e = cruces(dmu_e, t_dia, subida=False)     # mu' pasa de + a -  -> maximos
valles_e = cruces(dmu_e, t_dia, subida=True)     # mu' pasa de - a +  -> minimos
picos_s = cruces(dmu_s, t_dia, subida=False)
valles_s = cruces(dmu_s, t_dia, subida=True)


def a_fecha(d):
    d = float(d) % 365
    for i in range(12):
        if LIM_MES[i] <= d < LIM_MES[i + 1]:
            return f"{int(d - LIM_MES[i]) + 1:02d}-{MESES[i]}"
    return "31-dic"


print(f"    estaciones: maximos en dias {[round(p, 1) for p in picos_e]} "
      f"({[a_fecha(p) for p in picos_e]})")
print(f"    estaciones: minimos en dias {[round(p, 1) for p in valles_e]} "
      f"({[a_fecha(p) for p in valles_e]})")
print(f"    satelite:   maximos en dias {[round(p, 1) for p in picos_s]} "
      f"({[a_fecha(p) for p in picos_s]})")
print(f"    satelite:   minimos en dias {[round(p, 1) for p in valles_s]} "
      f"({[a_fecha(p) for p in valles_s]})")

# inicio / fin de temporada: cruces de la media funcional con su nivel medio anual
niv_e = integra(mu_e, t_dia) / 365.0
niv_s = integra(mu_s, t_pen) / 365.0
ini_e = cruces(mu_e - niv_e, t_dia, subida=True)
fin_e = cruces(mu_e - niv_e, t_dia, subida=False)
ini_s = cruces(mu_s - niv_s, t_pen, subida=True)
fin_s = cruces(mu_s - niv_s, t_pen, subida=False)
print(f"    nivel medio anual: estaciones {niv_e:.3f} | satelite {niv_s:.3f} (sqrt mm/dia)")
print(f"    temporadas estaciones: inicios {[a_fecha(x) for x in ini_e]} | "
      f"finales {[a_fecha(x) for x in fin_e]}")
print(f"    temporadas satelite:   inicios {[a_fecha(x) for x in ini_s]} | "
      f"finales {[a_fecha(x) for x in fin_s]}")

filas = []
for nom, pic, val, ini, fin in [("estaciones", picos_e, valles_e, ini_e, fin_e),
                                ("satelite", picos_s, valles_s, ini_s, fin_s)]:
    for tipo, lst in [("maximo", pic), ("minimo", val),
                      ("inicio_temporada", ini), ("fin_temporada", fin)]:
        for x in lst:
            filas.append(dict(fuente=nom, evento=tipo, dia=round(float(x), 2), fecha=a_fecha(x)))
pd.DataFrame(filas).to_csv(os.path.join(TAB, "fase2_temporadas_derivada.csv"), index=False)

fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
axs[0].plot(t_dia, mu_e, color="tab:blue", lw=2, label="media estaciones")
axs[0].plot(t_pen, mu_s, color="tab:orange", lw=2, label="media satelite")
axs[0].axhline(niv_e, color="tab:blue", ls=":", lw=1)
axs[0].axhline(niv_s, color="tab:orange", ls=":", lw=1)
for x in ini_e + fin_e:
    axs[0].axvline(x, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
eje_meses(axs[0], "$\\sqrt{P}$")
axs[0].legend(fontsize=8)
axs[0].set_title("Media funcional y cruces con el nivel medio anual (inicio/fin de temporada)")
axs[1].plot(t_dia, dmu_e, color="tab:blue", lw=1.6, label="$\\mu'(t)$ estaciones")
axs[1].plot(t_dia, dmu_s, color="tab:orange", lw=1.6, label="$\\mu'(t)$ satelite")
axs[1].axhline(0, color="0.3", lw=0.8)
eje_meses(axs[1], "derivada [$\\sqrt{mm}$/dia$^2$]")
axs[1].legend(fontsize=8)
axs[1].set_title("Derivada analitica de la media funcional")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_derivadas.png"), dpi=130)
plt.close(fig)


# =====================================================================
# 5. Profundidad funcional: Fraiman-Muniz y MBD (a mano) + boxplot funcional
# =====================================================================
def rangos_por_columna(X):
    """Rango 1..n de cada curva en cada t (promedio en empates), vectorizado."""
    n = X.shape[0]
    orden = np.argsort(X, axis=0, kind="stable")
    rk = np.empty_like(X, dtype=float)
    filas = np.arange(1, n + 1, dtype=float)[:, None]
    np.put_along_axis(rk, orden, np.broadcast_to(filas, X.shape).copy(), axis=0)
    return rk


def pesos_trapecio(t):
    """Pesos de la regla del trapecio normalizados a 1 (integral sobre |T|)."""
    t = np.asarray(t, float)
    w = np.zeros_like(t)
    w[1:-1] = (t[2:] - t[:-2]) / 2.0
    w[0] = (t[1] - t[0]) / 2.0
    w[-1] = (t[-1] - t[-2]) / 2.0
    return w / w.sum()


def profundidad_fraiman_muniz(X, t):
    """D_i = (1/|T|) * integral ( 1 - |F_t(x_i(t)) - 1/2| ) dt  (ec. 3, Fraiman-Muniz 2001)."""
    n = X.shape[0]
    F = rangos_por_columna(X) / n
    return (1.0 - np.abs(F - 0.5)) @ pesos_trapecio(t)


def profundidad_mbd(X, t):
    """Modified Band Depth con J=2 (Lopez-Pintado & Romo 2009), via rangos.
    La proporcion de pares (j,k) cuya banda contiene a x_i(t) es
    [(r-1)(n-r) + (n-1)] / C(n,2) con r el rango de x_i(t)."""
    n = X.shape[0]
    r = rangos_por_columna(X)
    prop = ((r - 1.0) * (n - r) + (n - 1.0)) / (n * (n - 1.0) / 2.0)
    return prop @ pesos_trapecio(t)


print("\n[5] profundidad funcional y boxplot funcional")
fm_e = profundidad_fraiman_muniz(Xe, t_dia)
mbd_e = profundidad_mbd(Xe, t_dia)
print(f"    estaciones: FM  min {fm_e.min():.4f} max {fm_e.max():.4f} "
      f"| MBD min {mbd_e.min():.4f} max {mbd_e.max():.4f}")
print(f"    correlacion de Spearman FM vs MBD: "
      f"{np.corrcoef(rangos_por_columna(fm_e[:, None]).ravel(), rangos_por_columna(mbd_e[:, None]).ravel())[0, 1]:.4f}")

try:
    from skfda.exploratory.depth import ModifiedBandDepth, IntegratedDepth
    HAY_DEPTH = HAY_SKFDA
except Exception as e:                                        # pragma: no cover
    HAY_DEPTH = False
    print(f"    aviso: skfda.exploratory.depth no importable ({type(e).__name__}); "
          "la verificacion de profundidad se omite")

if HAY_DEPTH:
    sub = np.sort(np.random.choice(Xe.shape[0], 400, replace=False))   # verificacion en submuestra
    fd_sub = FDataGrid(data_matrix=Xe[sub], grid_points=t_dia)
    mbd_sk = ModifiedBandDepth()(fd_sub)
    fm_sk = IntegratedDepth()(fd_sub)
    mbd_mio = profundidad_mbd(Xe[sub], t_dia)
    fm_mio = profundidad_fraiman_muniz(Xe[sub], t_dia)
    e1 = float(np.max(np.abs(mbd_mio - mbd_sk)))
    e2 = float(np.max(np.abs(fm_mio - fm_sk)))
    print(f"    verificacion contra skfda (n=400): max|MBD_manual - MBD_skfda| = {e1:.3e}")
    print(f"    verificacion contra skfda (n=400): max|FM_manual  - FM_skfda|  = {e2:.3e}")
    pd.DataFrame([dict(medida="MBD", n_verif=len(sub), max_dif=e1),
                  dict(medida="Fraiman-Muniz", n_verif=len(sub), max_dif=e2)]
                 ).to_csv(os.path.join(TAB, "fase2_verificacion_profundidad.csv"), index=False)


def boxplot_funcional(X, prof, factor=1.5):
    """Sun & Genton (2011): mediana = curva mas profunda; banda central = 50 % mas
    profundas; valla = banda central +- factor * ancho de la banda central."""
    n = X.shape[0]
    ordr = np.argsort(-prof)
    central = ordr[: max(2, int(np.ceil(n / 2)))]
    inf = X[central].min(axis=0)
    sup = X[central].max(axis=0)
    ancho = sup - inf
    valla_inf = inf - factor * ancho
    valla_sup = sup + factor * ancho
    fuera = (X < valla_inf[None, :]) | (X > valla_sup[None, :])
    return dict(mediana=X[ordr[0]], idx_mediana=int(ordr[0]), inf=inf, sup=sup,
                valla_inf=valla_inf, valla_sup=valla_sup,
                atipicas=np.where(fuera.any(axis=1))[0], central=central)


bp_e = boxplot_funcional(Xe, mbd_e)
atip = bp_e["atipicas"]
print(f"    atipicas (estaciones): {len(atip)} de {Xe.shape[0]} curvas "
      f"({100 * len(atip) / Xe.shape[0]:.2f} %)")
med_id = ce.iloc[bp_e["idx_mediana"]]
print(f"    curva mediana (MBD maxima): estacion {int(med_id.STATION_ID)} anio {int(med_id.YEAR)}")

det = ce[["STATION_ID", "YEAR"]].copy()
det["MBD"] = mbd_e
det["FraimanMuniz"] = fm_e
det["acum_obs_mm"] = ce.acum_obs_mm.to_numpy()
det["acum_rec_mm"] = acum_rec
det["atipica_boxplot"] = False
det.loc[det.index[atip], "atipica_boxplot"] = True
det.sort_values("MBD").to_csv(os.path.join(TAB, "fase2_profundidad_estaciones.csv"), index=False)
det[det.atipica_boxplot].sort_values(["YEAR", "STATION_ID"]).to_csv(
    os.path.join(TAB, "fase2_atipicas_estacion_anio.csv"), index=False)
print("    top 5 menos profundas (MBD):")
for _, r in det.nsmallest(5, "MBD").iterrows():
    print(f"      estacion {int(r.STATION_ID)} anio {int(r.YEAR)}  MBD={r.MBD:.4f}  "
          f"acumulado observado={r.acum_obs_mm:7.0f} mm")
anio_at = det[det.atipica_boxplot].YEAR.value_counts().sort_index()
print(f"    atipicas por anio (top): {anio_at.nlargest(6).to_dict()}")

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.fill_between(t_dia, bp_e["valla_inf"], bp_e["valla_sup"], color="0.92",
                label="valla 1.5 x banda central")
ax.fill_between(t_dia, bp_e["inf"], bp_e["sup"], color="tab:blue", alpha=0.30,
                label="banda central 50 % (MBD)")
for i in atip[:60]:
    ax.plot(t_dia, Xe[i], color="tab:red", lw=0.6, alpha=0.55)
ax.plot(t_dia, bp_e["mediana"], color="black", lw=2, label="mediana funcional (MBD max)")
ax.plot(t_dia, mu_e, color="tab:green", lw=1.4, ls="--", label="media funcional")
eje_meses(ax, "$\\sqrt{P}$ [$\\sqrt{mm}$/dia]")
ax.set_title(f"Fase 2 - boxplot funcional, estaciones 1981-2003 ({len(atip)} curvas atipicas)")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_boxplot_funcional.png"), dpi=130)
plt.close(fig)

# el mismo analisis para el satelite (malla pentadal, por costo computacional justificado)
mbd_s = profundidad_mbd(Xs_p, t_pen)
bp_s = boxplot_funcional(Xs_p, mbd_s)
print(f"    atipicas (satelite): {len(bp_s['atipicas'])} de {Xs_p.shape[0]} "
      f"({100 * len(bp_s['atipicas']) / Xs_p.shape[0]:.2f} %)")
ds = cs[["YEAR", "FILA", "COL", "LON", "LAT"]].copy()
ds["MBD"] = mbd_s
ds["atipica_boxplot"] = False
ds.loc[ds.index[bp_s["atipicas"]], "atipica_boxplot"] = True
ds[ds.atipica_boxplot].to_csv(os.path.join(TAB, "fase2_atipicas_satelite.csv"), index=False)


# =====================================================================
# 6. Variabilidad interanual y ENSO
# =====================================================================
print("\n[6] variabilidad interanual y ENSO")
oni = None
ruta_oni = os.path.join(AUX, "oni.txt")
if os.path.exists(ruta_oni):
    reg = []
    with open(ruta_oni) as f:
        for ln in f.read().splitlines():
            p = ln.split()
            if len(p) == 13 and p[0].isdigit():
                reg.append([int(p[0])] + [float(x) for x in p[1:]])
    oni = pd.DataFrame(reg, columns=["YEAR"] + [f"m{i}" for i in range(1, 13)])
    oni = oni.replace(-99.99, np.nan).replace(-9.99, np.nan)
    # fase ENSO del anio de lluvia: promedio DJF (dic del anio previo, ene-feb del anio)
    oni = oni.set_index("YEAR")
    djf = {}
    for y in oni.index:
        if y - 1 in oni.index:
            djf[y] = np.nanmean([oni.loc[y - 1, "m12"], oni.loc[y, "m1"], oni.loc[y, "m2"]])
    djf = pd.Series(djf, name="ONI_DJF")
    anual = oni[[f"m{i}" for i in range(1, 13)]].mean(axis=1)   # ONI medio del anio civil
    print(f"    serie Nino 3.4 cargada: {int(oni.index.min())}-{int(oni.index.max())}")
else:
    djf = pd.Series(dtype=float, name="ONI_DJF")
    anual = pd.Series(dtype=float, name="ONI_ANUAL")
    print("    aviso: aux/oni.txt no encontrado; se omite la clasificacion ENSO")


def fase_enso(v, u=0.5):
    if np.isnan(v):
        return "sin dato"
    return "El Nino" if v >= u else ("La Nina" if v <= -u else "Neutro")


# media funcional por anio (estaciones) y su acumulado
res_anio = []
mu_anio = {}
for y, g in ce.groupby("YEAR"):
    Xy = g[COLS].to_numpy() @ Phi_dia.T
    muy = media_funcional_paso_a_paso(Xy)
    mu_anio[int(y)] = muy
    res_anio.append(dict(YEAR=int(y), n_curvas=len(g),
                         acum_mm=float(g.acum_obs_mm.mean()),
                         acum_rec_mm=float((muy ** 2).sum()),
                         norma_L2_desvio=float(np.sqrt(integra((muy - mu_e) ** 2, t_dia))),
                         ONI_DJF=float(djf.get(int(y), np.nan)),
                         ONI_ANUAL=float(anual.get(int(y), np.nan)),
                         fase_ENSO=fase_enso(float(djf.get(int(y), np.nan))),
                         fase_ENSO_anual=fase_enso(float(anual.get(int(y), np.nan)))))
ra = pd.DataFrame(res_anio)
ra.to_csv(os.path.join(TAB, "fase2_interanual_estaciones.csv"), index=False)
print(f"    acumulado anual medio: {ra.acum_mm.mean():.0f} mm "
      f"(min {ra.acum_mm.min():.0f} en {int(ra.loc[ra.acum_mm.idxmin(), 'YEAR'])}, "
      f"max {ra.acum_mm.max():.0f} en {int(ra.loc[ra.acum_mm.idxmax(), 'YEAR'])})")
if ra.ONI_DJF.notna().any():
    rho = ra[["acum_mm", "ONI_DJF"]].dropna().corr(method="spearman").iloc[0, 1]
    rho2 = ra[["acum_mm", "ONI_ANUAL"]].dropna().corr(method="spearman").iloc[0, 1]
    print(f"    correlacion de Spearman acumulado anual vs ONI DJF: {rho:.3f} "
          f"(n={ra.ONI_DJF.notna().sum()} anios)")
    print(f"    correlacion de Spearman acumulado anual vs ONI medio anual: {rho2:.3f} "
          "(clasificacion mas pertinente para el anio civil de lluvia)")
    print("    medias por fase ENSO segun ONI medio anual (mm/anio):")
    for f, g in ra.groupby("fase_ENSO_anual"):
        print(f"      {f:9s} n={len(g):2d}  {g.acum_mm.mean():7.0f} mm  "
              f"anios {sorted(g.YEAR.tolist())}")
    ra.groupby("fase_ENSO_anual").agg(n=("YEAR", "size"), acum_medio_mm=("acum_mm", "mean"),
                                desvio_L2_medio=("norma_L2_desvio", "mean")
                                ).to_csv(os.path.join(TAB, "fase2_enso_resumen.csv"))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
col = {"El Nino": "tab:red", "La Nina": "tab:blue", "Neutro": "0.6", "sin dato": "0.8"}
for _, r in ra.iterrows():
    axs[0].plot(t_dia, mu_anio[int(r.YEAR)], lw=1.0, alpha=0.75, color=col[r.fase_ENSO_anual])
axs[0].plot(t_dia, mu_e, color="black", lw=2.2, label="media 1981-2003")
for f, c in col.items():
    if (ra.fase_ENSO_anual == f).any():
        axs[0].plot([], [], color=c, lw=1.4, label=f)
eje_meses(axs[0], "$\\sqrt{P}$")
axs[0].legend(fontsize=8)
axs[0].set_title("Media funcional por anio, coloreada por fase ENSO (ONI medio anual)")
axs[1].bar(ra.YEAR, ra.acum_mm, color=[col[f] for f in ra.fase_ENSO_anual])
axs[1].axhline(ra.acum_mm.mean(), color="black", ls="--", lw=1)
axs[1].set_xlabel("anio")
axs[1].set_ylabel("acumulado anual medio [mm]")
axs[1].set_title("Acumulado anual reconstruido y fase ENSO")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_interanual_enso.png"), dpi=130)
plt.close(fig)


# =====================================================================
# 7. Orografia: gradiente espacial con la fuente satelital (tiene coordenadas)
# =====================================================================
print("\n[7] gradiente orografico (fuente satelital, 1.443 pixeles)")
cs2 = cs.copy()
cs2["acum_mm"] = ((cs2[COLS].to_numpy() @ Phi_pen.T) ** 2).sum(axis=1) * 5.0
pix = cs2.groupby(["FILA", "COL", "LON", "LAT"], as_index=False).acum_mm.mean()
pix.to_csv(os.path.join(TAB, "fase2_acumulado_por_pixel.csv"), index=False)
print(f"    acumulado anual por pixel: min {pix.acum_mm.min():.0f} mm | "
      f"mediana {pix.acum_mm.median():.0f} mm | max {pix.acum_mm.max():.0f} mm")
oeste = pix[pix.LON <= -76.8].acum_mm.mean()
valle = pix[(pix.LON > -76.5) & (pix.LON < -75.9)].acum_mm.mean()
print(f"    media vertiente pacifica (lon <= -76.8): {oeste:.0f} mm | "
      f"valle interandino (-76.5 a -75.9): {valle:.0f} mm | razon {oeste / valle:.2f}")
rho_lon = pix[["LON", "acum_mm"]].corr(method="spearman").iloc[0, 1]
print(f"    Spearman acumulado vs longitud: {rho_lon:.3f} (negativo = mas lluvia al occidente)")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
sc = axs[0].scatter(pix.LON, pix.LAT, c=pix.acum_mm, s=14, cmap="YlGnBu")
fig.colorbar(sc, ax=axs[0], label="acumulado anual [mm]")
axs[0].set_xlabel("longitud")
axs[0].set_ylabel("latitud")
axs[0].set_title("Acumulado anual medio por pixel (1981-2012)")
for lo, hi, nom, c in [(-77.6, -76.8, "vertiente pacifica", "tab:green"),
                       (-76.8, -76.5, "cordillera occidental", "tab:olive"),
                       (-76.5, -75.9, "valle del rio Cauca", "tab:orange"),
                       (-75.9, -75.7, "flanco oriental", "tab:brown")]:
    s = cs2[(cs2.LON >= lo) & (cs2.LON < hi)]
    if len(s):
        m = media_funcional_paso_a_paso(s[COLS].to_numpy() @ Phi_pen.T)
        axs[1].plot(t_pen, m, lw=1.8, color=c, label=f"{nom} (n={s.groupby(['FILA','COL']).ngroups})")
eje_meses(axs[1], "$\\sqrt{P}$")
axs[1].legend(fontsize=8)
axs[1].set_title("Media funcional por franja longitudinal (orografia)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_orografia.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 8. FPCA (componentes principales funcionales) A MANO
#    Problema de autovalores generalizado V W b = lambda b con
#    W = integral Phi(t) Phi(t)^T dt (matriz de Gram de la base) y
#    V = covarianza de los coeficientes. Se simetriza con W^{1/2}.
#    Verificacion contra skfda.preprocessing.dim_reduction.FPCA.
# =====================================================================
print("\n[8] FPCA funcional (implementado a mano, verificado contra skfda)")

w_t = pesos_trapecio(t_dia) * (t_dia[-1] - t_dia[0])      # pesos sin normalizar
Phi_e = base_fourier(t_dia, K)                            # (n_t x K)
W = Phi_e.T @ (Phi_e * w_t[:, None])                      # Gram K x K
Cc = ce[COLS].to_numpy(float) - ce[COLS].to_numpy(float).mean(axis=0)
V = (Cc.T @ Cc) / (len(Cc) - 1)

lamW, Uw = np.linalg.eigh(W)                              # W simetrica def. positiva
Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T                   # W^{1/2}
Whi = Uw @ np.diag(1.0 / np.sqrt(lamW)) @ Uw.T            # W^{-1/2}
lam, U = np.linalg.eigh(Wh @ V @ Wh)
ordn = np.argsort(lam)[::-1]
lam, U = lam[ordn], U[:, ordn]
B = Whi @ U                                               # coef. de las eigenfunciones
xi = Phi_e @ B                                            # eigenfunciones (n_t x K)
for j in range(K):                                        # signo canonico: maximo positivo
    if xi[np.argmax(np.abs(xi[:, j])), j] < 0:
        xi[:, j] *= -1
        B[:, j] *= -1
scores = Cc @ W @ B                                       # scores = <x-mu, xi_j>
vexp = lam / lam.sum()
print("    varianza explicada: " + "  ".join(
    f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)) +
    f"  (acumulada PC1-PC2 {100*vexp[:2].sum():.1f}%)")
print(f"    ortonormalidad: max|<xi_i,xi_j> - delta_ij| = "
      f"{np.abs((xi.T @ (xi * w_t[:, None])) - np.eye(K)).max():.2e}")
print(f"    varianza de los scores vs autovalores: max dif = "
      f"{np.abs(scores.var(axis=0, ddof=1) - lam).max():.2e}")
if HAY_SKFDA:
    from skfda.preprocessing.dim_reduction import FPCA
    fd_e = FDataGrid(Xe, t_dia)
    fp = FPCA(n_components=3).fit(fd_e)
    sc_sk = fp.transform(fd_e)
    print("    skfda varianza explicada: " + "  ".join(
        f"PC{j+1} {100*v:.1f}%" for j, v in enumerate(fp.explained_variance_ratio_)))
    for j in range(3):
        print(f"    |corr(scores PC{j+1} manual, skfda)| = "
              f"{abs(np.corrcoef(scores[:, j], sc_sk[:, j])[0, 1]):.6f}")

pd.DataFrame(dict(PC=[f"PC{j+1}" for j in range(K)], autovalor=lam,
                  var_explicada=vexp, var_acumulada=np.cumsum(vexp))
             ).to_csv(os.path.join(TAB, "fase2_fpca_varianza.csv"), index=False)
pd.DataFrame(dict(STATION_ID=ce.STATION_ID, YEAR=ce.YEAR,
                  PC1=scores[:, 0], PC2=scores[:, 1], PC3=scores[:, 2],
                  acum_obs_mm=ce.acum_obs_mm, MBD=mbd_e)
             ).to_csv(os.path.join(TAB, "fase2_fpca_scores.csv"), index=False)

fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
for j in range(3):
    axs[0].plot(t_dia, xi[:, j], lw=1.6, label=f"PC{j+1} ({100*vexp[j]:.1f}%)")
axs[0].axhline(0, color="gray", lw=0.6)
eje_meses(axs[0], "eigenfuncion")
axs[0].legend(fontsize=8)
axs[0].set_title("Componentes principales funcionales")
sd1 = np.sqrt(lam[0])
axs[1].plot(t_dia, mu_e, color="black", lw=2, label="media")
axs[1].plot(t_dia, mu_e + sd1 * xi[:, 0], color="tab:red", lw=1.2, ls="--",
            label="media + sd$\\cdot$PC1")
axs[1].plot(t_dia, mu_e - sd1 * xi[:, 0], color="tab:blue", lw=1.2, ls="--",
            label="media - sd$\\cdot$PC1")
eje_meses(axs[1], "$\\sqrt{P}$")
axs[1].legend(fontsize=8)
axs[1].set_title("Efecto de PC1 sobre la media")
sc = axs[2].scatter(scores[:, 0], scores[:, 1], c=ce.acum_obs_mm, s=6,
                    cmap="viridis", alpha=0.6)
fig.colorbar(sc, ax=axs[2], label="acumulado observado [mm/anio]")
axs[2].set_xlabel(f"PC1 ({100*vexp[0]:.1f}%)")
axs[2].set_ylabel(f"PC2 ({100*vexp[1]:.1f}%)")
axs[2].set_title("Plano de scores")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_fpca.png"), dpi=130)
plt.close(fig)
print(f"    Spearman PC1 vs acumulado observado: "
      f"{pd.Series(scores[:, 0]).corr(ce.acum_obs_mm, method='spearman'):.3f}")

# =====================================================================
# 9. Validacion: curva ajustada vs dato observado
# =====================================================================
print("\n[9] validacion de la representacion funcional contra el dato observado")
err = acum_rec - ce.acum_obs_mm.to_numpy()
rmse = float(np.sqrt(np.mean(err ** 2)))
r_p = float(np.corrcoef(acum_rec, ce.acum_obs_mm.to_numpy())[0, 1])
ro = ce[["acum_obs_mm", "acum_rec_mm"]].corr(method="spearman").iloc[0, 1]
print(f"    n={len(ce)} curvas | RMSE {rmse:.0f} mm | r Pearson {r_p:.4f} | "
      f"rho Spearman {ro:.4f} | sesgo {err.mean():.0f} mm")
print("    la correlacion alta con sesgo sistematico confirma que la curva preserva el")
print("    ordenamiento y la forma (lo que usan las fases 2-4) y no el total acumulado.")
pd.DataFrame([dict(n=len(ce), rmse_mm=rmse, r_pearson=r_p, rho_spearman=float(ro),
                   sesgo_mm=float(err.mean()), factor_escala=fac)]
             ).to_csv(os.path.join(TAB, "fase2_validacion_acumulados.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
axs[0].scatter(ce.acum_obs_mm, acum_rec, s=5, alpha=0.3, color="tab:blue")
lim = [0, float(ce.acum_obs_mm.max()) * 1.02]
axs[0].plot(lim, lim, color="black", lw=1, ls="--", label="identidad")
axs[0].plot(lim, [x / fac for x in lim], color="tab:red", lw=1,
            label=f"ajuste 1/{fac:.2f}")
axs[0].set_xlabel("acumulado anual observado [mm]")
axs[0].set_ylabel("integral de la curva ajustada al cuadrado [mm]")
axs[0].set_title(f"Validacion (r={r_p:.3f})")
axs[0].legend(fontsize=8)
peor = det.nsmallest(3, "MBD")
for _, r in peor.iterrows():
    k = ce.index[(ce.STATION_ID == r.STATION_ID) & (ce.YEAR == r.YEAR)][0]
    axs[1].plot(t_dia, Xe[k], lw=1.3, label=f"{int(r.STATION_ID)} / {int(r.YEAR)}")
axs[1].plot(t_dia, bp_e["mediana"], color="black", lw=2, label="mediana funcional")
eje_meses(axs[1], "$\\sqrt{P}$")
axs[1].legend(fontsize=7)
axs[1].set_title("Curvas menos profundas (MBD minima) vs mediana")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase2_validacion.png"), dpi=130)
plt.close(fig)

print("\n[fin] figuras en entrega/figs/fase2_*.png, tablas en entrega/tablas/fase2_*.csv")
print(f"semilla {SEMILLA} | python {sys.version.split()[0]}")
