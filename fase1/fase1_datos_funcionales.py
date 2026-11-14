#!/usr/bin/env python3
# =====================================================================
# PROYECTO FINAL ANALITICA DE DATOS 2026-2S
# Fase 1 - Construccion de datos funcionales PASO A PASO (sin librerias)
#
# Nada de smooth.basis / fda / skfda: las bases se construyen a mano
# (recursion de Cox-de Boor para B-splines, serie trigonometrica para
# Fourier), los coeficientes salen de las ecuaciones normales resueltas
# con una factorizacion de Cholesky escrita aqui, y la matriz de
# suavizado se verifica numericamente (S idempotente, tr(S)=K).
# numpy se usa solo como contenedor de arreglos y producto matricial.
#
# Se comparan DOS bases porque la curva anual de lluvia es periodica
# (el 31-dic empalma con el 1-ene): la base B-spline no impone esa
# condicion y produce un artefacto de borde al final del anio, mientras
# la base de Fourier es periodica por construccion. La eleccion se
# decide con VCG, no por preferencia.
#
# Ejecutar:  python3 fase1_datos_funcionales.py
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMILLA = 42
np.random.seed(SEMILLA)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "entrega")
FIG = os.path.join(OUT, "figs")
TAB = os.path.join(OUT, "tablas")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

DOM = (0.0, 365.0)          # dominio anual comun a las dos fuentes
PERIODO = 365.0
ORDEN = 4                   # B-splines cubicos
N_MUESTRA = 60              # curvas usadas para elegir base, K y lambda
TIF = os.path.join(BASE, "repo/Datos/ImagenesSatelitePentadal/ImagenesSatelitePentadal")
LON0, LAT1, RES, NFIL, NCOL = -77.55, 5.00, 0.05, 39, 37   # georreferencia del GeoTIFF

MESES = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]   # mitad de cada mes
LIM_MES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
NOM_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago",
           "Sep", "Oct", "Nov", "Dic"]


# =====================================================================
# 1. Bases funcionales construidas a mano
# =====================================================================
def nodos_completos(n_int, orden=ORDEN, dom=DOM):
    """Nodos con multiplicidad `orden` en los extremos e interiores equiespaciados."""
    interiores = np.linspace(dom[0], dom[1], n_int + 2)[1:-1]
    return np.concatenate([np.full(orden, dom[0]), interiores, np.full(orden, dom[1])])


def base_bspline(t, nodos, orden=ORDEN):
    """Phi (len(t) x K) de B-splines de orden `orden` por recursion de Cox-de Boor."""
    t = np.asarray(t, dtype=float)
    K = len(nodos) - orden
    B = np.zeros((len(t), len(nodos) - 1))
    for i in range(len(nodos) - 1):
        if nodos[i + 1] > nodos[i]:
            B[:, i] = ((t >= nodos[i]) & (t < nodos[i + 1])).astype(float)
    ult = max(i for i in range(len(nodos) - 1) if nodos[i + 1] > nodos[i])
    B[t >= nodos[-1], ult] = 1.0
    for d in range(2, orden + 1):
        Bn = np.zeros((len(t), len(nodos) - d))
        for i in range(len(nodos) - d):
            den1 = nodos[i + d - 1] - nodos[i]
            den2 = nodos[i + d] - nodos[i + 1]
            term = np.zeros(len(t))
            if den1 > 0:
                term += (t - nodos[i]) / den1 * B[:, i]
            if den2 > 0:
                term += (nodos[i + d] - t) / den2 * B[:, i + 1]
            Bn[:, i] = term
        B = Bn
    return B[:, :K]


def base_fourier(t, K, periodo=PERIODO):
    """Phi (len(t) x K) de Fourier: 1, sen(2pi j t/T), cos(2pi j t/T). K impar."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t)]
    for j in range(1, (K - 1) // 2 + 1):
        w = 2.0 * np.pi * j * t / periodo
        cols.append(np.sin(w))
        cols.append(np.cos(w))
    return np.column_stack(cols[:K])


def construir(t, K, tipo):
    """Interfaz unica para las dos bases."""
    if tipo == "bspline":
        return base_bspline(t, nodos_completos(K - ORDEN))
    return base_fourier(t, K)


# =====================================================================
# 2. Minimos cuadrados con Cholesky propia (sin lm(), sin lstsq)
# =====================================================================
def cholesky_propia(A):
    """A = L L^T para A simetrica definida positiva. Algoritmo escrito a mano."""
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("matriz no definida positiva")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def resolver_chol(A, b):
    """Resuelve A x = b con la Cholesky propia (sustitucion adelante y atras)."""
    L = cholesky_propia(A)
    n = A.shape[0]
    B = b.reshape(n, -1)
    Y = np.zeros_like(B)
    for i in range(n):                      # L Y = B
        Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
    X = np.zeros_like(B)
    for i in range(n - 1, -1, -1):          # L^T X = Y
        X[i] = (Y[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
    return X.reshape(b.shape)


def ajustar(Phi, y, R=None, lam=0.0):
    """Coeficientes, matriz de suavizado y diagnosticos del ajuste."""
    G = Phi.T @ Phi
    if R is not None and lam > 0:
        G = G + lam * R
    c = resolver_chol(G, Phi.T @ y)
    A = resolver_chol(G, np.eye(G.shape[0]))     # G^{-1}
    S = Phi @ A @ Phi.T
    yhat = Phi @ c
    h = np.einsum("ij,jk,ik->i", Phi, A, Phi)    # diagonal de S
    n = len(y)
    sse = float(((y - yhat) ** 2).sum())
    nu = float(np.trace(S))
    cv = float(np.mean(((y - yhat) / (1.0 - np.clip(h, None, 1 - 1e-9))) ** 2))
    return dict(c=c, yhat=yhat, S=S, A=A, h=h, sse=sse, nu=nu,
                ecm=sse / n, cv=cv, gcv=n * sse / (n - nu) ** 2,
                sigma2=sse / (n - nu))


def penalizacion(K, tipo, n_malla=3651):
    """R = integral B''(t) B''(t)' dt por diferencias finitas + trapecios."""
    malla = np.linspace(DOM[0], DOM[1], n_malla)
    Bm = construir(malla, K, tipo)
    h = malla[1] - malla[0]
    D2 = (Bm[2:] - 2 * Bm[1:-1] + Bm[:-2]) / h ** 2
    R = (D2.T @ D2) * h
    return (R + R.T) / 2 + 1e-10 * np.eye(R.shape[0])


def eje_meses(ax, marcar_anio=True):
    ax.set_xticks(MESES)
    ax.set_xticklabels(NOM_MES, fontsize=8)
    ax.set_xticks(LIM_MES, minor=True)
    ax.set_xlim(DOM[0], DOM[1])
    ax.set_xlabel("mes del anio")
    if marcar_anio:   # bordes = cambio de anio, donde se juzga el empalme
        for x in DOM:
            ax.axvline(x, color="0.3", ls=":", lw=1)


# =====================================================================
# 3. Curvas de estaciones: datos depurados -> dominio comun
# =====================================================================
print("== cargando estaciones depuradas (salida de la fase 4.1) ==")
est = pd.read_csv(os.path.join(OUT, "estaciones_depuradas.csv"),
                  dtype={"STATION_ID": str})
dias_anio = np.where((est.YEAR % 4 == 0) & ((est.YEAR % 100 != 0) |
                                            (est.YEAR % 400 == 0)), 366, 365)
est["T"] = (est.DOY - 0.5) / dias_anio * 365.0
est["SQRTP"] = np.sqrt(est.VALUE)     # estabiliza la varianza de la lluvia diaria
print(f"   {len(est)} registros | {est.STATION_ID.nunique()} estaciones | "
      f"{est.groupby(['STATION_ID','YEAR']).ngroups} pares estacion-anio")

pares = est.groupby(["STATION_ID", "YEAR"]).size().index.to_list()
idx = np.random.choice(len(pares), size=min(N_MUESTRA, len(pares)), replace=False)
muestra = [pares[i] for i in sorted(idx)]
grupos = {k: v for k, v in est.groupby(["STATION_ID", "YEAR"])}
print(f"   muestra para elegir base/K/lambda: {len(muestra)} pares (semilla {SEMILLA})")


# =====================================================================
# 4. Seleccion de base y de K con ECM, VC y VCG
# =====================================================================
print("\n== seleccion de base y K (estaciones) ==")
rejilla = ([("bspline", K) for K in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 73]] +
           [("fourier", K) for K in [3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51, 73]])
filas = []
for tipo, K in rejilla:
    ecm = cv = gcv = 0.0
    usados = 0
    for key in muestra:
        g = grupos[key]
        Phi = construir(g["T"].to_numpy(), K, tipo)
        try:
            r = ajustar(Phi, g["SQRTP"].to_numpy())
        except np.linalg.LinAlgError:
            continue
        ecm += r["ecm"]; cv += r["cv"]; gcv += r["gcv"]; usados += 1
    if usados:
        filas.append(dict(base=tipo, K=K, ECM=ecm / usados, VC=cv / usados,
                          VCG=gcv / usados, curvas=usados))
        print(f"   {tipo:8s} K={K:3d}  ECM={filas[-1]['ECM']:.4f}  "
              f"VC={filas[-1]['VC']:.4f}  VCG={filas[-1]['VCG']:.4f}")

selK = pd.DataFrame(filas)
selK.to_csv(os.path.join(TAB, "fase1_seleccion_K_estaciones.csv"), index=False)
mejor = selK.loc[selK.VCG.idxmin()]
TIPO_EST, K_EST = str(mejor.base), int(mejor.K)
print(f"   -> base elegida por VCG: {TIPO_EST} con K={K_EST} (VCG={mejor.VCG:.4f})")
for tt in selK.base.unique():
    print(f"      mejor VCG con {tt}: {selK[selK.base == tt].VCG.min():.4f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, col in zip(ax, ["ECM", "VC", "VCG"]):
    for tt, mk in [("bspline", "o-"), ("fourier", "s-")]:
        s = selK[selK.base == tt]
        a.plot(s.K, s[col], mk, ms=4, label=tt)
    a.axvline(K_EST, color="red", ls="--", lw=1)
    a.set_xscale("log"); a.set_xlabel("K (nº de funciones base)")
    a.set_title(col); a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle("Seleccion de base y K - curvas de estaciones (sqrt P)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_seleccion_K_estaciones.png"), dpi=130)
plt.close(fig)


# =====================================================================
# 5. Verificacion numerica de la matriz de suavizado
# =====================================================================
print("\n== verificacion de la matriz de suavizado S ==")
g = grupos[muestra[0]]
Phi = construir(g["T"].to_numpy(), K_EST, TIPO_EST)
r = ajustar(Phi, g["SQRTP"].to_numpy())
S = r["S"]
err_idem = float(np.max(np.abs(S @ S - S)))
tr = float(np.trace(S))
print(f"   max|S@S - S| = {err_idem:.3e}   (debe ser ~0: S es proyeccion)")
print(f"   tr(S) = {tr:.6f}   K = {K_EST}   |tr(S)-K| = {abs(tr-K_EST):.3e}")
print(f"   S simetrica: max|S - S^T| = {float(np.max(np.abs(S - S.T))):.3e}")

R = penalizacion(K_EST, TIPO_EST)
print("\n== seleccion de lambda con penalizacion de rugosidad ==")
fl = []
for lam in [0.0, 1e-2, 1e0, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12]:
    r2 = ajustar(Phi, g["SQRTP"].to_numpy(), R=R, lam=lam)
    fl.append(dict(lam=lam, nu=r2["nu"], ECM=r2["ecm"], VC=r2["cv"], VCG=r2["gcv"]))
    print(f"   lambda={lam:8.0e}  nu=tr(S_l)={r2['nu']:7.3f}  "
          f"ECM={r2['ecm']:.4f}  VC={r2['cv']:.4f}  VCG={r2['gcv']:.4f}")
sel_lam = pd.DataFrame(fl)
sel_lam.to_csv(os.path.join(TAB, "fase1_seleccion_lambda.csv"), index=False)
lam_opt = float(sel_lam.loc[sel_lam.VCG.idxmin(), "lam"])
nu_opt = float(sel_lam.loc[sel_lam.VCG.idxmin(), "nu"])
print(f"   -> lambda elegido por VCG: {lam_opt:g}  (nu = {nu_opt:.3f})")

pd.DataFrame([dict(base=TIPO_EST, K=K_EST, max_abs_SS_menos_S=err_idem, traza_S=tr,
                   dif_traza_K=abs(tr - K_EST),
                   simetria=float(np.max(np.abs(S - S.T))),
                   lambda_opt=lam_opt, nu_lambda_opt=nu_opt)]
             ).to_csv(os.path.join(TAB, "fase1_verificacion_S.csv"), index=False)


# =====================================================================
# 6. Covarianza residual y bandas de confianza punto a punto
# =====================================================================
print("\n== bandas de confianza punto a punto ==")
t_fino = np.linspace(DOM[0], DOM[1], 366)
Phi_f = construir(t_fino, K_EST, TIPO_EST)
var_pt = r["sigma2"] * np.einsum("ij,jk,ik->i", Phi_f, r["A"], Phi_f)
ee = np.sqrt(var_pt)
curva = Phi_f @ r["c"]
sup, inf = curva + 1.96 * ee, curva - 1.96 * ee
print(f"   sigma^2 residual = {r['sigma2']:.4f} | EE medio = {ee.mean():.4f} | "
      f"ancho medio de banda = {float(np.mean(sup-inf)):.4f} (escala sqrt mm)")
print(f"   empalme 31-dic/1-ene: |curva(365)-curva(0)| = "
      f"{abs(float(curva[-1]-curva[0])):.4f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(g["T"], g["SQRTP"], ".", ms=3, color="0.6", label="dato diario (sqrt mm)")
ax.plot(t_fino, curva, "b-", lw=1.8, label=f"curva ajustada ({TIPO_EST}, K={K_EST})")
ax.fill_between(t_fino, inf, sup, color="b", alpha=.2, label="banda punto a punto 95%")
eje_meses(ax); ax.set_ylabel("sqrt(P) [sqrt mm]")
ax.set_title(f"Estacion {muestra[0][0]} - anio {muestra[0][1]}")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_banda_confianza.png"), dpi=130)
plt.close(fig)
pd.DataFrame(dict(t=t_fino, ajuste=curva, ee=ee, inf=inf, sup=sup)).to_csv(
    os.path.join(TAB, "fase1_banda_confianza.csv"), index=False)


# =====================================================================
# 6b. Control climatologico: bimodalidad y empalme de fin de anio
# =====================================================================
print("\n== control de bimodalidad y periodicidad del perfil medio ==")
perfil = est.groupby(est.DOY).SQRTP.mean()
t_perf = perfil.index.to_numpy() - 0.5
y_perf = perfil.to_numpy()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_perf, y_perf, ".", ms=3, color="0.7", label="media diaria (186 estaciones)")
filas_bm = []
combos = [(TIPO_EST, K_EST, "red")]
for tt, kk, cc in [("fourier", 7, "green"), ("bspline", 6, "blue")]:
    if (tt, kk) != (TIPO_EST, K_EST):
        combos.append((tt, kk, cc))
for tt, kk, color in combos:
    Phc = construir(t_perf, kk, tt)
    rc = ajustar(Phc, y_perf)
    cf = construir(t_fino, kk, tt) @ rc["c"]
    pic = [i for i in range(1, len(cf) - 1) if cf[i] > cf[i-1] and cf[i] > cf[i+1]]
    dias = [round(float(t_fino[i]), 1) for i in pic]
    salto = abs(float(cf[-1] - cf[0]))
    filas_bm.append(dict(base=tt, K=kk, n_picos=len(pic), dias_pico=str(dias),
                         salto_fin_anio=salto, ECM=rc["ecm"], VCG=rc["gcv"]))
    print(f"   {tt:8s} K={kk:3d} picos={len(pic)} dias={dias} "
          f"salto_31dic_1ene={salto:.4f} ECM={rc['ecm']:.4f}")
    etq = (f"{tt} K={kk}" + ("" if salto < 1e-6 else
                             f" (no periodica: salto {salto:.2f})"))
    ax.plot(t_fino, cf, "-", color=color, lw=1.6, label=etq)
eje_meses(ax); ax.set_ylabel("sqrt(P) media [sqrt mm]")
ax.set_title("Perfil climatologico medio: bimodalidad y empalme de fin de anio")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_bimodalidad.png"), dpi=130)
plt.close(fig)
pd.DataFrame(filas_bm).to_csv(os.path.join(TAB, "fase1_bimodalidad.csv"), index=False)


# =====================================================================
# 7. Coeficientes de TODAS las curvas de estaciones
# =====================================================================
print("\n== coeficientes funcionales de todas las curvas de estaciones ==")
filas_c = []
for (sid, anio), gg in est.groupby(["STATION_ID", "YEAR"]):
    Ph = construir(gg["T"].to_numpy(), K_EST, TIPO_EST)
    try:
        rr = ajustar(Ph, gg["SQRTP"].to_numpy(), R=R, lam=lam_opt)
    except np.linalg.LinAlgError:
        continue
    filas_c.append([sid, anio] + list(rr["c"]))
coef_est = pd.DataFrame(filas_c, columns=["STATION_ID", "YEAR"] +
                        [f"c{i+1}" for i in range(K_EST)])
coef_est.to_csv(os.path.join(OUT, "coef_estaciones.csv"), index=False)
print(f"   {len(coef_est)} curvas (estacion-anio) con {K_EST} coeficientes "
      f"[base {TIPO_EST}, lambda {lam_opt:g}]")


# =====================================================================
# 8. Fuente satelital: pentadas -> mismo dominio anual
# =====================================================================
print("\n== fuente satelital (GeoTIFF pentadales) ==")
from PIL import Image


def dia_central_pentada(anio, p):
    """Pentada p (1..72) = sub-pentada j del mes m; devuelve dia central y nº de dias."""
    m = (p - 1) // 6 + 1
    j = (p - 1) % 6 + 1
    bis = anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)
    dias_mes = [31, 29 if bis else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d_ini = 5 * (j - 1) + 1
    d_fin = dias_mes[m - 1] if j == 6 else 5 * j
    doy_ini = sum(dias_mes[:m - 1]) + d_ini
    doy_fin = sum(dias_mes[:m - 1]) + d_fin
    return (doy_ini + doy_fin) / 2.0, d_fin - d_ini + 1, sum(dias_mes)


archivos = sorted(f for f in os.listdir(TIF)
                  if f.startswith("ppt") and f.endswith(".tif"))
print(f"   {len(archivos)} rasteres | malla {NFIL}x{NCOL} | res {RES} grados")
cubo = {}
for f in archivos:
    cubo[(int(f[3:7]), int(f[7:9]))] = np.array(Image.open(os.path.join(TIF, f)),
                                                dtype=float)
anios_sat = sorted({a for a, _ in cubo})
completos = [a for a in anios_sat if sum(1 for (y, _) in cubo if y == a) == 72]
print(f"   anios satelitales: {anios_sat[0]}-{anios_sat[-1]} | "
      f"completos (72 pentadas): {len(completos)} | "
      f"incompletos: {sorted(set(anios_sat)-set(completos))}")

fi, co = NFIL // 2, NCOL // 2
print(f"   pixel demo fila={fi} col={co} -> "
      f"lon={LON0+(co+0.5)*RES:.3f} lat={LAT1-(fi+0.5)*RES:.3f}")
anio_demo = 1981
tp, yp = [], []
for p in range(1, 73):
    tc, n_d, dias_tot = dia_central_pentada(anio_demo, p)
    tp.append((tc - 0.5) / dias_tot * 365.0)
    yp.append(cubo[(anio_demo, p)][fi, co] / n_d)    # mm/pentada -> mm/dia
tp, yp = np.array(tp), np.sqrt(np.array(yp))

print("\n== seleccion de base y K (satelite, 72 observaciones/anio) ==")
filas_s = []
for tipo, K in ([("bspline", K) for K in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]] +
                [("fourier", K) for K in [3, 5, 7, 9, 11, 13, 15, 21, 31]]):
    Ph = construir(tp, K, tipo)
    try:
        rs = ajustar(Ph, yp)
    except np.linalg.LinAlgError:
        continue
    filas_s.append(dict(base=tipo, K=K, ECM=rs["ecm"], VC=rs["cv"], VCG=rs["gcv"]))
    print(f"   {tipo:8s} K={K:3d}  ECM={rs['ecm']:.4f}  "
          f"VC={rs['cv']:.4f}  VCG={rs['gcv']:.4f}")
sel_sat = pd.DataFrame(filas_s)
sel_sat.to_csv(os.path.join(TAB, "fase1_seleccion_K_satelite.csv"), index=False)
mj = sel_sat.loc[sel_sat.VCG.idxmin()]
TIPO_SAT, K_SAT = str(mj.base), int(mj.K)
print(f"   -> base elegida por VCG: {TIPO_SAT} con K={K_SAT} (VCG={mj.VCG:.4f})")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, col in zip(ax, ["ECM", "VC", "VCG"]):
    for tt, mk in [("bspline", "o-"), ("fourier", "s-")]:
        s = sel_sat[sel_sat.base == tt]
        a.plot(s.K, s[col], mk, ms=4, label=tt)
    a.axvline(K_SAT, color="red", ls="--", lw=1)
    a.set_xscale("log"); a.set_xlabel("K"); a.set_title(col)
    a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle("Seleccion de base y K - curvas satelitales")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_seleccion_K_satelite.png"), dpi=130)
plt.close(fig)

Ph = construir(tp, K_SAT, TIPO_SAT)
rs = ajustar(Ph, yp)
Ss = rs["S"]
print(f"   verificacion satelite: max|SS-S|={float(np.max(np.abs(Ss@Ss-Ss))):.3e} "
      f"tr(S)={float(np.trace(Ss)):.4f} (K={K_SAT})")

curva_sat = construir(t_fino, K_SAT, TIPO_SAT) @ rs["c"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_fino, curva, "b-", lw=1.8,
        label=f"estacion {muestra[0][0]} ({muestra[0][1]}) - {TIPO_EST} K={K_EST}")
ax.plot(tp, yp, "s", ms=4, color="darkorange", label="pentada satelital (sqrt mm/dia)")
ax.plot(t_fino, curva_sat, "-", color="darkorange", lw=1.8,
        label=f"curva satelital {anio_demo} - {TIPO_SAT} K={K_SAT}")
eje_meses(ax); ax.set_ylabel("sqrt(P) [sqrt mm/dia]")
ax.set_title("Las dos fuentes en el mismo dominio anual [0,365]")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_dos_fuentes.png"), dpi=130)
plt.close(fig)

print("\n== coeficientes satelitales por pixel-anio (anios completos) ==")
filas_sat = []
for anio in completos:
    tpa, factor = [], []
    for p in range(1, 73):
        tc, n_d, dias_tot = dia_central_pentada(anio, p)
        tpa.append((tc - 0.5) / dias_tot * 365.0)
        factor.append(n_d)
    tpa = np.array(tpa); factor = np.array(factor, dtype=float)
    Pha = construir(tpa, K_SAT, TIPO_SAT)
    cubo_anio = np.stack([cubo[(anio, p)] for p in range(1, 73)])
    Y = np.sqrt(cubo_anio / factor[:, None, None]).reshape(72, -1)
    C = resolver_chol(Pha.T @ Pha, Pha.T @ Y)
    for j in range(C.shape[1]):
        f_, c_ = divmod(j, NCOL)
        filas_sat.append([anio, f_, c_, LON0 + (c_ + .5) * RES, LAT1 - (f_ + .5) * RES]
                         + list(C[:, j]))
coef_sat = pd.DataFrame(filas_sat, columns=["YEAR", "FILA", "COL", "LON", "LAT"] +
                        [f"c{i+1}" for i in range(K_SAT)])
coef_sat.to_csv(os.path.join(OUT, "coef_satelite.csv"), index=False)
print(f"   {len(coef_sat)} curvas pixel-anio con {K_SAT} coeficientes [base {TIPO_SAT}]")

print("\n===== FASE 1 TERMINADA =====")
print(f"estaciones: base {TIPO_EST} K={K_EST} lambda={lam_opt:g} | "
      f"satelite: base {TIPO_SAT} K={K_SAT}")
print(f"numpy {np.__version__} | pandas {pd.__version__} | "
      f"matplotlib {matplotlib.__version__} | semilla {SEMILLA}")
