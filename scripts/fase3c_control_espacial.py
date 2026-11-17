#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 3c: control del emparejamiento con las coordenadas oficiales.

Responde dos preguntas que el control ingenuo (pixel real vs pixel aleatorio)
no puede responder:

  (1) la correlacion satelite-estacion decae con la distancia? Si toda la
      region comparte el forzamiento de gran escala (ENSO), un pixel al azar
      ya correlaciona alto y la comparacion contra el azar no informa. El
      control correcto es el PERFIL r(distancia).

  (2) el metodo de la fase 3 (pixel de maxima correlacion dentro de 35 km del
      centroide de la subzona) estaba inflado por seleccion post-hoc? Se mide
      sobre LAS MISMAS estaciones, con la coordenada real como arbitro.

Salidas:
  entrega/tablas/fase3c_r_por_distancia.csv
  entrega/tablas/fase3c_seleccion_vs_real.csv
  entrega/figs/fase3c_decaimiento.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(RAIZ, "entrega")
TAB = os.path.join(ENT, "tablas")
FIG = os.path.join(ENT, "figs")
K, PERIODO = 5, 365.0
COLS = [f"c{j+1}" for j in range(K)]
rng = np.random.default_rng(2026)


def base_fourier(t, K, periodo=PERIODO):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


def km(dlon, dlat, lat=4.0):
    return np.sqrt((dlon * 111.32 * np.cos(np.deg2rad(lat))) ** 2 + (dlat * 110.57) ** 2)


t = np.arange(365) + 0.5
Phi = base_fourier(t, K)
LIM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

ce = pd.read_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), dtype={"STATION_ID": str})
cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
cs["PIX"] = cs.FILA.astype(str) + "_" + cs.COL.astype(str)
E = pd.read_csv(os.path.join(TAB, "fase3b_emparejamiento.csv"), dtype={"STATION_ID": str, "PIX": str})
print(f"[0] {len(E)} estaciones con coordenada oficial dentro de la malla")

anios = sorted(set(ce.YEAR) & set(cs.YEAR))
ce = ce[ce.YEAR.isin(anios) & ce.STATION_ID.isin(E.STATION_ID)]
cs = cs[cs.YEAR.isin(anios)]
y0, y1 = min(anios), max(anios)
nm = (y1 - y0 + 1) * 12


def serie_mensual(df, clave, llaves):
    A = np.full((len(llaves), nm), np.nan)
    idx = {k: i for i, k in enumerate(llaves)}
    C = df[COLS].to_numpy(float) @ Phi.T
    kk = df[clave].to_numpy()
    yy = df.YEAR.to_numpy(int)
    for r in range(len(df)):
        i = idx.get(kk[r])
        if i is None:
            continue
        base = (yy[r] - y0) * 12
        for m in range(12):
            A[i, base + m] = C[r, LIM[m]:LIM[m + 1]].mean()
    return A


def anomalias(A):
    out = np.full_like(A, np.nan)
    for m in range(12):
        sl = slice(m, A.shape[1], 12)
        out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
    return out


lst_e = sorted(ce.STATION_ID.unique())
lst_p = sorted(cs.PIX.unique())
print("[1] series mensuales de anomalias")
Ae = anomalias(serie_mensual(ce, "STATION_ID", lst_e))
As = anomalias(serie_mensual(cs, "PIX", lst_p))
pxy = cs[["PIX", "LON", "LAT"]].drop_duplicates().set_index("PIX").loc[lst_p]
PL, PA = pxy.LON.to_numpy(), pxy.LAT.to_numpy()
ip = {p: i for i, p in enumerate(lst_p)}
me = E.set_index("STATION_ID")


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 24:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    s = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / s) if s > 0 else np.nan


# ---------------------------------------------------------------------
print("[2] perfil r(distancia): cada estacion contra TODOS los pixeles")
BINS = [0, 5, 10, 20, 40, 80, 160, 400]
acum = {i: [] for i in range(len(BINS) - 1)}
for i, e in enumerate(lst_e):
    lo, la = float(me.LON_EST[e]), float(me.LAT_EST[e])
    d = km(PL - lo, PA - la, la)
    sel = rng.choice(len(lst_p), min(220, len(lst_p)), replace=False)
    for j in sel:
        r = corr(Ae[i], As[j])
        if np.isfinite(r):
            b = np.searchsorted(BINS, d[j], side="right") - 1
            if 0 <= b < len(BINS) - 1:
                acum[b].append(r)
filas = []
for b in range(len(BINS) - 1):
    v = np.array(acum[b])
    if len(v):
        filas.append(dict(dist_min_km=BINS[b], dist_max_km=BINS[b + 1], n=len(v),
                          r_media=v.mean(), r_sd=v.std(),
                          ic95_inf=v.mean() - 1.96 * v.std() / np.sqrt(len(v)),
                          ic95_sup=v.mean() + 1.96 * v.std() / np.sqrt(len(v))))
Rd = pd.DataFrame(filas)
Rd.to_csv(os.path.join(TAB, "fase3c_r_por_distancia.csv"), index=False)
print(Rd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
cerca = Rd[Rd.dist_max_km <= 10].r_media.mean()
lejos = Rd[Rd.dist_min_km >= 80].r_media.mean()
print(f"\n    r medio a <= 10 km: {cerca:.3f} | a >= 80 km: {lejos:.3f} "
      f"| decaimiento {cerca - lejos:+.3f}")

# ---------------------------------------------------------------------
print("\n[3] seleccion post-hoc (fase 3) vs coordenada real (fase 3b)")
V = os.path.join(ENT, "pares_estacion_pixel.csv.centroides")
if os.path.exists(V):
    viejo = pd.read_csv(V, dtype={"STATION_ID": str})
    vp = viejo.drop_duplicates("STATION_ID").set_index("STATION_ID")
    comun = [e for e in lst_e if e in vp.index]
    print(f"    estaciones en ambos metodos: {len(comun)}")
    f2 = []
    for e in comun:
        i = ip_e = lst_e.index(e)
        pr = str(me.PIX[e])
        # el PIX viejo venia como "FILA_COL"
        pv = str(vp.PIX[e])
        if pr not in ip or pv not in ip:
            continue
        lo, la = float(me.LON_EST[e]), float(me.LAT_EST[e])
        jv = ip[pv]
        f2.append(dict(STATION_ID=e,
                       r_real=corr(Ae[i], As[ip[pr]]),
                       r_seleccionado=corr(Ae[i], As[jv]),
                       dist_real_km=float(me.dist_km[e]),
                       dist_seleccionado_km=float(km(PL[jv] - lo, PA[jv] - la, la)),
                       mismo_pixel=(pr == pv)))
    S = pd.DataFrame(f2).dropna()
    S.to_csv(os.path.join(TAB, "fase3c_seleccion_vs_real.csv"), index=False)
    if len(S):
        print(f"    r con el pixel que CONTIENE la estacion:        {S.r_real.mean():.3f}")
        print(f"    r con el pixel elegido por maxima correlacion:  {S.r_seleccionado.mean():.3f}")
        print(f"    inflacion del metodo anterior:                  "
              f"{S.r_seleccionado.mean() - S.r_real.mean():+.3f}")
        print(f"    distancia real del pixel seleccionado: mediana "
              f"{S.dist_seleccionado_km.median():.1f} km "
              f"(el verdadero esta a {S.dist_real_km.median():.1f} km)")
        print(f"    acerto el pixel correcto en {int(S.mismo_pixel.sum())} de {len(S)} casos "
              f"({100*S.mismo_pixel.mean():.0f} %)")
else:
    print("    no hay version con centroides para comparar")

# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = (Rd.dist_min_km + Rd.dist_max_km) / 2
ax.errorbar(x, Rd.r_media, yerr=1.96 * Rd.r_sd / np.sqrt(Rd.n), marker="o",
            color="tab:blue", capsize=3, label="r medio (IC 95 %)")
ax.fill_between(x, Rd.r_media - Rd.r_sd, Rd.r_media + Rd.r_sd, alpha=0.15,
                color="tab:blue", label="+-1 desviacion")
ax.set_xscale("log")
ax.set_xlabel("distancia estacion-pixel (km, escala log)")
ax.set_ylabel("correlacion de anomalias mensuales")
ax.set_title("Decaimiento espacial de la correlacion satelite-estacion\n"
             "(coordenadas oficiales, 1981-2003)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase3c_decaimiento.png"), dpi=150)
print("\n[4] figura entrega/figs/fase3c_decaimiento.png")
print("\nFASE 3c OK")
