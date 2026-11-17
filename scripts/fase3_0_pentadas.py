#!/usr/bin/env python3
# =====================================================================
# FASE 3.0 - Reajuste de las curvas de estacion en resolucion PENTADAL
#
# Motivo (defecto de comparabilidad detectado al pasar a la fase 3):
# la fase 1 ajusto las estaciones sobre sqrt(P) de valores DIARIOS y el
# producto satelital sobre sqrt(P) de PROMEDIOS PENTADALES. La raiz es
# concava, asi que por la desigualdad de Jensen
#       media_5dias( sqrt(P) )  <=  sqrt( media_5dias(P) ),
# y la brecha crece con la intermitencia de la lluvia (una estacion con
# dias secos y chubascos la sufre mucho mas que un producto satelital ya
# suavizado). Comparar las dos fuentes en esa escala sobreestima el
# satelital por un factor cercano a 3, que es un artefacto de la
# representacion y no un sesgo fisico del sensor.
#
# Correccion: se reconstruyen los coeficientes de estacion con exactamente
# el mismo pre-procesamiento del satelital -> agregar a 72 pentadas, pasar
# a mm/dia, sacar raiz y ajustar la misma base de Fourier K=5 por minimos
# cuadrados con Cholesky propia (sin lm(), sin lstsq).
#
# Entrada : entrega/estaciones_depuradas.csv
# Salidas : entrega/coef_estaciones_pentadal.csv
#           entrega/tablas/fase3_reajuste_pentadal.csv
# =====================================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(BASE, "entrega")
TAB = os.path.join(ENT, "tablas")
K, PERIODO = 5, 365.0
MIN_DIAS_PENTADA = 3        # dias con dato para aceptar una pentada
MIN_PENTADAS = 60           # pentadas validas para aceptar un anio


def base_fourier(t, K, periodo=PERIODO):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


def cholesky_propia(A):
    """A = L L^T por el algoritmo de Cholesky, implementado a mano."""
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - float(L[i, :j] @ L[j, :j])
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("matriz no definida positiva")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def resuelve_cholesky(A, b):
    """Resuelve A c = b con A simetrica def. positiva, por sustitucion."""
    L = cholesky_propia(A)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - float(L[i, :i] @ y[:i])) / L[i, i]
    c = np.zeros(n)
    for i in range(n - 1, -1, -1):
        c[i] = (y[i] - float(L[i + 1:, i] @ c[i + 1:])) / L[i, i]
    return c


def tabla_pentadas(anio):
    """Para cada pentada 1..72: (doy_ini, doy_fin, dia_central_escalado, n_dias)."""
    bis = anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)
    dias_mes = [31, 29 if bis else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dias_tot = sum(dias_mes)
    out = []
    for p in range(1, 73):
        m = (p - 1) // 6 + 1
        j = (p - 1) % 6 + 1
        d_ini = 5 * (j - 1) + 1
        d_fin = dias_mes[m - 1] if j == 6 else 5 * j
        doy_ini = sum(dias_mes[:m - 1]) + d_ini
        doy_fin = sum(dias_mes[:m - 1]) + d_fin
        centro = (doy_ini + doy_fin) / 2.0
        out.append((doy_ini, doy_fin, (centro - 0.5) / dias_tot * 365.0,
                    d_fin - d_ini + 1))
    return out


print("=" * 72)
print("FASE 3.0 - reajuste pentadal de las curvas de estacion")
print("=" * 72)
d = pd.read_csv(os.path.join(ENT, "estaciones_depuradas.csv"),
                dtype={"STATION_ID": str},
                usecols=["STATION_ID", "YEAR", "DOY", "VALUE"])
print(f"    registros diarios: {len(d):,} | estaciones: {d.STATION_ID.nunique()} "
      f"| anios: {d.YEAR.min()}-{d.YEAR.max()}")

tablas = {a: tabla_pentadas(a) for a in sorted(d.YEAR.unique())}
# indice pentada por (anio, doy)
pent_de = {}
for a, tb in tablas.items():
    v = np.zeros(400, dtype=int)
    for p, (i0, i1, _, _) in enumerate(tb):
        v[i0:i1 + 1] = p + 1
    pent_de[a] = v

d["PENT"] = [pent_de[a][doy] for a, doy in zip(d.YEAR, d.DOY)]
d = d[d.PENT > 0]
g = (d.groupby(["STATION_ID", "YEAR", "PENT"])
     .agg(mm=("VALUE", "sum"), n=("VALUE", "size")).reset_index())
g = g[g.n >= MIN_DIAS_PENTADA]
g["mm_dia"] = g.mm / g.n
print(f"    pentadas validas (>= {MIN_DIAS_PENTADA} dias con dato): {len(g):,}")

filas, met = [], []
for (e, a), gg in g.groupby(["STATION_ID", "YEAR"]):
    if len(gg) < MIN_PENTADAS:
        continue
    tb = tablas[a]
    tp = np.array([tb[p - 1][2] for p in gg.PENT])
    y = np.sqrt(gg.mm_dia.to_numpy(float))
    w = gg.n.to_numpy(float)                  # pondera por dias observados
    Phi = base_fourier(tp, K)
    A = Phi.T @ (Phi * w[:, None])
    b = Phi.T @ (y * w)
    c = resuelve_cholesky(A, b)
    res = y - Phi @ c
    filas.append([e, a] + list(c))
    met.append(dict(STATION_ID=e, YEAR=a, n_pentadas=len(gg),
                    rmse=float(np.sqrt((res ** 2 * w).sum() / w.sum())),
                    r2=float(1 - (res ** 2).sum()
                             / max(((y - y.mean()) ** 2).sum(), 1e-12)),
                    acum_obs_mm=float((gg.mm_dia * [tb[p - 1][3] for p in gg.PENT]).sum())))

C = pd.DataFrame(filas, columns=["STATION_ID", "YEAR"] + [f"c{i+1}" for i in range(K)])
M = pd.DataFrame(met)
C.to_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), index=False)
M.to_csv(os.path.join(TAB, "fase3_reajuste_pentadal.csv"), index=False)
print(f"    curvas reajustadas: {len(C)} (estaciones {C.STATION_ID.nunique()}, "
      f"{C.YEAR.min()}-{C.YEAR.max()})")
print(f"    ajuste: R2 mediano {M.r2.median():.3f} | RMSE mediano {M.rmse.median():.4f} "
      f"sqrt(mm/dia)")

# comparacion con los coeficientes diarios de la fase 1
cd = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"), dtype={"STATION_ID": str})
t_dia = np.arange(0.5, 365.0, 1.0)
Ph = base_fourier(t_dia, K)
cc = C.merge(cd, on=["STATION_ID", "YEAR"], suffixes=("_p", "_d"))
Xp = cc[[f"c{i+1}_p" for i in range(K)]].to_numpy(float) @ Ph.T
Xd = cc[[f"c{i+1}_d" for i in range(K)]].to_numpy(float) @ Ph.T
int_p = (Xp ** 2).sum(axis=1)
int_d = (Xd ** 2).sum(axis=1)
obs = M.set_index(["STATION_ID", "YEAR"]).acum_obs_mm
obs = obs.loc[list(zip(cc.STATION_ID, cc.YEAR))].to_numpy()
print(f"\n    efecto Jensen medido en {len(cc)} curvas comunes:")
print(f"      integral con ajuste pentadal : {int_p.mean():7.0f} mm/anio")
print(f"      integral con ajuste diario   : {int_d.mean():7.0f} mm/anio")
print(f"      acumulado observado real     : {obs.mean():7.0f} mm/anio")
print(f"      razon pentadal/diario {np.median(int_p / int_d):.2f} | "
      f"sesgo pentadal vs observado {100 * (int_p.mean() / obs.mean() - 1):+.1f} % | "
      f"sesgo diario vs observado {100 * (int_d.mean() / obs.mean() - 1):+.1f} %")
print("    => el ajuste pentadal es el que hay que comparar contra el satelital,")
print("       porque comparte su pre-procesamiento exacto.")
