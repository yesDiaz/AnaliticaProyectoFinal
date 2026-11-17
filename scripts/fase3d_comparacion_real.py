#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 3d: comparacion funcional pareada con el emparejamiento georreferenciado.

Las cifras de la fase 3 original (sesgo funcional, banda bootstrap, sesgo
estructural frente a altitud/longitud/anio) se calcularon sobre pares
construidos con centroides de subzona, que la fase 3c demostro erroneos. Aqui
se recalculan sobre los 2.324 pares con coordenada oficial, usando la altitud
real de la estacion en lugar de la mediana de la subzona.

Salidas:
  entrega/tablas/fase3d_comparacion.csv
  entrega/tablas/fase3d_sesgo_estructural.csv
  entrega/figs/fase3d_sesgo_funcional.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(RAIZ, "entrega")
TAB, FIG = os.path.join(ENT, "tablas"), os.path.join(ENT, "figs")
K, PERIODO = 5, 365.0
rng = np.random.default_rng(2026)


def base_fourier(t, K, periodo=PERIODO):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


t = np.arange(365) + 0.5
Phi = base_fourier(t, K)
D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
SAT = [f"sat_c{i+1}" for i in range(K)]
EST = [f"est_c{i+1}" for i in range(K)]
Ce, Cs = D[EST].to_numpy(float), D[SAT].to_numpy(float)
Xe, Xs = Ce @ Phi.T, Cs @ Phi.T
print(f"[1] pares georreferenciados: {len(D)} | {D.STATION_ID.nunique()} estaciones | "
      f"{D.YEAR.min()}-{D.YEAR.max()}")

dif = Xe - Xs
sesgo = dif.mean(axis=0)
B = 1000
boot = np.empty((B, 365))
for b in range(B):
    idx = rng.integers(0, len(dif), len(dif))
    boot[b] = dif[idx].mean(axis=0)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
sig = (lo > 0) | (hi < 0)
print(f"[2] sesgo funcional medio (estacion - satelite): {sesgo.mean():+.4f} raiz(mm)/dia")
print(f"    maximo {sesgo.max():+.4f} (dia {int(sesgo.argmax())+1}) | "
      f"minimo {sesgo.min():+.4f} (dia {int(sesgo.argmin())+1})")
print(f"    dias con banda bootstrap al 95 % que excluye el cero: "
      f"{int(sig.sum())} de 365 ({100*sig.mean():.1f} %)")

acu_e, acu_s = Xe.sum(axis=1), Xs.sum(axis=1)
razon = acu_s.mean() / acu_e.mean()
print(f"[3] integral anual media: estacion {acu_e.mean():.1f} | satelite {acu_s.mean():.1f} "
      f"| razon satelite/estacion {razon:.3f}")

D["sesgo_int"] = dif.sum(axis=1) / 365
filas = []
for col, nom in [("ALT_EST", "altitud real"), ("LON_EST", "longitud"),
                 ("LAT_EST", "latitud"), ("YEAR", "anio"), ("dist_km", "distancia al pixel")]:
    if col not in D.columns:
        continue
    s = D[["sesgo_int", col]].dropna()
    if len(s) < 30:
        continue
    rho = s.corr(method="spearman").iloc[0, 1]
    filas.append(dict(variable=nom, n=len(s), spearman=float(rho)))
    print(f"    sesgo integrado vs {nom:20s}: rho = {rho:+.3f}  (n = {len(s)})")
T = pd.DataFrame(filas)
T.to_csv(os.path.join(TAB, "fase3d_sesgo_estructural.csv"), index=False)
pd.DataFrame(dict(dia=np.arange(1, 366), sesgo=sesgo, ic_inf=lo, ic_sup=hi,
                  significativo=sig)).to_csv(
    os.path.join(TAB, "fase3d_comparacion.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].plot(t, Xe.mean(axis=0), lw=1.8, color="k", label="estaciones")
axs[0].plot(t, Xs.mean(axis=0), lw=1.8, color="tab:orange", label="satelite")
axs[0].set_title("Medias funcionales pareadas (coordenadas reales)")
axs[0].set_xlabel("dia del anio")
axs[0].set_ylabel("$\\sqrt{mm/dia}$")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)
axs[1].plot(t, sesgo, color="tab:blue", lw=1.6)
axs[1].fill_between(t, lo, hi, alpha=0.25, color="tab:blue", label="IC 95 % (bootstrap)")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_title(f"Sesgo funcional estacion - satelite\n"
                 f"({100*sig.mean():.0f} % de los dias con diferencia significativa)")
axs[1].set_xlabel("dia del anio")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase3d_sesgo_funcional.png"), dpi=150)
print("\nFASE 3d OK")
