#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 3b: emparejamiento estacion-pixel con COORDENADAS REALES.

Sustituye la aproximacion por centroide de subzona hidrografica usada en la
fase 3 original. El catalogo oficial (publicado en la consigna del campus el
9-nov-2026) trae longitud, latitud y altitud de cada estacion, de modo que:

  - el emparejamiento se hace con el pixel satelital que CONTIENE la estacion
    (vecino mas cercano real), no con el de maxima correlacion dentro de un
    radio de 35 km alrededor de un centroide incierto;
  - la altitud es la de la estacion, no la mediana de su subzona;
  - el sesgo satelite-estacion se puede explicar con la altura real.

Salidas:
  entrega/pares_estacion_pixel.csv        (dataset para la fase 4, sobrescrito)
  entrega/tablas/fase3b_emparejamiento.csv
  entrega/tablas/fase3b_comparacion_metodos.csv
  entrega/figs/fase3b_mapa.png
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
NUE = os.path.join(RAIZ, "datos_nuevos")
K = 5
COLS = [f"c{j+1}" for j in range(K)]
rng = np.random.default_rng(2026)


def km(dlon, dlat, lat=4.0):
    """Distancia aproximada en km para desplazamientos pequenos."""
    return np.sqrt((dlon * 111.32 * np.cos(np.deg2rad(lat))) ** 2 + (dlat * 110.57) ** 2)


print("[1] catalogo oficial de estaciones")
ll = pd.read_csv(os.path.join(NUE, "CVC_Wheather_Stations_Long_Lat.csv"), encoding="latin-1")
cat = pd.read_excel(os.path.join(NUE, "CatalogoEstacionesElevacion.xlsx"))
ll["CODIGO"] = ll.CODIGO.astype(str).str.strip()
cat["codigo"] = cat.codigo.astype(str).str.strip()
coords = (ll[["CODIGO", "STATION_NAME", "CATEGORY", "Municipality", "BASIN", "Lon", "lat"]]
          .rename(columns={"CODIGO": "STATION_ID", "Lon": "LON_EST", "lat": "LAT_EST"})
          .merge(cat[["codigo", "alt"]].rename(columns={"codigo": "STATION_ID", "alt": "ALT_EST"}),
                 on="STATION_ID", how="left")
          .drop_duplicates("STATION_ID"))
print(f"    catalogo: {len(coords)} estaciones con coordenadas, "
      f"{coords.ALT_EST.notna().sum()} con altitud")

ce = pd.read_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), dtype={"STATION_ID": str}) \
    if os.path.exists(os.path.join(ENT, "coef_estaciones_pentadal.csv")) else None
ce_d = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"), dtype={"STATION_ID": str})
cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
cs["PIX"] = cs.FILA.astype(str) + "_" + cs.COL.astype(str)
ce_m = ce if ce is not None else ce_d
print(f"    curvas de estaciones: {len(ce_m)} | curvas satelitales: {len(cs)}")

pix = cs[["PIX", "LON", "LAT"]].drop_duplicates().reset_index(drop=True)
print(f"    malla satelital: {len(pix)} pixeles, "
      f"lon [{pix.LON.min():.3f}, {pix.LON.max():.3f}] "
      f"lat [{pix.LAT.min():.3f}, {pix.LAT.max():.3f}]")

# ---------------------------------------------------------------------
print("\n[2] emparejamiento por vecino mas cercano real")
est = sorted(ce_m.STATION_ID.unique())
C = coords.set_index("STATION_ID")
pl, pa = pix.LON.to_numpy(), pix.LAT.to_numpy()
filas = []
sin_coord = []
for e in est:
    if e not in C.index or not np.isfinite(C.LON_EST.get(e, np.nan)):
        sin_coord.append(e)
        continue
    lo, la = float(C.LON_EST[e]), float(C.LAT_EST[e])
    d = km(pl - lo, pa - la, la)
    j = int(np.argmin(d))
    filas.append(dict(STATION_ID=e, NOMBRE=C.STATION_NAME.get(e, ""),
                      MUNICIPIO=C.Municipality.get(e, ""), CUENCA=C.BASIN.get(e, ""),
                      LON_EST=lo, LAT_EST=la, ALT_EST=C.ALT_EST.get(e, np.nan),
                      PIX=str(pix.PIX[j]), LON_PIX=float(pl[j]), LAT_PIX=float(pa[j]),
                      dist_km=float(d[j])))
P = pd.DataFrame(filas)
dentro = P.dist_km <= 8.0        # la malla es de 0.05 deg ~ 5.5 km: media diagonal ~ 3.9 km
print(f"    estaciones con coordenada: {len(P)} de {len(est)} "
      f"(sin coordenada en el catalogo: {len(sin_coord)})")
print(f"    dentro de la malla (<= 8 km del centro del pixel): {int(dentro.sum())}")
print(f"    distancia estacion-pixel: mediana {P.dist_km.median():.2f} km | "
      f"p90 {P.dist_km.quantile(0.9):.1f} km | max {P.dist_km.max():.0f} km")
P = P[dentro].reset_index(drop=True)
print(f"    pixeles distintos: {P.PIX.nunique()} "
      f"({len(P) - P.PIX.nunique()} estaciones comparten pixel con otra)")
if P.ALT_EST.notna().any():
    print(f"    altitud real: {P.ALT_EST.min():.0f} - {P.ALT_EST.max():.0f} m "
          f"(mediana {P.ALT_EST.median():.0f})")

# control: el emparejamiento real debe correlacionar mejor que uno aleatorio
def mensual(df, clave, llaves):
    df = df.copy()
    A = np.full((len(llaves), (df.YEAR.max() - df.YEAR.min() + 1) * 12), np.nan)
    idx = {k: i for i, k in enumerate(llaves)}
    y0 = df.YEAR.min()
    t = np.arange(365) + 0.5
    arm = [np.ones_like(t)]
    for j in range(1, (K + 1) // 2 + 1):
        arm += [np.sin(2 * np.pi * j * t / 365), np.cos(2 * np.pi * j * t / 365)]
    Phi = np.column_stack(arm[:K]) * np.sqrt(2.0 / 365) ** 0  # misma base que fase 1
    lim = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    for _, r in df.iterrows():
        if r[clave] not in idx:
            continue
        cur = np.asarray(r[COLS], float) @ Phi.T
        for m in range(12):
            A[idx[r[clave]], (int(r.YEAR) - y0) * 12 + m] = cur[lim[m]:lim[m + 1]].mean()
    return A


print("\n[3] control del emparejamiento con anomalias mensuales")
ce_c = ce_m[ce_m.STATION_ID.isin(P.STATION_ID)]
cs_c = cs[cs.PIX.isin(P.PIX)]
anios = sorted(set(ce_c.YEAR) & set(cs_c.YEAR))
ce_c = ce_c[ce_c.YEAR.isin(anios)]
cs_c = cs_c[cs_c.YEAR.isin(anios)]


def anomalias(A):
    out = np.full_like(A, np.nan)
    for m in range(12):
        sl = slice(m, A.shape[1], 12)
        out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
    return out


lst_e = sorted(ce_c.STATION_ID.unique())
lst_p = sorted(cs_c.PIX.unique())
Ae = anomalias(mensual(ce_c, "STATION_ID", lst_e))
As = anomalias(mensual(cs_c, "PIX", lst_p))
ip = {p: i for i, p in enumerate(lst_p)}
mp = P.set_index("STATION_ID")
r_real, r_nulo = [], []
for i, e in enumerate(lst_e):
    ok = np.isfinite(Ae[i])
    if ok.sum() < 24:
        continue
    a = Ae[i, ok] - Ae[i, ok].mean()
    j = ip.get(str(mp.PIX[e]))
    if j is None:
        continue
    b = As[j, ok]
    m2 = np.isfinite(b)
    if m2.sum() < 24:
        continue
    aa, bb = a[m2] - a[m2].mean(), b[m2] - b[m2].mean()
    r_real.append(float((aa @ bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)))
    sel = rng.integers(0, len(lst_p), 30)
    B = As[np.ix_(sel, ok)][:, m2]
    B = B - np.nanmean(B, axis=1, keepdims=True)
    num = B @ aa
    den = np.sqrt(np.nansum(B ** 2, axis=1)) * np.linalg.norm(aa) + 1e-12
    r_nulo += list(num / den)
r_real = np.array(r_real)
r_nulo = np.array(r_nulo)
print(f"    r del pixel que contiene la estacion: media {np.nanmean(r_real):.3f} "
      f"(n = {len(r_real)})")
print(f"    r con pixeles aleatorios de la malla:  media {np.nanmean(r_nulo):.3f}")
print(f"    ganancia: {np.nanmean(r_real) - np.nanmean(r_nulo):+.3f}")
print("    [fase 3 con centroides: r 0.526 vs nulo 0.274]")
P["r_anom"] = [float(np.nan)] * len(P)
P.loc[P.STATION_ID.isin(lst_e), "r_anom"] = pd.Series(
    {e: r for e, r in zip([e for e in lst_e if e in set(P.STATION_ID)], r_real)}).reindex(
    P.loc[P.STATION_ID.isin(lst_e), "STATION_ID"]).to_numpy()
P.to_csv(os.path.join(TAB, "fase3b_emparejamiento.csv"), index=False)

# ---------------------------------------------------------------------
print("\n[4] dataset pareado para la fase 4")
ce_i = ce_m.set_index(["STATION_ID", "YEAR"])
cs_i = cs.set_index(["PIX", "YEAR"])
key = []
for (e, y) in ce_i.index:
    if e not in mp.index:
        continue
    p = str(mp.PIX[e])
    if (p, y) in cs_i.index:
        key.append((e, int(y), p))
key = pd.DataFrame(key, columns=["STATION_ID", "YEAR", "PIX"]).drop_duplicates()
sal = key.copy()
for j in range(K):
    sal[f"sat_c{j+1}"] = cs_i.loc[list(zip(key.PIX, key.YEAR)), COLS[j]].to_numpy()
    sal[f"est_c{j+1}"] = ce_i.loc[list(zip(key.STATION_ID, key.YEAR)), COLS[j]].to_numpy()
sal = sal.merge(P[["STATION_ID", "ALT_EST", "LON_EST", "LAT_EST", "LON_PIX", "LAT_PIX",
                   "dist_km", "r_anom", "NOMBRE", "MUNICIPIO"]], on="STATION_ID", how="left")
# compatibilidad con la fase 4: covariables con el nombre que ya usa el script
sal["ALT_SUB"] = sal.ALT_EST
med = sal.ALT_SUB.median()
sal["ALT_SUB"] = sal.ALT_SUB.fillna(med)
viejo = os.path.join(ENT, "pares_estacion_pixel.csv")
if os.path.exists(viejo) and not os.path.exists(viejo + ".centroides"):
    os.rename(viejo, viejo + ".centroides")
sal.to_csv(viejo, index=False)
print(f"    entrega/pares_estacion_pixel.csv: {len(sal)} filas | "
      f"{sal.STATION_ID.nunique()} estaciones | {sal.YEAR.min()}-{sal.YEAR.max()}")
print(f"    (version con centroides guardada como pares_estacion_pixel.csv.centroides)")
print(f"    altitud faltante imputada con la mediana ({med:.0f} m) en "
      f"{int(sal.ALT_EST.isna().sum())} filas")

cmp = pd.DataFrame([
    dict(metodo="centroide de subzona (fase 3)", estaciones=65, pixeles_distintos=np.nan,
         dist_mediana_km=np.nan, r_anom_media=0.526, r_nulo=0.274),
    dict(metodo="coordenada real (fase 3b)", estaciones=int(sal.STATION_ID.nunique()),
         pixeles_distintos=int(P.PIX.nunique()), dist_mediana_km=float(P.dist_km.median()),
         r_anom_media=float(np.nanmean(r_real)), r_nulo=float(np.nanmean(r_nulo)))])
cmp.to_csv(os.path.join(TAB, "fase3b_comparacion_metodos.csv"), index=False)
print("\n" + cmp.to_string(index=False))

# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 6.8))
ax.scatter(pix.LON, pix.LAT, s=1.2, c="0.82", label="malla satelital (0,05 deg)")
sc = ax.scatter(P.LON_EST, P.LAT_EST, c=P.ALT_EST, s=26, cmap="terrain",
                edgecolor="k", linewidth=0.3, label="estaciones (altitud real)")
plt.colorbar(sc, ax=ax, label="altitud (m)", shrink=0.8)
ax.set_xlabel("longitud")
ax.set_ylabel("latitud")
ax.set_title(f"Estaciones con coordenada oficial sobre la malla satelital\n"
             f"{len(P)} estaciones emparejadas por vecino mas cercano")
ax.legend(loc="lower left", fontsize=8)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase3b_mapa.png"), dpi=150)
print("\n[5] figura entrega/figs/fase3b_mapa.png")
print("\nFASE 3b OK")
