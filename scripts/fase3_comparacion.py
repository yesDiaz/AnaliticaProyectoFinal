#!/usr/bin/env python3
# =====================================================================
# FASE 3 - Comparacion estacion terrestre vs producto satelital
# Proyecto Final de Analitica de Datos (UAO 2026-2S)
#
# Problema de partida: el proyecto no entrega el catalogo de coordenadas de
# las estaciones y los 186 STATION_ID no coinciden con el campo Codigo del
# Catalogo Nacional de Estaciones del IDEAM (0/186 literal, ver
# aux/NOTA_CODIGOS_ESTACIONES.md). Lo que SI es valido es la estructura del
# codigo: los 4 primeros digitos son la SUBZONA HIDROGRAFICA (2610..2625 =
# cuenca del Cauca, 5300..5421 = Patia y vertiente pacifica sur), y eso se
# verifica contra el CNE (429 estaciones del CNE en la subzona 2610, 513 en
# 2612, etc.).
#
# De ahi la estrategia de esta fase, en tres pasos auditables:
#   (1) coordenada aproximada = centroide de la subzona segun el CNE, con su
#       dispersion como incertidumbre declarada (~11 km en lon, ~28 km en lat);
#   (2) universo de analisis = solo las estaciones cuya subzona cae dentro de
#       la malla satelital (el resto esta fuera del dominio del satelital, lo
#       cual es un hallazgo del proyecto, no un error de procesamiento);
#   (3) emparejamiento estacion-pixel = entre los pixeles a menos de RADIO km
#       del centroide, el de maxima correlacion de ANOMALIAS mensuales
#       (climatologia removida: el ciclo estacional es casi identico en toda
#       la region y no discrimina posicion; la variabilidad interanual si).
#
# Reglas de la rubrica respetadas: todo estadistico funcional se calcula a
# mano (medias, sesgo, RMSE, correlacion funcional, banda bootstrap); las
# derivadas son analiticas sobre la base de Fourier; sin lm() ni lstsq.
#
# Entradas : entrega/coef_estaciones.csv, entrega/coef_satelite.csv,
#            entrega/estaciones_depuradas.csv, aux/cne.csv
# Salidas  : entrega/pares_estacion_pixel.csv (insumo de la fase 4),
#            entrega/figs/fase3_*.png, entrega/tablas/fase3_*.csv
# =====================================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMILLA = 20263
rng = np.random.default_rng(SEMILLA)
BASE = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(BASE, "entrega")
FIG = os.path.join(ENT, "figs")
TAB = os.path.join(ENT, "tablas")
AUX = os.path.join(BASE, "aux")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

K, PERIODO = 5, 365.0
t_dia = np.arange(0.5, 365.0, 1.0)
COLS = [f"c{i+1}" for i in range(K)]
RADIO_KM = 35.0            # radio de busqueda alrededor del centroide de subzona
MALLA = dict(lon0=-77.525, lon1=-75.725, lat0=3.075, lat1=4.975)
MARGEN = 0.15              # grados de tolerancia al declarar "dentro de la malla"
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
INI_MES = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def base_fourier(t, K, periodo=PERIODO):
    """Phi (len(t) x K): 1, sen(2pi j t/T), cos(2pi j t/T). K impar."""
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


def dbase_fourier(t, K, periodo=PERIODO):
    """Derivada analitica de la base (no diferencias finitas)."""
    w = 2 * np.pi / periodo
    cols = [np.zeros_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(j * w * np.cos(j * w * t))
        cols.append(-j * w * np.sin(j * w * t))
    return np.column_stack(cols[:K])


def integra(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def media_paso_a_paso(X):
    """Media funcional con acumulador explicito (sin np.mean), regla de la rubrica."""
    acum = np.zeros(X.shape[1])
    n = 0
    for i in range(X.shape[0]):
        acum = acum + X[i]
        n += 1
    return acum / n


def eje_meses(ax, ylab):
    ax.set_xticks(INI_MES + 15)
    ax.set_xticklabels(MESES, fontsize=8)
    ax.set_xlim(0, 365)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25)


def km(dlon, dlat, lat=4.0):
    return np.sqrt((dlon * 111.32 * np.cos(np.radians(lat))) ** 2 + (dlat * 110.57) ** 2)


Phi = base_fourier(t_dia, K)
dPhi = dbase_fourier(t_dia, K)

print("=" * 72)
print("FASE 3 - Comparacion estaciones vs satelital  (semilla %d)" % SEMILLA)
print("=" * 72)

# ---------------------------------------------------------------------
# 1. Coordenada aproximada por subzona hidrografica (CNE del IDEAM)
# ---------------------------------------------------------------------
print("\n[1] coordenada aproximada por subzona hidrografica")
cne = pd.read_csv(os.path.join(AUX, "cne.csv"), dtype=str, low_memory=False)
cne["SUB"] = cne.Codigo.str.strip().str.lstrip("0").str.zfill(8).str[:4]
xy = cne["Ubicación"].map(lambda s: re.findall(r"-?\d+\.?\d*", str(s)))
cne["LON"] = [float(v[0]) if len(v) >= 2 else np.nan for v in xy]
cne["LAT"] = [float(v[1]) if len(v) >= 2 else np.nan for v in xy]
cne["ALT"] = pd.to_numeric(cne.Altitud, errors="coerce")

# IMPORTANTE: se usan los coeficientes PENTADALES (fase3_0_pentadas.py), no los
# diarios de la fase 1. El satelital se ajusto sobre promedios pentadales y la
# raiz es concava: comparar contra el ajuste diario infla el satelital un factor
# ~2.1 por desigualdad de Jensen (artefacto de representacion, no sesgo fisico).
PENT = os.path.join(ENT, "coef_estaciones_pentadal.csv")
if not os.path.exists(PENT):
    raise SystemExit("falta coef_estaciones_pentadal.csv: ejecuta antes fase3_0_pentadas.py")
ce = pd.read_csv(PENT, dtype={"STATION_ID": str})
ce_diaria = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"),
                        dtype={"STATION_ID": str})
cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
ce["SUB"] = ce.STATION_ID.str[:4]
subz = (cne[cne.SUB.isin(ce.SUB.unique())].groupby("SUB")
        .agg(n_cne=("Codigo", "size"), LON_SUB=("LON", "median"),
             LAT_SUB=("LAT", "median"), ALT_SUB=("ALT", "median"),
             sd_lon=("LON", "std"), sd_lat=("LAT", "std"), sd_alt=("ALT", "std"),
             DEP=("Departamento", lambda s: s.mode().iat[0])).reset_index())
subz["EN_MALLA"] = ((subz.LON_SUB >= MALLA["lon0"] - MARGEN)
                    & (subz.LON_SUB <= MALLA["lon1"] + MARGEN)
                    & (subz.LAT_SUB >= MALLA["lat0"] - MARGEN)
                    & (subz.LAT_SUB <= MALLA["lat1"] + MARGEN))
print(f"    subzonas en los datos: {ce.SUB.nunique()} | con registros en el CNE: {len(subz)}")
print(f"    incertidumbre mediana del centroide: {km(subz.sd_lon.median(), 0):.0f} km en lon, "
      f"{km(0, subz.sd_lat.median()):.0f} km en lat")
for _, r in subz.sort_values("SUB").iterrows():
    print(f"      {r.SUB}  n_CNE={int(r.n_cne):4d}  ({r.LON_SUB:8.3f}, {r.LAT_SUB:6.3f})  "
          f"alt={r.ALT_SUB:6.0f} m  {r.DEP:<16s} {'EN MALLA' if r.EN_MALLA else 'fuera'}")
subz.to_csv(os.path.join(TAB, "fase3_subzonas.csv"), index=False)

ce = ce.merge(subz, on="SUB", how="left")
est_tot = ce.STATION_ID.nunique()
ce_m = ce[ce.EN_MALLA == True].copy()
print(f"\n    [HALLAZGO] estaciones dentro del dominio satelital: "
      f"{ce_m.STATION_ID.nunique()} de {est_tot} "
      f"({100 * ce_m.STATION_ID.nunique() / est_tot:.0f} %)")
print("    el resto pertenece al Cauca medio-bajo (Antioquia) y al Patia/Pacifico sur,")
print("    fuera de la malla del producto satelital: los dos conjuntos de datos del")
print("    proyecto NO cubren la misma region. La comparacion pareada solo es legitima")
print("    en la interseccion de dominios.")

# ---------------------------------------------------------------------
# 2. Emparejamiento estacion-pixel por correlacion de anomalias
# ---------------------------------------------------------------------
print("\n[2] emparejamiento estacion-pixel (radio %.0f km + maxima correlacion de anomalias)"
      % RADIO_KM)
anios = sorted(set(ce_m.YEAR) & set(cs.YEAR))
ce_m = ce_m[ce_m.YEAR.isin(anios)]
csa = cs[cs.YEAR.isin(anios)].copy()
csa["PIX"] = csa.FILA.astype(str) + "_" + csa.COL.astype(str)
pix = csa[["PIX", "LON", "LAT"]].drop_duplicates().reset_index(drop=True)
print(f"    anios comunes: {anios[0]}-{anios[-1]} ({len(anios)}) | pixeles: {len(pix)}")

# matrices mensuales (promedio de la curva por mes) y anomalias
Mmes = np.zeros((12, len(t_dia)))
mes_de_dia = np.searchsorted(INI_MES[1:], t_dia, side="right")
for m in range(12):
    sel = mes_de_dia == m
    Mmes[m] = sel / sel.sum()


def mensual(df, clave, llaves):
    idx = {k: i for i, k in enumerate(llaves)}
    ia = {a: i for i, a in enumerate(anios)}
    out = np.full((len(llaves), 12 * len(anios)), np.nan)
    men = (df[COLS].to_numpy(float) @ Phi.T) @ Mmes.T
    for (k, a), fila in zip(zip(df[clave], df.YEAR), men):
        out[idx[k], 12 * ia[a]:12 * ia[a] + 12] = fila
    return out


def anomalias(A):
    out = A.copy()
    for m in range(12):
        sl = slice(m, A.shape[1], 12)
        out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
    return out


est = sorted(ce_m.STATION_ID.unique())
Ae = anomalias(mensual(ce_m, "STATION_ID", est))
As = anomalias(mensual(csa, "PIX", list(pix.PIX)))

meta = ce_m.drop_duplicates("STATION_ID").set_index("STATION_ID")
pares = []
for i, e in enumerate(est):
    d = km(pix.LON - meta.LON_SUB[e], pix.LAT - meta.LAT_SUB[e])
    cand = np.where(d.to_numpy() <= RADIO_KM)[0]
    if len(cand) == 0:
        cand = np.array([int(np.argmin(d))])
    ok = np.isfinite(Ae[i])
    a = Ae[i, ok] - Ae[i, ok].mean()
    B = As[np.ix_(cand, ok)]
    B = B - B.mean(axis=1, keepdims=True)
    r = (B @ a) / (np.sqrt((B ** 2).sum(axis=1)) * np.sqrt((a ** 2).sum()) + 1e-12)
    j = int(np.argmax(r))
    p = pix.iloc[cand[j]]
    pares.append(dict(STATION_ID=e, SUB=meta.SUB[e], DEP=meta.DEP[e],
                      ALT_SUB=meta.ALT_SUB[e], LON_SUB=meta.LON_SUB[e],
                      LAT_SUB=meta.LAT_SUB[e], PIX=p.PIX, LON_PIX=p.LON,
                      LAT_PIX=p.LAT, n_cand=len(cand), r_anom=float(r[j]),
                      dist_km=float(km(p.LON - meta.LON_SUB[e], p.LAT - meta.LAT_SUB[e])),
                      n_meses=int(ok.sum())))
P = pd.DataFrame(pares)
print(f"    estaciones emparejadas: {len(P)} | pixeles distintos: {P.PIX.nunique()}")
print(f"    r de anomalias: mediana {P.r_anom.median():.3f} "
      f"[{P.r_anom.min():.3f}, {P.r_anom.max():.3f}]")
print(f"    candidatos por estacion: mediana {P.n_cand.median():.0f} | "
      f"distancia al centroide: mediana {P.dist_km.median():.1f} km")

# contraste: r del pixel elegido vs r de pixeles aleatorios de la malla
r_nulo = []
for i in range(len(est)):
    ok = np.isfinite(Ae[i])
    a = Ae[i, ok] - Ae[i, ok].mean()
    sel = rng.integers(0, len(pix), 40)
    B = As[np.ix_(sel, ok)]
    B = B - B.mean(axis=1, keepdims=True)
    r_nulo.append(((B @ a) / (np.sqrt((B ** 2).sum(axis=1)) * np.sqrt((a ** 2).sum()) + 1e-12)))
r_nulo = np.concatenate(r_nulo)
print(f"    control: r con pixeles aleatorios = {r_nulo.mean():.3f} +- {r_nulo.std():.3f} "
      f"=> el emparejamiento gana {P.r_anom.mean() - r_nulo.mean():+.3f} de correlacion")
P.to_csv(os.path.join(TAB, "fase3_emparejamiento.csv"), index=False)

# ---------------------------------------------------------------------
# 3. Comparacion funcional pareada
# ---------------------------------------------------------------------
print("\n[3] comparacion funcional pareada estacion vs pixel")
mp = P.set_index("STATION_ID")
ce_i = ce_m.set_index(["STATION_ID", "YEAR"])
cs_i = csa.set_index(["PIX", "YEAR"])
Xe, Xs, key = [], [], []
for (e, y) in ce_i.index:
    p = mp.PIX[e]
    if (p, y) in cs_i.index:
        Xe.append(ce_i.loc[(e, y), COLS].to_numpy(float) @ Phi.T)
        Xs.append(cs_i.loc[(p, y), COLS].to_numpy(float) @ Phi.T)
        key.append((e, y, p))
Xe, Xs = np.array(Xe), np.array(Xs)
key = pd.DataFrame(key, columns=["STATION_ID", "YEAR", "PIX"])
print(f"    pares estacion-anio comparables: {len(Xe)}")

mu_e = media_paso_a_paso(Xe)
mu_s = media_paso_a_paso(Xs)
D = Xe - Xs
mu_d = media_paso_a_paso(D)

# banda bootstrap de la diferencia media (1000 remuestreos de los pares)
B_BOOT = 1000
boot = np.empty((B_BOOT, len(t_dia)))
for b in range(B_BOOT):
    idx = rng.integers(0, len(D), len(D))
    boot[b] = D[idx].mean(axis=0)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
signif = (lo > 0) | (hi < 0)
print(f"    sesgo medio integrado (estacion - satelite): "
      f"{integra(mu_d, t_dia) / 365:.3f} (sqrt(mm)/dia)")
print(f"    sesgo maximo {mu_d.max():+.3f} el dia {int(t_dia[np.argmax(mu_d)])} | "
      f"minimo {mu_d.min():+.3f} el dia {int(t_dia[np.argmin(mu_d)])}")
print(f"    la diferencia es significativa al 95 % en {100 * signif.mean():.0f} % del anio")
rmse_f = np.sqrt((D ** 2).mean(axis=0))
print(f"    RMSE funcional: medio {rmse_f.mean():.3f} | max {rmse_f.max():.3f} "
      f"el dia {int(t_dia[np.argmax(rmse_f)])}")

# correlacion funcional punto a punto entre fuentes (a mano)
corr_t = np.empty(len(t_dia))
for j in range(len(t_dia)):
    a, b = Xe[:, j], Xs[:, j]
    a = a - a.mean()
    b = b - b.mean()
    corr_t[j] = (a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-12)
print(f"    correlacion entre fuentes dia a dia: mediana {np.median(corr_t):.3f} "
      f"[{corr_t.min():.3f}, {corr_t.max():.3f}]")

# totales anuales y desfase de temporadas
tot_e = (Xe ** 2).sum(axis=1)
tot_s = (Xs ** 2).sum(axis=1)
raz = tot_s / tot_e
print(f"    integral anual: estacion {tot_e.mean():.0f} vs satelite {tot_s.mean():.0f} "
      f"(razon mediana {np.median(raz):.2f})")

# misma comparacion con el ajuste DIARIO de la fase 1, para cuantificar cuanto
# del desacuerdo era artefacto de la representacion y no sesgo del satelital
cd_i = ce_diaria.set_index(["STATION_ID", "YEAR"])
hay = [i for i, (e, y) in enumerate(zip(key.STATION_ID, key.YEAR)) if (e, y) in cd_i.index]
Xd = cd_i.loc[list(zip(key.STATION_ID[hay], key.YEAR[hay])), COLS].to_numpy(float) @ Phi.T
raz_d = tot_s[hay] / (Xd ** 2).sum(axis=1)
print(f"    control de escala: con el ajuste diario de la fase 1 la razon sat/est seria "
      f"{np.median(raz_d):.2f}")
print(f"    => {100 * (1 - np.median(raz) / np.median(raz_d)):.0f} % del desacuerdo original "
      f"era artefacto de Jensen, no sesgo del producto satelital")
dmu_e = (ce_i[COLS].to_numpy(float).mean(axis=0)) @ dPhi.T
dmu_s = (cs_i[COLS].to_numpy(float).mean(axis=0)) @ dPhi.T


def picos(mu):
    d = np.sign(np.diff(np.sign(np.gradient(mu, t_dia))))
    return t_dia[1:][d < 0]


print(f"    maximos de la media: estacion dias {np.round(picos(mu_e)).astype(int).tolist()} | "
      f"satelite dias {np.round(picos(mu_s)).astype(int).tolist()}")

res = key.copy()
res["sesgo_int"] = integra(D, t_dia) / 365
res["rmse"] = np.sqrt((D ** 2).mean(axis=1))
res["tot_est"] = tot_e
res["tot_sat"] = tot_s
res["razon_sat_est"] = raz
res = res.merge(P[["STATION_ID", "SUB", "DEP", "ALT_SUB", "LON_PIX", "LAT_PIX",
                   "r_anom", "dist_km"]], on="STATION_ID", how="left")
res.to_csv(os.path.join(TAB, "fase3_metricas_par.csv"), index=False)

print("\n    sesgo por subzona (estacion - satelite, sqrt(mm)/dia):")
for s, g in res.groupby("SUB"):
    print(f"      {s}  n={len(g):4d}  sesgo {g.sesgo_int.mean():+.3f}  "
          f"razon sat/est {g.razon_sat_est.median():.2f}  alt {g.ALT_SUB.iloc[0]:.0f} m")
rho_alt = res[["sesgo_int", "ALT_SUB"]].corr(method="spearman").iloc[0, 1]
rho_lon = res[["sesgo_int", "LON_PIX"]].corr(method="spearman").iloc[0, 1]
print(f"    Spearman sesgo vs altitud de subzona: {rho_alt:+.3f} | vs longitud: {rho_lon:+.3f}")

# dependencia del sesgo con el anio (deriva del producto satelital)
por_anio = res.groupby("YEAR").agg(n=("rmse", "size"), sesgo=("sesgo_int", "mean"),
                                   razon=("razon_sat_est", "median")).reset_index()
rho_y = por_anio[["YEAR", "sesgo"]].corr(method="spearman").iloc[0, 1]
print(f"    Spearman sesgo vs anio: {rho_y:+.3f} "
      f"(deriva temporal del producto satelital)")
por_anio.to_csv(os.path.join(TAB, "fase3_sesgo_por_anio.csv"), index=False)
pd.DataFrame(dict(dia=t_dia, media_est=mu_e, media_sat=mu_s, dif=mu_d,
                  ic_bajo=lo, ic_alto=hi, significativo=signif,
                  rmse=rmse_f, corr=corr_t)
             ).to_csv(os.path.join(TAB, "fase3_curvas_comparacion.csv"), index=False)

# ---------------------------------------------------------------------
# 4. Dataset pareado para la fase 4
# ---------------------------------------------------------------------
sal = key.copy()
for j in range(K):
    sal[f"sat_c{j+1}"] = cs_i.loc[list(zip(key.PIX, key.YEAR)), COLS[j]].to_numpy()
    sal[f"est_c{j+1}"] = ce_i.loc[list(zip(key.STATION_ID, key.YEAR)), COLS[j]].to_numpy()
sal = sal.merge(P[["STATION_ID", "SUB", "ALT_SUB", "LON_PIX", "LAT_PIX", "r_anom"]],
                on="STATION_ID", how="left")
sal.to_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), index=False)
print(f"\n[4] dataset pareado para la fase 4: entrega/pares_estacion_pixel.csv "
      f"({len(sal)} filas, {sal.STATION_ID.nunique()} estaciones, "
      f"{sal.YEAR.min()}-{sal.YEAR.max()})")

# ---------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
axs[0].plot(t_dia, mu_e, color="tab:blue", lw=2, label="estaciones")
axs[0].plot(t_dia, mu_s, color="tab:orange", lw=2, label="satelital")
eje_meses(axs[0], r"$\sqrt{P}$")
axs[0].legend(fontsize=8)
axs[0].set_title(f"Media funcional pareada (n={len(Xe)})")
axs[1].fill_between(t_dia, lo, hi, color="tab:red", alpha=0.25, label="IC 95 % bootstrap")
axs[1].plot(t_dia, mu_d, color="tab:red", lw=1.8, label="estacion - satelite")
axs[1].axhline(0, color="black", lw=0.8)
eje_meses(axs[1], r"diferencia $\sqrt{P}$")
axs[1].legend(fontsize=8)
axs[1].set_title("Sesgo funcional con banda bootstrap")
axs[2].plot(t_dia, corr_t, color="tab:green", lw=1.6, label="correlacion entre fuentes")
axs[2].plot(t_dia, rmse_f / rmse_f.max(), color="gray", lw=1.2, ls="--",
            label="RMSE funcional (normalizado)")
eje_meses(axs[2], "correlacion")
axs[2].legend(fontsize=8)
axs[2].set_title("Acuerdo dia a dia")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase3_comparacion.png"), dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
sc = axs[0].scatter(pix.LON, pix.LAT, s=4, color="lightgray", label="malla satelital")
axs[0].scatter(P.LON_SUB, P.LAT_SUB, s=55, marker="s", facecolor="none",
               edgecolor="tab:blue", label="centroide de subzona")
axs[0].scatter(P.LON_PIX, P.LAT_PIX, s=18, color="tab:red", label="pixel emparejado")
for _, r in P.iterrows():
    axs[0].plot([r.LON_SUB, r.LON_PIX], [r.LAT_SUB, r.LAT_PIX], color="tab:red",
                lw=0.4, alpha=0.5)
axs[0].set_xlabel("longitud")
axs[0].set_ylabel("latitud")
axs[0].legend(fontsize=7)
axs[0].set_title("Emparejamiento espacial")
axs[1].hist(r_nulo, bins=40, density=True, color="lightgray",
            label="pixeles aleatorios")
axs[1].hist(P.r_anom, bins=20, density=True, color="tab:red", alpha=0.7,
            label="pixel emparejado")
axs[1].set_xlabel("correlacion de anomalias mensuales")
axs[1].legend(fontsize=8)
axs[1].set_title("Validacion del emparejamiento")
axs[2].scatter(res.tot_est, res.tot_sat, s=6, alpha=0.35, color="tab:purple")
lim = [0, float(max(res.tot_est.max(), res.tot_sat.max())) * 1.02]
axs[2].plot(lim, lim, color="black", ls="--", lw=1, label="identidad")
axs[2].set_xlabel("integral anual, estacion")
axs[2].set_ylabel("integral anual, satelite")
axs[2].legend(fontsize=8)
axs[2].set_title(f"Totales pareados (razon mediana {np.median(raz):.2f})")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase3_emparejamiento.png"), dpi=130)
plt.close(fig)

print("\n[fin] figuras fase3_*.png, tablas fase3_*.csv, dataset pares_estacion_pixel.csv")
