# Emparejamiento estacion-pixel SIN catalogo de coordenadas:
# se asigna a cada estacion el pixel cuya serie mensual (23 anios x 12 meses,
# escala sqrt(P)) maximiza la correlacion de Pearson. Luego se valida que el
# resultado tenga coherencia espacial: estaciones de la misma subzona
# hidrografica (4 primeros digitos del codigo) deben caer en pixeles cercanos.
import numpy as np, pandas as pd, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT = os.path.join(BASE, "entrega")
K, PER = 5, 365.0
t = np.arange(0.5, 365.0, 1.0)


def phi(t, K, per=PER):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols += [np.sin(2 * np.pi * j * t / per), np.cos(2 * np.pi * j * t / per)]
    return np.column_stack(cols[:K])


Phi = phi(t, K)
mes = np.array([pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(d)) for d in t])
mes = np.array([m.month for m in mes])
M = np.zeros((12, len(t)))
for m in range(1, 13):
    M[m - 1] = (mes == m) / (mes == m).sum()        # promedio por mes

COLS = [f"c{i+1}" for i in range(K)]
ce = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"), dtype={"STATION_ID": str})
cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
anios = sorted(set(ce.YEAR) & set(cs.YEAR))
print("anios comunes:", anios[0], "-", anios[-1], f"({len(anios)})")

ce = ce[ce.YEAR.isin(anios)]
cs = cs[cs.YEAR.isin(anios)]
est = sorted(ce.STATION_ID.unique())
pix = cs[["FILA", "COL", "LON", "LAT"]].drop_duplicates().reset_index(drop=True)
pix["PIX"] = pix.FILA.astype(str) + "_" + pix.COL.astype(str)
cs["PIX"] = cs.FILA.astype(str) + "_" + cs.COL.astype(str)
print("estaciones:", len(est), "| pixeles:", len(pix))


def matriz_mensual(df, clave, llaves):
    """(n_unidades x 12*n_anios) con el promedio mensual de la curva, NaN si falta."""
    idx = {k: i for i, k in enumerate(llaves)}
    ia = {a: i for i, a in enumerate(anios)}
    out = np.full((len(llaves), 12 * len(anios)), np.nan)
    X = df[COLS].to_numpy(float) @ Phi.T          # curvas diarias
    men = X @ M.T                                 # (n x 12)
    for (k, a), fila in zip(zip(df[clave], df.YEAR), men):
        out[idx[k], 12 * ia[a]:12 * ia[a] + 12] = fila
    return out


Ae = matriz_mensual(ce, "STATION_ID", est)
As = matriz_mensual(cs, "PIX", list(pix.PIX))
print("cobertura media estaciones: %.2f | satelite: %.2f"
      % (np.isfinite(Ae).mean(), np.isfinite(As).mean()))

# ANOMALIAS: se resta a cada unidad su climatologia mensual, de modo que la
# correlacion mida variabilidad interanual local y no el ciclo estacional comun
# (que es casi identico en toda la region y no discrimina posicion).
def anomalias(A):
    out = A.copy()
    for m in range(12):
        col = slice(m, A.shape[1], 12)
        bloque = A[:, col]
        out[:, col] = bloque - np.nanmean(bloque, axis=1, keepdims=True)
    return out


Ae = anomalias(Ae)
As = anomalias(As)
print("modo: correlacion de ANOMALIAS mensuales (climatologia removida)")

# correlacion sobre las columnas comunes sin NaN, por estacion
res = []
for i, e in enumerate(est):
    ok = np.isfinite(Ae[i])
    a = Ae[i, ok]
    B = As[:, ok]
    bueno = np.isfinite(B).all(axis=1)
    a = a - a.mean()
    Bc = B[bueno] - B[bueno].mean(axis=1, keepdims=True)
    r = (Bc @ a) / (np.sqrt((Bc ** 2).sum(axis=1)) * np.sqrt((a ** 2).sum()) + 1e-12)
    j = int(np.argmax(r))
    fila = pix[bueno].iloc[j]
    res.append(dict(STATION_ID=e, n_meses=int(ok.sum()), PIX=fila.PIX,
                    LON=fila.LON, LAT=fila.LAT, r_max=float(r[j]),
                    r_p95=float(np.percentile(r, 95)), r_mediana=float(np.median(r))))
R = pd.DataFrame(res)
R["SUBZONA"] = R.STATION_ID.str[:4]
print("\nr_max: mediana %.3f | min %.3f | max %.3f" % (R.r_max.median(), R.r_max.min(), R.r_max.max()))
print("margen r_max - p95: mediana %.3f" % (R.r_max - R.r_p95).median())
print("pixeles distintos asignados:", R.PIX.nunique(), "de", len(R), "estaciones")

# validacion espacial: distancia media intra-subzona vs aleatoria
def dist(a, b):
    return np.sqrt(((a.LON - b.LON) * 111.3 * np.cos(np.radians(4))) ** 2
                   + ((a.LAT - b.LAT) * 110.6) ** 2)


intra, n = [], 0
for z, g in R.groupby("SUBZONA"):
    if len(g) < 2:
        continue
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            intra.append(dist(g.iloc[i], g.iloc[j]))
    n += 1
rng = np.random.default_rng(7)
alea = [dist(R.iloc[i], R.iloc[j]) for i, j in rng.integers(0, len(R), (3000, 2))]
print("\nsubzonas con >=2 estaciones:", n)
print("distancia media intra-subzona: %.1f km (n=%d pares)" % (np.mean(intra), len(intra)))
print("distancia media entre pares aleatorios: %.1f km" % np.mean(alea))
print("ratio intra/aleatorio: %.3f  (<1 = coherencia espacial)" % (np.mean(intra) / np.mean(alea)))
print("\nconteo por subzona (top):", R.SUBZONA.value_counts().head(8).to_dict())
R.to_csv(os.path.join(BASE, "aux", "emparejamiento_corr.csv"), index=False)
print("\n", R.head(10).to_string(index=False))
