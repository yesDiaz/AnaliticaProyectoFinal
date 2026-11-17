# Coordenadas aproximadas por SUBZONA HIDROGRAFICA.
# Los 4 primeros digitos del STATION_ID son el codigo de subzona hidrografica
# del IDEAM. Del Catalogo Nacional de Estaciones (CNE, datos.gov.co hp9r-jxuu)
# se extrae el centroide y la altitud media de las estaciones de cada subzona,
# con su dispersion, para usarlos como coordenada aproximada y cuantificar la
# incertidumbre. No reemplaza el catalogo real: lo aproxima de forma auditable.
import numpy as np, pandas as pd, os, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cne = pd.read_csv(os.path.join(BASE, "aux", "cne.csv"), dtype=str, low_memory=False)
cne["Codigo"] = cne.Codigo.str.strip()
cne["SUB"] = cne.Codigo.str.lstrip("0").str.zfill(8).str[:4]


def parse_xy(s):
    m = re.findall(r"-?\d+\.?\d*", str(s))
    return (float(m[0]), float(m[1])) if len(m) >= 2 else (np.nan, np.nan)


xy = cne["Ubicación"].map(parse_xy)
cne["LON"] = [a for a, b in xy]
cne["LAT"] = [b for a, b in xy]
cne["ALT"] = pd.to_numeric(cne.Altitud, errors="coerce")

est = pd.read_csv(os.path.join(BASE, "entrega", "coef_estaciones.csv"),
                  dtype={"STATION_ID": str}).STATION_ID.unique()
sub_est = pd.Series(sorted({e[:4] for e in est}))
print("subzonas presentes en los datos:", len(sub_est), "->", list(sub_est))

g = (cne[cne.SUB.isin(sub_est)].groupby("SUB")
     .agg(n=("Codigo", "size"), LON=("LON", "median"), LAT=("LAT", "median"),
          ALT=("ALT", "median"), sd_lon=("LON", "std"), sd_lat=("LAT", "std"),
          sd_alt=("ALT", "std"), dep=("Departamento", lambda s: s.mode().iat[0])))
print("\ncentroides de subzona desde el CNE:")
print(g.to_string())

MALLA = dict(lon0=-77.525, lon1=-75.725, lat0=3.075, lat1=4.975)
g["en_malla"] = ((g.LON >= MALLA["lon0"] - 0.1) & (g.LON <= MALLA["lon1"] + 0.1)
                 & (g.LAT >= MALLA["lat0"] - 0.1) & (g.LAT <= MALLA["lat1"] + 0.1))
print("\nsubzonas dentro de la malla satelital:", int(g.en_malla.sum()), "de", len(g))
falta = [s for s in sub_est if s not in g.index]
print("subzonas sin registros en el CNE:", falta)

df = pd.DataFrame(dict(STATION_ID=est))
df["SUB"] = df.STATION_ID.str[:4]
df = df.merge(g.reset_index()[["SUB", "n", "LON", "LAT", "ALT", "sd_lon", "sd_lat",
                               "sd_alt", "dep", "en_malla"]], on="SUB", how="left")
print("\nestaciones con coordenada aproximada:", int(df.LON.notna().sum()), "de", len(df))
print("estaciones cuya subzona cae en la malla:", int((df.en_malla == True).sum()))
print("\npor departamento:", df.dep.value_counts().to_dict())
print("dispersion intra-subzona (km): lon %.0f lat %.0f"
      % (df.sd_lon.median() * 111.3 * np.cos(np.radians(4)), df.sd_lat.median() * 110.6))
df.to_csv(os.path.join(BASE, "aux", "coords_subzona.csv"), index=False)
