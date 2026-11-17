"""Empareja STATION_ID (10 digitos) del dataset del proyecto con el CNE del IDEAM.

Hipotesis verificada empiricamente:
  id10 = subzona(4) + "00" + consecutivo(3) + categoria(1)
  CNE8 = subzona(4) + consecutivo(3) + "0"
"""
import pandas as pd

AUX = "/srv/mark02/analitica-final/aux"
coef = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_estaciones.csv")
ids = sorted(coef.STATION_ID.unique())
cne = pd.read_csv(f"{AUX}/cne.csv", dtype={"Codigo": str}, low_memory=False)
cne["cod_s"] = cne.Codigo.str.strip().str.lstrip("0")
cne = cne[cne.cod_s.str.len() == 8].copy()
cne = cne.drop_duplicates(subset="cod_s")
idx = cne.set_index("cod_s")

filas = []
for i in ids:
    s = str(i)
    cne8 = s[:4] + s[6:9] + "0"
    if cne8 in idx.index:
        r = idx.loc[cne8]
        filas.append(dict(STATION_ID=i, CNE8=cne8, NOMBRE=r.Nombre, CATEGORIA=r.Categoria,
                          DEPARTAMENTO=r.Departamento, MUNICIPIO=r.Municipio,
                          ALTITUD=r.Altitud, LON=r.LONGITUD, LAT=r.LATITUD,
                          ZONA_HIDRO=r["Zona Hidrografica"], SUBZONA=r["Subzona hidrografica"]))
    else:
        filas.append(dict(STATION_ID=i, CNE8=cne8, NOMBRE=None))

d = pd.DataFrame(filas)
ok = d.dropna(subset=["LAT"])
print(f"emparejadas: {len(ok)}/{len(d)}")
print("departamentos:", ok.DEPARTAMENTO.value_counts().to_dict())
print("categorias:", ok.CATEGORIA.value_counts().to_dict())
print("lon:", ok.LON.min(), ok.LON.max(), "| lat:", ok.LAT.min(), ok.LAT.max())
print("altitud:", ok.ALTITUD.min(), round(ok.ALTITUD.mean(), 1), ok.ALTITUD.max())
print("zonas hidro:", ok.ZONA_HIDRO.value_counts().to_dict())
# coherencia: primeros 2 digitos del id = area/zona (26 Cauca, 53/54 Pacifico)
ok2 = ok.copy()
ok2["z"] = ok2.STATION_ID.astype(str).str[:2]
print(pd.crosstab(ok2.z, ok2.ZONA_HIDRO).to_string())
# malla satelital
sat = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_satelite.csv", usecols=["LON", "LAT"])
print("malla sat lon:", sat.LON.min(), sat.LON.max(), "lat:", sat.LAT.min(), sat.LAT.max())
dentro = ((ok.LON >= sat.LON.min()) & (ok.LON <= sat.LON.max()) &
          (ok.LAT >= sat.LAT.min()) & (ok.LAT <= sat.LAT.max())).sum()
print("estaciones dentro de la malla satelital:", dentro, "de", len(ok))
d.to_csv(f"{AUX}/estaciones_cne_match.csv", index=False)
print("-> aux/estaciones_cne_match.csv")
