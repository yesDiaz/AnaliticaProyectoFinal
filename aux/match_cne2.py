import pandas as pd, numpy as np
coef = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_estaciones.csv")
ids = [str(i) for i in sorted(coef.STATION_ID.unique())]
cne = pd.read_csv("/srv/mark02/analitica-final/aux/cne.csv", dtype={"Codigo": str}, low_memory=False)
cne["cod"] = cne.Codigo.str.strip()
cne["cod_s"] = cne.cod.str.lstrip("0")
print("largos CNE:", cne.cod.str.len().value_counts().to_dict())
print("largos sin ceros:", cne.cod_s.str.len().value_counts().to_dict())
print("ejemplos zona 26:", cne.loc[cne.cod_s.str.startswith("26"), "cod"].head(10).tolist())
print("ids proyecto:", ids[:6])
# prueba A: primeros 8 digitos del id del proyecto
pref8 = {i: i[:8] for i in ids}
setc = set(cne.cod_s)
print("match pref8:", sum(1 for v in pref8.values() if v in setc))
# prueba B: id completo (10) sin ceros
print("match full:", sum(1 for i in ids if i in setc))
# prueba C: id[:-2] con padding
print("ejemplo buscado:", [c for c in cne.cod_s if c.startswith("261000")][:10])
print("ejemplo 26 todos:", sorted(set(c[:4] for c in cne.cod_s if c.startswith("26")))[:20])
