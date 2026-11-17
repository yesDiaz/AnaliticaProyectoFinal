import pandas as pd
AUX = "/srv/mark02/analitica-final/aux"
cne = pd.read_csv(f"{AUX}/cne.csv", dtype={"Codigo": str}, low_memory=False)
cne["cod_s"] = cne.Codigo.str.strip().str.lstrip("0")
v = cne[cne.Departamento.str.contains("Valle", case=False, na=False)].copy()
print("CNE estaciones Valle:", len(v), "| largos:", v.cod_s.str.len().value_counts().to_dict())
v["p4"] = v.cod_s.str[:4]
print("prefijos4 Valle (top20):", v.p4.value_counts().head(20).to_dict())
print("subzonas Valle (top12):", v["Subzona hidrografica"].value_counts().head(12).to_dict())
v8 = v[v.cod_s.str.len() == 8]
print("\nejemplo Valle p4=2612:", v8[v8.p4 == "2612"].cod_s.head(10).tolist())
print("ejemplo Valle p4=2622:", v8[v8.p4 == "2622"].cod_s.head(10).tolist())
print("ejemplo Valle p4=5420:", v8[v8.p4 == "5420"].cod_s.head(10).tolist())
# cuantos CNE del Valle hay por prefijo que usamos
import collections
coef = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_estaciones.csv")
ids = [str(i) for i in sorted(coef.STATION_ID.unique())]
pref = collections.Counter(i[:4] for i in ids)
print("\nproyecto p4:", dict(pref))
for p in sorted(pref):
    print(p, "proyecto:", pref[p], "| CNE Valle:", (v8.p4 == p).sum(), "| CNE total:", (cne.cod_s.str[:4] == p).sum())
