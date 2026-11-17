import pandas as pd
coef = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_estaciones.csv")
ids = [str(i) for i in sorted(coef.STATION_ID.unique())]
cne = pd.read_csv("/srv/mark02/analitica-final/aux/cne.csv", dtype={"Codigo": str}, low_memory=False)
cne["cod_s"] = cne.Codigo.str.strip().str.lstrip("0")
setc = set(cne.cod_s)

reglas = {
    "A id[:4]+id[-3:]+0": lambda i: i[:4] + i[-3:] + "0",
    "B id[:4]+id[-4:]": lambda i: i[:4] + i[-4:],
    "C id[:4]+id[-3:].zfill(4)": lambda i: i[:4] + i[-3:].zfill(4),
    "D id[:6]+id[-4:]": lambda i: i[:6] + i[-4:],
    "E id[:4]+id[4:8]": lambda i: i[:4] + i[4:8],
    "F id[:4]+id[-4:-1]+'0'": lambda i: i[:4] + i[-4:-1] + "0",
}
for nom, f in reglas.items():
    ok = [i for i in ids if f(i) in setc]
    print(f"{nom}: {len(ok)}/{len(ids)}")

# detalle regla A: cuales fallan
f = reglas["A id[:4]+id[-3:]+0"]
fallan = [i for i in ids if f(i) not in setc]
print("fallan A:", len(fallan), fallan[:20])
# para los que fallan, ver que codigos CNE existen con ese prefijo de 4
for i in fallan[:6]:
    pref = i[:4]
    cands = sorted(c for c in setc if c.startswith(pref) and len(c) == 8)
    print(i, "->", f(i), "| CNE cercanos:", [c for c in cands if c[4:7].lstrip('0') == i[-3:].lstrip('0')][:5], "| total cands:", len(cands))
