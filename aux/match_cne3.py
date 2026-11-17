import pandas as pd, numpy as np, re
coef = pd.read_csv("/srv/mark02/analitica-final/entrega/coef_estaciones.csv")
ids = [str(i) for i in sorted(coef.STATION_ID.unique())]
cne = pd.read_csv("/srv/mark02/analitica-final/aux/cne.csv", dtype={"Codigo": str}, low_memory=False)
cne["cod_s"] = cne.Codigo.str.strip().str.lstrip("0")
c10 = cne[cne.cod_s.str.len() == 10]
print("CNE 10 digitos, zona 26 ejemplos:", sorted(set(c10.loc[c10.cod_s.str.startswith('2610'), 'cod_s']))[:20])
print("CNE 10 digitos, zona 26 (cualquier):", sorted(set(c10.loc[c10.cod_s.str.startswith('26'), 'cod_s']))[:20])
# estructura: comparar sufijos
print("ids proyecto zona 26:", [i for i in ids if i.startswith('2610')][:10])
# intento: ultimos 2 digitos = categoria? ver distribucion
print("ultimos2 proyecto:", pd.Series([i[-2:] for i in ids]).value_counts().head(10).to_dict())
print("prefijos4 proyecto:", pd.Series([i[:4] for i in ids]).value_counts().head(15).to_dict())
# en CNE: columna Codigo de 8 digitos -> estructura ZZSS NNNN
c8 = cne[cne.cod_s.str.len() == 8]
print("CNE8 prefijos 2610:", sorted(set(c8.loc[c8.cod_s.str.startswith('2610'), 'cod_s'])))
print("CNE8 ultimos2 dist:", pd.Series([c[-2:] for c in c8.cod_s]).value_counts().head(6).to_dict())
# hipotesis: proyecto = ZZSS + 000 + N  -> probar quitar ceros intermedios
def alt(i):
    return i[:4] + i[4:].lstrip('0')
print("alt ejemplos:", [(i, alt(i)) for i in ids[:6]])
setc = set(cne.cod_s)
print("match alt:", sum(1 for i in ids if alt(i) in setc))
