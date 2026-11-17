#!/usr/bin/env python3
"""Diagnostico inicial de los datos de estaciones (previo a 4.1)."""
import pandas as pd, numpy as np, os

BASE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(BASE, "repo/Resultados/datos_unificados_ordenados.csv")

df = pd.read_csv(F, dtype={"STATION_ID": str}, encoding="utf-8-sig")
print("filas:", len(df), "| columnas:", list(df.columns))
for c in ["YEAR", "MONTH", "DAY", "VALUE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n-- rango temporal --")
print("YEAR min/max:", df.YEAR.min(), df.YEAR.max())
print("estaciones unicas:", df.STATION_ID.nunique())
print("pares estacion-anio:", df.groupby(['STATION_ID','YEAR']).ngroups)

print("\n-- VALUE --")
print(df.VALUE.describe())
print("NaN:", int(df.VALUE.isna().sum()), "| negativos:", int((df.VALUE < 0).sum()))
print("ceros:", int((df.VALUE == 0).sum()), f"({100*(df.VALUE==0).mean():.1f}%)")
for u in [100, 200, 300, 500, 1000]:
    print(f"  > {u} mm: {int((df.VALUE > u).sum())}")
print("top 10 valores:", sorted(df.VALUE.dropna().unique())[-10:])

print("\n-- periodo del proyecto 1981-2013 --")
sub = df[(df.YEAR >= 1981) & (df.YEAR <= 2013)]
print("filas:", len(sub), "| estaciones:", sub.STATION_ID.nunique())
print("registros por anio:")
print(sub.groupby("YEAR").size().to_string())

print("\n-- estaciones con mas cobertura 1981-2003 --")
s2 = df[(df.YEAR >= 1981) & (df.YEAR <= 2003)]
cob = s2.groupby("STATION_ID").agg(n=("VALUE","size"), anios=("YEAR","nunique"),
                                    nonan=("VALUE", lambda x: x.notna().sum()))
cob["pct_dias"] = 100*cob.n/(cob.anios*365.25)
print(cob.sort_values("n", ascending=False).head(15).to_string())
print("\nestaciones con >=20 anios en 1981-2003:", int((cob.anios >= 20).sum()))
print("estaciones con >=15 anios:", int((cob.anios >= 15).sum()))

print("\n-- dias por estacion-anio (1981-2003) --")
dpa = s2.groupby(["STATION_ID","YEAR"]).size()
print(dpa.describe())
print("pares con >=365 dias:", int((dpa >= 365).sum()), "| >=330:", int((dpa>=330).sum()),
      "| <100:", int((dpa<100).sum()))
