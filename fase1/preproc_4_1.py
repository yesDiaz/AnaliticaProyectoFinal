#!/usr/bin/env python3
# =====================================================================
# PROYECTO FINAL ANALITICA DE DATOS 2026-2S
# Fase 4.1 - Preprocesamiento y seleccion de datos (estaciones en tierra)
#
# Reglas explicitas, trazabilidad por paso y ninguna decision tomada con
# informacion del conjunto de prueba (la particion oficial es
# 1981-2005 / 2006-2009 / 2010-2013; todos los umbrales se fijan con
# criterios fisicos/climatologicos, no con estadisticos del test).
#
# Ejecutar:  python3 preproc_4_1.py
# =====================================================================
import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(BASE, "repo/Resultados/datos_unificados_ordenados.csv")
OUT = os.path.join(BASE, "entrega")
TAB = os.path.join(OUT, "tablas")
os.makedirs(TAB, exist_ok=True)

# --- Parametros de las reglas (justificados en el informe) ------------
ANIO_INI, ANIO_FIN = 1981, 2013     # periodo del proyecto
UMBRAL_IMPOSIBLE = 300.0            # mm/dia: por encima = imposible fisicamente
UMBRAL_REVISION = 200.0             # mm/dia: extremo plausible, se reporta
FRAC_TOTAL_MES = 0.90               # dia que concentra >=90% del mes
MIN_TOTAL_MES = 50.0                # y con al menos 50 mm
MIN_CEROS_MES = 25                  # y con >=25 dias del mes en cero
RACHA_CERO = 45                     # dias consecutivos en 0 = falta encubierta
MIN_DIAS_ANIO = 330                 # cobertura minima por estacion-anio (90.4%)
MAX_HUECO = 15                      # hueco consecutivo maximo admitido
MIN_ANIOS_EST = 15                  # anios validos minimos por estacion

traza = []


def registrar(paso, df, nota=""):
    """Agrega una fila a la tabla de trazabilidad."""
    val = df[df.VALUE.notna()]
    traza.append(dict(
        paso=paso,
        registros=len(df),
        registros_validos=len(val),
        estaciones=val.STATION_ID.nunique(),
        pares_estacion_anio=val.groupby(["STATION_ID", "YEAR"]).ngroups,
        nota=nota,
    ))
    print(f"[{paso}] registros={len(df)} validos={len(val)} "
          f"estaciones={val.STATION_ID.nunique()} "
          f"pares={val.groupby(['STATION_ID','YEAR']).ngroups}  {nota}")


# =====================================================================
# Paso 0. Carga y recorte al periodo del proyecto
# =====================================================================
df = pd.read_csv(IN_CSV, dtype={"STATION_ID": str}, encoding="utf-8-sig")
for c in ["YEAR", "MONTH", "DAY", "VALUE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"crudo total: {len(df)} registros, {df.STATION_ID.nunique()} estaciones, "
      f"{int(df.YEAR.min())}-{int(df.YEAR.max())}")
cobertura_real = (int(df.YEAR.min()), int(df.YEAR.max()))

df = df[(df.YEAR >= ANIO_INI) & (df.YEAR <= ANIO_FIN)].copy()
registrar("0. crudo 1981-2013", df, f"estaciones disponibles solo hasta {cobertura_real[1]}")

# =====================================================================
# Paso 1. Fechas invalidas (errores de digitacion en MONTH/DAY)
# =====================================================================
fecha = pd.to_datetime(
    dict(year=df.YEAR, month=df.MONTH, day=df.DAY), errors="coerce")
n_fecha_mala = int(fecha.isna().sum())
df["FECHA"] = fecha
df = df[df.FECHA.notna()].copy()
df["DOY"] = df.FECHA.dt.dayofyear
registrar("1. fechas validas", df, f"eliminados {n_fecha_mala} registros con fecha imposible")

# duplicados estacion-fecha
dup = df.duplicated(subset=["STATION_ID", "FECHA"], keep=False)
n_dup = int(dup.sum())
if n_dup:
    # se conserva la mediana del dia duplicado (criterio robusto)
    df = (df.groupby(["STATION_ID", "FECHA"], as_index=False)
            .agg(YEAR=("YEAR", "first"), MONTH=("MONTH", "first"),
                 DAY=("DAY", "first"), DOY=("DOY", "first"),
                 VALUE=("VALUE", "median")))
registrar("1b. sin duplicados", df, f"{n_dup} registros duplicados estacion-fecha colapsados")

# =====================================================================
# Paso 2. Valores imposibles y errores de digitacion
#   Regla: P < 0  -> imposible (no existe lluvia negativa)
#          P > 300 mm/dia -> imposible en el Valle del Cauca
#          (los maximos diarios historicos registrados en el valle
#           geografico del Cauca no superan ~250 mm; se deja margen)
#   Se marcan como faltante (NaN), no se imputan todavia.
# =====================================================================
n_neg = int((df.VALUE < 0).sum())
n_imp = int((df.VALUE > UMBRAL_IMPOSIBLE).sum())
extremos = df[df.VALUE > UMBRAL_REVISION].sort_values("VALUE", ascending=False)
extremos.to_csv(os.path.join(TAB, "4_1_valores_extremos.csv"), index=False)

df.loc[(df.VALUE < 0) | (df.VALUE > UMBRAL_IMPOSIBLE), "VALUE"] = np.nan
registrar("2. valores imposibles", df,
          f"{n_neg} negativos y {n_imp} > {UMBRAL_IMPOSIBLE} mm -> NaN")

# =====================================================================
# Paso 3. Totales mensuales registrados en un solo dia
#   Test: en una estacion-mes existe un dia d tal que
#         VALUE[d] >= 0.90 * suma del mes,  VALUE[d] >= 50 mm  y
#         al menos 25 dias del mes estan en 0.
#   El mes completo se marca como faltante: no se sabe como se reparte.
# =====================================================================
g = df.groupby(["STATION_ID", "YEAR", "MONTH"])["VALUE"]
agg = g.agg(suma="sum", maxi="max", ceros=lambda s: int((s == 0).sum()),
            n="size").reset_index()

# 3a. el dia concentra >=90% del mes y el resto del mes esta en ceros
sosp_a = agg[(agg.maxi >= FRAC_TOTAL_MES * agg.suma) &
             (agg.maxi >= MIN_TOTAL_MES) &
             (agg.ceros >= MIN_CEROS_MES)].assign(regla="3a_mes_en_ceros")

# 3b. el maximo del mes cae el ULTIMO dia del mes y concentra >=50% del total.
#     Patron tipico de la planilla mensual cerrada el ultimo dia: el
#     observador anota el acumulado del mes en la casilla final.
ult = df.loc[df.groupby(["STATION_ID", "YEAR", "MONTH"]).DAY.idxmax(),
             ["STATION_ID", "YEAR", "MONTH", "DAY", "VALUE"]].rename(
    columns={"DAY": "ultimo_dia", "VALUE": "val_ultimo"})
agg2 = agg.merge(ult, on=["STATION_ID", "YEAR", "MONTH"])
sosp_b = agg2[(agg2.val_ultimo >= 0.50 * agg2.suma) &
              (agg2.val_ultimo >= MIN_TOTAL_MES * 2) &
              (agg2.val_ultimo >= agg2.maxi)].assign(regla="3b_total_ultimo_dia")

sosp = pd.concat([sosp_a, sosp_b], ignore_index=True).drop_duplicates(
    subset=["STATION_ID", "YEAR", "MONTH"])
sosp.to_csv(os.path.join(TAB, "4_1_totales_mensuales_en_un_dia.csv"), index=False)
key = df.set_index(["STATION_ID", "YEAR", "MONTH"]).index
mask_sosp = key.isin(sosp.set_index(["STATION_ID", "YEAR", "MONTH"]).index)
n_meses = len(sosp)
df.loc[mask_sosp, "VALUE"] = np.nan
registrar("3. totales mensuales en un dia", df,
          f"{n_meses} estacion-mes anulados "
          f"({len(sosp_a)} por regla 3a, {len(sosp_b)} por regla 3b fin de mes)")

# =====================================================================
# Paso 4. Ceros que son datos faltantes
#   Test: racha de >= 45 dias consecutivos con VALUE == 0.
#   En el regimen bimodal del Valle del Cauca ni la temporada seca
#   (jun-ago) alcanza 45 dias seguidos sin lluvia medible; una racha asi
#   indica estacion sin operar con el 0 usado como relleno.
# =====================================================================
df = df.sort_values(["STATION_ID", "FECHA"]).reset_index(drop=True)
es_cero = (df.VALUE == 0).to_numpy()
est = df.STATION_ID.to_numpy()
# id de bloque: cambia al cambiar de estacion o de condicion
corte = np.empty(len(df), dtype=bool)
corte[0] = True
corte[1:] = (es_cero[1:] != es_cero[:-1]) | (est[1:] != est[:-1])
bloque = np.cumsum(corte)
largo = pd.Series(bloque).map(pd.Series(bloque).value_counts()).to_numpy()
mask_racha = es_cero & (largo >= RACHA_CERO)
n_racha = int(mask_racha.sum())
rachas = (df[mask_racha].assign(BLOQUE=bloque[mask_racha])
          .groupby(["STATION_ID", "BLOQUE"])
          .agg(inicio=("FECHA", "min"), fin=("FECHA", "max"), dias=("FECHA", "size"))
          .reset_index())
rachas.to_csv(os.path.join(TAB, "4_1_rachas_cero_sospechosas.csv"), index=False)
df.loc[mask_racha, "VALUE"] = np.nan
registrar("4. ceros falsos", df,
          f"{len(rachas)} rachas >= {RACHA_CERO} dias ({n_racha} registros) -> NaN")

# =====================================================================
# Paso 5. Dias faltantes, huecos y cobertura minima por estacion-anio
# =====================================================================
def diagnostico_par(sub):
    dias_anio = 366 if (sub.YEAR.iloc[0] % 4 == 0 and
                        (sub.YEAR.iloc[0] % 100 != 0 or sub.YEAR.iloc[0] % 400 == 0)) else 365
    doy_ok = set(sub.loc[sub.VALUE.notna(), "DOY"])
    n_ok = len(doy_ok)
    faltan = sorted(set(range(1, dias_anio + 1)) - doy_ok)
    hueco = 0
    actual = 0
    prev = None
    for d in faltan:
        actual = actual + 1 if prev is not None and d == prev + 1 else 1
        hueco = max(hueco, actual)
        prev = d
    return pd.Series(dict(dias_validos=n_ok, dias_anio=dias_anio,
                          pct=100.0 * n_ok / dias_anio, hueco_max=hueco))

pares = (df.groupby(["STATION_ID", "YEAR"])[["YEAR", "DOY", "VALUE"]]
           .apply(diagnostico_par).reset_index())
pares["acepta"] = (pares.dias_validos >= MIN_DIAS_ANIO) & (pares.hueco_max <= MAX_HUECO)
pares.to_csv(os.path.join(TAB, "4_1_cobertura_estacion_anio.csv"), index=False)
print(f"\npares evaluados={len(pares)} aceptados={int(pares.acepta.sum())} "
      f"rechazados={int((~pares.acepta).sum())}")

ok = pares[pares.acepta].set_index(["STATION_ID", "YEAR"]).index
df = df[df.set_index(["STATION_ID", "YEAR"]).index.isin(ok)].copy()
registrar("5. cobertura minima", df,
          f">= {MIN_DIAS_ANIO} dias validos y hueco <= {MAX_HUECO} dias")

# =====================================================================
# Paso 6. Estaciones con historia suficiente
#   Se exigen >= 15 anios validos: el FPCA y la particion temporal
#   necesitan curvas de la misma estacion en varios anios.
# =====================================================================
anios_est = df.groupby("STATION_ID").YEAR.nunique()
est_ok = anios_est[anios_est >= MIN_ANIOS_EST].index
df = df[df.STATION_ID.isin(est_ok)].copy()
registrar("6. estaciones con historia", df, f">= {MIN_ANIOS_EST} anios validos")

# =====================================================================
# Paso 7. Imputacion de huecos cortos (<= 15 dias) para cerrar la curva
#   Se interpola linealmente en el dia del anio SOLO dentro del anio y
#   solo en huecos <= MAX_HUECO (los pares con huecos mayores ya se
#   eliminaron). La interpolacion no usa datos de otros anios ni del
#   conjunto de prueba.
# =====================================================================
antes_na = int(df.VALUE.isna().sum())
df = df.sort_values(["STATION_ID", "FECHA"])
df["VALUE"] = (df.groupby(["STATION_ID", "YEAR"])["VALUE"]
                 .transform(lambda s: s.interpolate(limit=MAX_HUECO,
                                                    limit_direction="both")))
df["VALUE"] = df.VALUE.clip(lower=0)
despues_na = int(df.VALUE.isna().sum())
registrar("7. imputacion huecos cortos", df,
          f"{antes_na - despues_na} valores interpolados, quedan {despues_na} NaN")

# =====================================================================
# Salidas
# =====================================================================
traza_df = pd.DataFrame(traza)
traza_df.to_csv(os.path.join(TAB, "4_1_trazabilidad.csv"), index=False)
print("\n===== TABLA DE TRAZABILIDAD =====")
print(traza_df.to_string(index=False))

sal = df[["STATION_ID", "YEAR", "MONTH", "DAY", "DOY", "VALUE"]].copy()
sal.to_csv(os.path.join(OUT, "estaciones_depuradas.csv"), index=False)
print(f"\nguardado: entrega/estaciones_depuradas.csv ({len(sal)} filas)")

resumen = dict(
    estaciones=sal.STATION_ID.nunique(),
    pares=sal.groupby(["STATION_ID", "YEAR"]).ngroups,
    anio_min=int(sal.YEAR.min()), anio_max=int(sal.YEAR.max()),
    media_mm_dia=round(float(sal.VALUE.mean()), 3),
    pct_ceros=round(100.0 * float((sal.VALUE == 0).mean()), 2),
)
print("resumen final:", resumen)
pd.DataFrame([resumen]).to_csv(os.path.join(TAB, "4_1_resumen_final.csv"), index=False)

# ---------------------------------------------------------------------
# Reporte de zona hidrografica (2 primeros digitos del codigo IDEAM).
# La malla satelital cubre lon -77.55..-75.70 y lat 3.05..5.00; las
# estaciones de zonas distintas de la del rio Cauca (26) deben revisarse
# contra sus coordenadas antes del emparejamiento estacion-pixel.
# ---------------------------------------------------------------------
zona = (sal.assign(ZONA=sal.STATION_ID.str[:2])
           .groupby("ZONA")
           .agg(estaciones=("STATION_ID", "nunique"),
                pares=("YEAR", "size")).reset_index())
zona.to_csv(os.path.join(TAB, "4_1_zonas_hidrograficas.csv"), index=False)
print("\nzonas hidrograficas presentes:")
print(zona.to_string(index=False))

