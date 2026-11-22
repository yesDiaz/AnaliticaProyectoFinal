# -*- coding: utf-8 -*-
"""
Proyecto final de Analítica de Datos 2026-2S (UAO) — Fases 0 (preprocesamiento v2), 1, 2, 3, 3b y 4 (red funcional).
Precipitación en el Valle del Cauca: estaciones en tierra (diario) vs producto satelital (péntada).

Ejecución (desde la carpeta entrega/):
    ../venv/bin/python proyecto_final.py            # todas las fases
    ../venv/bin/python proyecto_final.py fase2      # solo una fase (usa entrega/cache/)

Reproducibilidad
- Rutas relativas a este archivo: ../datos (datos del curso), ./datos_externos (ONI de NOAA-CPC),
  ./cache (intermedios), ./figuras y ./tablas (salidas).
- Semilla fija SEMILLA = 2026 (solo se usa en permutaciones y remuestreos).
- Versiones usadas (se imprimen y se guardan en tablas/versiones_python.csv):
  Python 3.12, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, matplotlib 3.11.2, openpyxl 3.1.5,
  Pillow (lectura de los GeoTIFF), scikit-fda 0.10.1 (solo fase 2 y 3, verificación).
- Fase 1 (construcción de los datos funcionales) NO usa librerías de datos funcionales: la base
  B-spline (Cox-de Boor), la factorización de Cholesky, los mínimos cuadrados, VC, VCG y la matriz
  de suavizado están programados aquí. numpy solo se usa como contenedor de arreglos (productos de
  matrices). scipy aparece UNA vez en fase 1, en una verificación externa de Φ (no en la construcción).

Mapa figura/tabla -> función: buscar "Fig" o "Tab" en este archivo (cada salida lleva su número).
"""
import os, sys, glob, json, time, platform
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "..", "datos")
EXT = os.path.join(AQUI, "datos_externos")
CACHE = os.path.join(AQUI, "cache")
FIG = os.path.join(AQUI, "figuras")
TAB = os.path.join(AQUI, "tablas")
for d in (CACHE, FIG, TAB):
    os.makedirs(d, exist_ok=True)

SEMILLA = 2026
ANIO_INI, ANIO_FIN = 1981, 2013          # periodo del enunciado
ANIO_EST_FIN = 2003                       # las estaciones solo traen datos hasta 2003
ANIO_TRAIN_FIN = 1996                     # partición nueva (asesoría 20-nov): 1981-1996 entrenamiento+validación, 1997-2003 prueba
ANIO_REF_FIN = 1995                       # años de referencia para umbrales del control de calidad
                                          # (entrenamiento en ambas particiones posibles: no hay fuga)
DIAS = 365                                # dominio anual común [0, 365] en días (29-feb se descarta)
T_FINO = np.arange(DIAS) + 0.5            # malla de evaluación: centro de cada día del año

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_EST, C_SAT = "#1f4e79", "#c8691c"       # colores fijos: estaciones (azul), satélite (naranja)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guarda_fig(fig, nombre):
    ruta = os.path.join(FIG, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    log("figura", nombre)


def guarda_tab(df, nombre, **kw):
    df.to_csv(os.path.join(TAB, nombre), index=kw.get("index", False))
    log("tabla", nombre, df.shape)


# Calendario de 365 días (sin 29 de febrero): día del año y péntadas
DIAS_MES = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
INICIO_MES = np.concatenate([[0], np.cumsum(DIAS_MES)[:-1]])   # día (0-index) de inicio de cada mes


def doy365(mes, dia):
    """Día del año 1..365 en calendario sin 29-feb."""
    return INICIO_MES[np.asarray(mes) - 1] + np.asarray(dia)


def pentadas_365():
    """Para cada péntada k=1..72: inicio (0-index), longitud en días (calendario 365) y punto medio."""
    ini, lon = [], []
    for m in range(12):
        for j in range(6):
            a = 5 * j
            b = 5 * j + 5 if j < 5 else DIAS_MES[m]
            ini.append(INICIO_MES[m] + a)
            lon.append(b - a)
    ini, lon = np.array(ini), np.array(lon)
    return ini, lon, ini + lon / 2.0


PENT_INI, PENT_LON, T_PENT = pentadas_365()


def fecha_txt(t):
    """Convierte t en [0,365] a 'dd-mmm'."""
    d = int(np.clip(np.floor(t), 0, 364))
    m = np.searchsorted(INICIO_MES, d, side="right") - 1
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    return f"{d - INICIO_MES[m] + 1:02d}-{meses[m]}"


def eje_meses(ax):
    ax.set_xticks(INICIO_MES + DIAS_MES / 2)
    ax.set_xticklabels(list("EFMAMJJASOND"))
    ax.set_xlim(0, DIAS)


# =====================================================================================
# FASE 0. Lectura, inventario y preprocesamiento de las estaciones
# =====================================================================================

def lee_estaciones():
    """Lee los tres archivos de estaciones por bloques (el xlsx en modo streaming) y guarda
    una copia compacta en cache/estaciones_crudo.pkl."""
    ruta = os.path.join(CACHE, "estaciones_crudo.pkl")
    if os.path.exists(ruta):
        return pd.read_pickle(ruta)
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(DATOS, "P_1930_1979.xlsx"), read_only=True)
    filas = []
    for i, r in enumerate(wb["Hoja1"].iter_rows(values_only=True)):
        if i == 0:
            continue                                   # STATION_ID, SENSORNAME, YEAR, MONTH, DAY, VALUE, UNITS
        filas.append((r[0], r[2], r[3], r[4], r[5]))
    x = pd.DataFrame(filas, columns=["STATION_ID", "YEAR", "MONTH", "DAY", "VALUE"])
    x["fuente"] = "P_1930_1979.xlsx"
    partes = [x]
    for f in ("P_1980_1990.csv", "P_1991_2003.csv"):
        for bloque in pd.read_csv(os.path.join(DATOS, f), chunksize=500_000):
            bloque["fuente"] = f
            partes.append(bloque)
    t = pd.concat(partes, ignore_index=True)
    t["VALUE"] = pd.to_numeric(t["VALUE"], errors="coerce").astype("float64")
    for c in ("STATION_ID", "YEAR", "MONTH", "DAY"):
        t[c] = t[c].astype("int64")
    t.to_pickle(ruta)
    return t


def rachas(valor_bool):
    """Devuelve (inicio, longitud) de las rachas de True de un vector booleano."""
    v = np.concatenate([[False], valor_bool, [False]]).astype(np.int8)
    d = np.diff(v)
    ini = np.where(d == 1)[0]
    fin = np.where(d == -1)[0]
    return ini, fin - ini


def fase0():
    log("FASE 0: preprocesamiento")
    crudo = lee_estaciones()
    # ---- Inventario básico (Tab. 1)
    inv = (crudo.groupby("fuente")
           .agg(filas=("VALUE", "size"), estaciones=("STATION_ID", "nunique"),
                anio_min=("YEAR", "min"), anio_max=("YEAR", "max"),
                valor_max=("VALUE", "max"), no_enteros=("VALUE", lambda v: int(((v % 1) != 0).sum())),
                negativos=("VALUE", lambda v: int((v < 0).sum())), nulos=("VALUE", lambda v: int(v.isna().sum())))
           .reset_index())
    guarda_tab(inv, "Tab01_inventario_estaciones.csv")

    traz = []   # tabla de trazabilidad (Tab. 2)

    def registra(paso, df_valido, detalle, n_afectados=np.nan):
        """df_valido: filas con dato válido (no NaN) dentro de 1981-2003."""
        v = df_valido[df_valido["VALUE"].notna()]
        traz.append({"paso": paso, "estaciones": v["STATION_ID"].nunique(),
                     "pares_estacion_anio": v.groupby(["STATION_ID", "YEAR"]).ngroups,
                     "dias_validos": len(v), "ceros": int((v["VALUE"] == 0).sum()), "valores_afectados": n_afectados, "regla": detalle})
        log(paso, traz[-1]["estaciones"], traz[-1]["pares_estacion_anio"], n_afectados)

    # Registros de 1930-1980 fuera del periodo: NO se usan para construir curvas; solo los años
    # <= 1995 (incluido 1930-1980) sirven de referencia para los umbrales del control de calidad.
    d = crudo.copy()
    d = d[~((d.MONTH == 2) & (d.DAY == 29))]          # calendario de 365 días
    d["doy"] = doy365(d.MONTH.values, d.DAY.values)
    d = d.sort_values(["STATION_ID", "YEAR", "doy"]).reset_index(drop=True)
    d["bandera"] = ""
    d["v_orig"] = d.VALUE.copy()                      # valor crudo (para la prueba cero ≠ faltante)
    per = (d.YEAR >= ANIO_INI) & (d.YEAR <= ANIO_FIN)
    ref = d.YEAR <= ANIO_REF_FIN
    registra("P0 crudo (1981-2013; 29-feb excluido)", d[per], "Registros tal como vienen en los archivos")

    # ---- P1: valores imposibles: negativos, no numéricos o > 1000 mm/día
    imp = (d.VALUE < 0) | d.VALUE.isna() | (d.VALUE > 1000)
    ejemplos_imp = d[imp & per]
    d.loc[imp, "bandera"] = "imposible"
    d.loc[imp, "VALUE"] = np.nan
    registra("P1 valores imposibles", d[per], "VALUE<0, no numérico o >1000 mm/día -> NaN", int((imp & per).sum()))

    # ---- P2: totales mensuales registrados en un solo día
    # Un mes es "total en un día" si: (a) tiene un único día con lluvia >0 y los demás días
    # registrados en 0; (b) ese valor es >= 50 mm y >= 50 % del total mensual climatológico de
    # la estación para ese mes; (c) en los años de referencia la estación registra en ese mes una
    # mediana >= 5 días con lluvia (un mes con un solo día lluvioso es improbable allí).
    d["ym"] = d.YEAR * 100 + d.MONTH
    d["humedo"] = (d.VALUE > 0).astype(int)
    d["v0"] = d.VALUE.fillna(-1.0)
    mes = (d.groupby(["STATION_ID", "YEAR", "MONTH"])
           .agg(n=("VALUE", "count"), tot=("VALUE", "sum"), humedos=("humedo", "sum"),
                vmax=("VALUE", "max"), dia_max=("v0", "idxmax"))
           .reset_index())
    mref = mes[(mes.YEAR <= ANIO_REF_FIN) & (mes.n >= 25)]
    clim = (mref.groupby(["STATION_ID", "MONTH"])
            .agg(tot_med=("tot", "median"), hum_med=("humedos", "median")).reset_index())
    mes = mes.merge(clim, on=["STATION_ID", "MONTH"], how="left")
    tm = ((mes.humedos == 1) & (mes.vmax >= 50) & (mes.vmax >= 0.5 * mes.tot_med) & (mes.hum_med >= 5))
    tm_meses = mes[tm].copy()
    tm_meses["dia_del_mes"] = d.loc[tm_meses.dia_max.values, "DAY"].values
    guarda_tab(tm_meses[["STATION_ID", "YEAR", "MONTH", "dia_del_mes", "vmax", "tot_med", "hum_med"]]
               .rename(columns={"vmax": "valor_mm", "tot_med": "total_mensual_climatologico",
                                "hum_med": "dias_lluvia_climatologicos"}),
               "Tab03_totales_mensuales_un_dia.csv")
    clave_tm = set(zip(tm_meses.STATION_ID, tm_meses.YEAR, tm_meses.MONTH))
    marca = pd.Series(list(zip(d.STATION_ID, d.YEAR, d.MONTH))).isin(clave_tm).values
    d.loc[marca, "bandera"] = "total_mensual"
    d.loc[marca, "VALUE"] = np.nan                     # el mes completo pasa a faltante
    registra("P2 totales mensuales en un día", d[per],
             "Mes con 1 solo día >0 (>=50 mm y >=50 % del total climatológico) donde la estación suele "
             "tener >=5 días de lluvia -> mes completo NaN", int((marca & per.values).sum()))

    # ---- P3: errores de digitación: valor > 3 x percentil 99,9 de los días lluviosos de la estación
    #      (calculado con los años de referencia) y > 150 mm
    q999 = (d[ref & (d.VALUE > 0)].groupby("STATION_ID").VALUE.quantile(0.999).rename("q999"))
    d = d.merge(q999, left_on="STATION_ID", right_index=True, how="left")
    dig = (d.VALUE > 3 * d.q999) & (d.VALUE > 150)
    guarda_tab(d[dig][["STATION_ID", "YEAR", "MONTH", "DAY", "VALUE", "q999"]], "Tab04_digitacion_sospechosa.csv")
    d.loc[dig, "bandera"] = "digitacion"
    d.loc[dig, "VALUE"] = np.nan
    registra("P3 errores de digitación", d[per], "VALUE > 3·P99.9(estación, años<=1995) y >150 mm -> NaN",
             int((dig & per).sum()))

    # ---- P3b: meses con escala errónea (p. ej. coma decimal desplazada en todo un mes): total mensual
    #      > 5 x percentil 95 de los totales mensuales de la estación en los años de referencia
    mtot = d.groupby(["STATION_ID", "YEAR", "MONTH"]).VALUE.agg(tot="sum", n="count").reset_index()
    p95 = (mtot[(mtot.YEAR <= ANIO_REF_FIN) & (mtot.n >= 25)].groupby("STATION_ID").tot.quantile(0.95).rename("p95m"))
    mtot = mtot.merge(p95, left_on="STATION_ID", right_index=True, how="left")
    mal = mtot[mtot.tot > 5 * mtot.p95m]
    guarda_tab(mal, "Tab04b_meses_escala_erronea.csv")
    clave = set(zip(mal.STATION_ID, mal.YEAR, mal.MONTH))
    marca = pd.Series(list(zip(d.STATION_ID, d.YEAR, d.MONTH))).isin(clave).values
    d.loc[marca, "bandera"] = "mes_escala_erronea"
    d.loc[marca, "VALUE"] = np.nan
    registra("P3b meses con escala errónea", d[per], "Total mensual > 5·P95 de los totales mensuales de la estación (años<=1995) -> mes NaN",
             int((marca & per.values).sum()))

    # ---- P3c: bloques copiados (error de transcripción): una ventana de 10 días consecutivos con al
    #      menos 6 días de lluvia que aparece idéntica en otra fecha de la misma estación. La
    #      probabilidad de que 10 valores de lluvia se repitan por azar es despreciable; como no se
    #      sabe cuál copia es la original, ambas pasan a NaN.
    from numpy.lib.stride_tricks import sliding_window_view
    VENT = 10
    dup = np.zeros(len(d), bool); casos = []
    for sid, idx in d.groupby("STATION_ID").indices.items():
        v = d.VALUE.values[idx]
        if len(v) < 2 * VENT:
            continue
        Sv = sliding_window_view(v, VENT)
        ok_ = (np.nan_to_num(Sv) > 0).sum(1) >= 6
        vistos = {}
        for i in np.where(ok_)[0]:
            k = Sv[i].tobytes()
            if k in vistos and i - vistos[k] >= VENT:
                j = vistos[k]
                dup[idx[i:i + VENT]] = True; dup[idx[j:j + VENT]] = True
                casos.append({"STATION_ID": sid, "anio_original": int(d.YEAR.values[idx[j]]), "doy_original": int(d.doy.values[idx[j]]),
                              "anio_copia": int(d.YEAR.values[idx[i]]), "doy_copia": int(d.doy.values[idx[i]]), "desfase_dias": int(i - j)})
            else:
                vistos.setdefault(k, i)
    guarda_tab(pd.DataFrame(casos), "Tab04c_bloques_copiados.csv")
    d.loc[dup, "bandera"] = "bloque_copiado"
    d.loc[dup, "VALUE"] = np.nan
    registra("P3c bloques de 10 días copiados", d[per], "Ventana de 10 días (>=6 con lluvia) idéntica en otra fecha de la estación -> ambas NaN",
             int((dup & per.values).sum()))

    # =================================================================================
    # Desde aquí: VERSIÓN 2 del preprocesamiento (asesorías del 19 y 20-nov-2026).
    # Cambios: rachas de ceros con 2 vecinas a <= 25 km (P4), inventario de % de faltantes por
    # estación y año, comparación OMM / NOAA / nacional, regla anual (<= 25 % faltantes) separada de
    # la regla del periodo (> 45 % de datos), altitud SRTM 90 m validada, relación lluvia-altura,
    # discriminación por altura + radio y comparación IDW / CSW / kriging ordinario.
    # =================================================================================
    d = d[per & (d.YEAR <= ANIO_EST_FIN)].reset_index(drop=True)
    cat = catalogo_estaciones()
    lon0, lon1, lat0, lat1 = -77.55, -75.70, 3.05, 5.00
    geo = (cat[(cat.lon > lon0) & (cat.lon < lon1) & (cat.lat > lat0) & (cat.lat < lat1)]
           .drop_duplicates("station").set_index("station"))
    NA = ANIO_EST_FIN - ANIO_INI + 1
    est_all = np.sort(d.STATION_ID.unique())
    ne = len(est_all)
    col = (d.YEAR.values - ANIO_INI) * DIAS + d.doy.values - 1
    fil = np.searchsorted(est_all, d.STATION_ID.values)
    V = np.full((ne, NA * DIAS), np.nan); V[fil, col] = d.VALUE.values          # tras P1-P3c
    V0 = np.full((ne, NA * DIAS), np.nan); V0[fil, col] = d.v_orig.values       # crudo (antes de P1)
    BAN = np.full((ne, NA * DIAS), "", dtype=object); BAN[fil, col] = d.bandera.values
    es_geo = np.isin(est_all, geo.index)
    ig = np.where(es_geo)[0]
    lon_e = geo.lon.reindex(est_all).values; lat_e = geo.lat.reindex(est_all).values
    alt_e = geo.alt.reindex(est_all).values
    region_e = np.where(est_all // 10**8 >= 50, "Pacífico", "Andina (Cauca)")
    DIST = haversine_km(lon_e[:, None], lat_e[:, None], lon_e[None, :], lat_e[None, :])
    DIST[~es_geo, :] = np.nan; DIST[:, ~es_geo] = np.nan
    anio_col = ANIO_INI + np.arange(NA * DIAS) // DIAS
    TRAIN = anio_col <= ANIO_TRAIN_FIN                                            # 1981-1996

    def registra_V(paso, M_, detalle, n_af, filas=None):
        filas = np.arange(ne) if filas is None else filas
        Mv = M_[filas].reshape(len(filas), NA, DIAS)
        ok_ = np.isfinite(Mv)
        traz.append({"paso": paso, "estaciones": int(ok_.any((1, 2)).sum()), "pares_estacion_anio": int(ok_.any(2).sum()),
                     "dias_validos": int(ok_.sum()), "ceros": int((Mv == 0).sum()), "valores_afectados": n_af, "regla": detalle})
        log(paso, traz[-1]["estaciones"], traz[-1]["pares_estacion_anio"], n_af)

    # ---- P4 (v2): rachas de >= 30 días seguidos en 0 mm. Se comparan con las 2 estaciones más
    #      cercanas a <= 25 km (con dato en >= 80 % de los días de la racha). Si las vecinas llovieron
    #      (media >= 1 mm/día y >= 20 % de días con > 1 mm) la racha es falla del sensor -> NaN (se
    #      imputa). Si las vecinas también estuvieron secas es sequía real -> se conserva. Sin vecinas:
    #      falla solo si hay "alzas" (>= 10 mm en los 5 días antes Y en los 5 días después).
    R_VEC, N_VEC, MIN_RACHA = 25.0, 2, 30
    filas_r = []; falsos = np.zeros_like(V, bool)
    for i in ig:
        ini, lon_ = rachas(V[i] == 0)
        for a, L in zip(ini, lon_):
            if L < MIN_RACHA:
                continue
            b = a + L
            orden_ = np.argsort(np.nan_to_num(DIST[i], nan=1e9))
            cand = [j for j in orden_ if j != i and DIST[i, j] <= R_VEC and np.isfinite(V[j, a:b]).mean() >= 0.8][:N_VEC]
            w0, w1 = V[i, max(a - 5, 0):a], V[i, b:b + 5]
            pre = np.nanmax(w0) if np.isfinite(w0).any() else np.nan
            post = np.nanmax(w1) if np.isfinite(w1).any() else np.nan
            alzas = bool(pre >= 10) and bool(post >= 10)
            if cand:
                pv = float(np.mean([np.nanmean(V[j, a:b]) for j in cand]))
                fw = float(np.mean([np.mean(V[j, a:b][np.isfinite(V[j, a:b])] > 1) for j in cand]))
                dec = "falla_sensor" if (pv >= 1.0 and fw >= 0.2) else "sequia_real"
            else:
                pv = fw = np.nan
                dec = "falla_sensor_sin_vecinas" if alzas else "sequia_real_sin_vecinas"
            if dec.startswith("falla"):
                falsos[i, a:b] = True
            y0 = int(anio_col[a])
            filas_r.append({"STATION_ID": est_all[i], "region": region_e[i], "inicio": f"{y0}-{fecha_txt(a % DIAS)}",
                            "fin": f"{int(anio_col[b - 1])}-{fecha_txt((b - 1) % DIAS)}", "longitud_dias": int(L), "col_inicio": int(a),
                            "vecinas": ";".join(str(est_all[j]) for j in cand),
                            "dist_vecinas_km": ";".join(f"{DIST[i, j]:.1f}" for j in cand),
                            "dalt_vecinas_m": ";".join(f"{abs(alt_e[j] - alt_e[i]):.0f}" for j in cand),
                            "lluvia_vecinas_mm_dia": pv, "frac_dias_lluvia_vecinas": fw,
                            "max_5d_antes_mm": pre, "max_5d_despues_mm": post, "alzas_antes_y_despues": alzas,
                            "decision": dec})
    Tr = pd.DataFrame(filas_r)
    guarda_tab(Tr, "Tab05_rachas_cero_30d_vecinas.csv")
    V[falsos] = np.nan; BAN[falsos] = "racha_cero_falla"
    registra_V("P4 rachas >= 30 ceros (2 vecinas <= 25 km)", V,
               "Racha >=30 días en 0 mm; vecinas con lluvia (>=1 mm/día y >=20 % días >1 mm) -> falla -> NaN; vecinas secas -> sequía real (se conserva)",
               int(falsos.sum()))

    # ---- Prueba "cero ≠ faltante": todo 0 mm del crudo que terminó como NaN debe tener una bandera
    #      de control de calidad documentada; ningún paso convierte ceros en faltantes por ser ceros.
    cero_a_nan = (V0 == 0) & np.isnan(V)
    sin_bandera = int((cero_a_nan & (BAN == "")).sum())
    assert sin_bandera == 0, f"{sin_bandera} ceros pasaron a NaN sin regla"
    racha_max = []
    for i in ig:
        ini, lon_ = rachas(V[i] == 0)
        if len(lon_):
            k = lon_.argmax(); racha_max.append((est_all[i], int(lon_[k]), int(anio_col[ini[k]])))
    rm = pd.DataFrame(racha_max, columns=["station", "dias", "anio"]).sort_values("dias", ascending=False)
    pc = pd.Series(BAN[cero_a_nan]).value_counts()
    guarda_tab(pd.DataFrame([{"ceros_crudo_1981_2003": int((V0 == 0).sum()), "ceros_conservados": int((V == 0).sum()),
                              "ceros_a_NaN_total": int(cero_a_nan.sum()),
                              **{f"ceros_a_NaN_por_{k}": int(v) for k, v in pc.items()},
                              "ceros_a_NaN_sin_regla (debe ser 0)": sin_bandera,
                              "racha_seca_conservada_mas_larga_dias": int(rm.dias.iloc[0]),
                              "estacion_racha_mas_larga": int(rm.station.iloc[0]), "anio_racha_mas_larga": int(rm.anio.iloc[0]),
                              "estaciones_con_racha_seca_conservada>=60d": int((rm.dias >= 60).sum())}]),
              "Tab02c_prueba_cero_no_es_faltante.csv")

    # ---- P5: inventario de estaciones idóneas: % de faltantes por estación y por año (calendario
    #      completo 1981-2003; 0 mm cuenta como dato).
    Vy = V.reshape(ne, NA, DIAS)
    nfal = np.isnan(Vy).sum(2)
    pfal = 100 * nfal / DIAS
    maxcons = np.zeros((ne, NA), int)
    for i in range(ne):
        for y in range(NA):
            _, l_ = rachas(np.isnan(Vy[i, y])); maxcons[i, y] = l_.max() if len(l_) else 0
    inv_w = pd.DataFrame(np.round(pfal, 1), columns=[str(a) for a in range(ANIO_INI, ANIO_EST_FIN + 1)])
    inv_w.insert(0, "STATION_ID", est_all); inv_w.insert(1, "region", region_e); inv_w.insert(2, "con_coordenadas", es_geo)
    inv_w["pct_datos_1981_2003"] = np.round(100 - pfal.mean(1), 1)
    guarda_tab(inv_w, "Tab06_faltantes_pct_estacion_anio.csv")

    # ---- P6: comparación de reglas. Regla ANUAL (¿el año sirve?) separada de la regla del PERIODO
    #      (¿la estación sirve?).
    anio_omm = (nfal <= 10) & (maxcons <= 5)
    n_omm = anio_omm.sum(1)
    pct_datos = 100 - pfal.mean(1)
    anio_25 = pfal <= 25
    reglas = [
        ("OMM", "año: <=10 faltantes y <=5 seguidos", "estación: >=80 % de años válidos (>=19 de 23)", n_omm >= np.ceil(0.8 * NA), anio_omm),
        ("NOAA", "año: igual que OMM", "estación: >=10 años válidos", n_omm >= 10, anio_omm),
        ("Nacional (IDEAM)", "año: <=25 % faltantes (se imputa)", "estación: >45 % de datos en 1981-2003", pct_datos > 45, anio_25),
        ("Solo regla anual del profesor", "año: <=25 % faltantes (se imputa)", "estación: >=1 año utilizable", anio_25.any(1), anio_25),
    ]
    filas_reg = []
    for nom, ra, rp, sel_e, sel_a in reglas:
        sg = sel_e & es_geo
        filas_reg.append({"regla": nom, "regla_anual": ra, "regla_periodo": rp,
                          "estaciones_todas": int(sel_e.sum()), "estaciones_con_coord": int(sg.sum()),
                          "Pacifico_con_coord": int((sg & (region_e == "Pacífico")).sum()),
                          "Andina_con_coord": int((sg & (region_e != "Pacífico")).sum()),
                          "pares_estacion_anio_utilizables": int((sel_a & sg[:, None]).sum())})
    filas_reg.append({"regla": "universo (con coordenadas en la malla)", "estaciones_todas": ne, "estaciones_con_coord": int(es_geo.sum()),
                      "Pacifico_con_coord": int((es_geo & (region_e == "Pacífico")).sum()),
                      "Andina_con_coord": int((es_geo & (region_e != "Pacífico")).sum())})
    TR = pd.DataFrame(filas_reg)
    guarda_tab(TR, "Tab06c_comparacion_reglas_estaciones.csv")
    # Regla ELEGIDA: periodo = nacional (> 45 % de datos) con coordenadas; año = <= 25 % faltantes.
    sel_est = (pct_datos > 45) & es_geo
    sel_pair = anio_25 & sel_est[:, None]
    registra_V("P6 regla del periodo: >45 % de datos (nacional) y con coordenadas", np.where(sel_est[:, None], V, np.nan),
               "Estación con >45 % de días con dato en 1981-2003 y coordenadas CVC dentro de la malla satelital",
               int((~sel_est & np.isfinite(V).any(1)).sum()))
    Vsel = np.where(np.repeat(sel_pair, DIAS, axis=1), V, np.nan)
    registra_V("P7 regla anual: año con <=25 % de faltantes", Vsel,
               "Año-estación con <=25 % de días faltantes (>=274 con dato); los demás años se descartan (no se imputan)",
               int((sel_est[:, None] & ~anio_25 & np.isfinite(Vy).any(2)).sum()))
    resumen = pd.DataFrame({"STATION_ID": est_all, "region": region_e, "con_coordenadas": es_geo, "lon": lon_e, "lat": lat_e,
                            "alt_m": alt_e, "alt_fuente": geo.alt_fuente.reindex(est_all).values,
                            "pct_datos_1981_2003": np.round(pct_datos, 1), "anios_validos_OMM": n_omm,
                            "anios_con_<=25%_faltantes": anio_25.sum(1), "OMM": reglas[0][3], "NOAA": reglas[1][3],
                            "Nacional": reglas[2][3], "ELEGIDA": sel_est})
    guarda_tab(resumen, "Tab06b_resumen_estaciones_reglas.csv")

    # ---- Altitud: catálogo cuando existe; si falta, SRTM 90 m. Validación SRTM vs catálogo.
    va = geo.dropna(subset=["alt_catalogo", "alt_srtm90m"])
    e_ = va.alt_srtm90m - va.alt_catalogo
    guarda_tab(pd.DataFrame([{"estaciones_con_ambas": len(va), "sesgo_m(SRTM-catalogo)": e_.mean(), "MAE_m": e_.abs().mean(),
                              "RMSE_m": np.sqrt((e_ ** 2).mean()), "mediana_abs_m": e_.abs().median(), "max_abs_m": e_.abs().max(),
                              "%_dentro_100m": 100 * (e_.abs() <= 100).mean(), "r_Pearson": va.alt_srtm90m.corr(va.alt_catalogo),
                              "estaciones_con_alt_SRTM_imputada": int((geo.alt_fuente == "SRTM 90 m").sum()),
                              "elegidas_con_alt_SRTM": int((geo.alt_fuente.reindex(est_all[sel_est]) == "SRTM 90 m").sum())}]),
              "Tab06d_validacion_altitud_srtm.csv")

    # ---- Relación lluvia-altura (estaciones elegidas; total anual medio con años <= 25 % faltantes)
    tot_anual = np.where(anio_25, np.nanmean(Vy, 2) * DIAS, np.nan)
    tot_e = np.nanmean(np.where(sel_est[:, None], tot_anual, np.nan), 1)
    la = pd.DataFrame({"station": est_all, "alt": alt_e, "total": tot_e, "region": region_e})[sel_est].dropna()
    from scipy import stats
    filas_la = []
    for nom, s_ in (("todas", la), ("Andina (Cauca)", la[la.region != "Pacífico"]), ("Pacífico", la[la.region == "Pacífico"])):
        x_, y_ = s_.alt.values, s_.total.values
        r1 = y_ - np.polyval(np.polyfit(x_, y_, 1), x_); r2 = y_ - np.polyval(np.polyfit(x_, y_, 2), x_)
        n_ = len(x_); sst = ((y_ - y_.mean()) ** 2).sum()
        F = ((r1 ** 2).sum() - (r2 ** 2).sum()) / ((r2 ** 2).sum() / (n_ - 3))
        filas_la.append({"grupo": nom, "estaciones": n_, "pendiente_mm_por_100m": 100 * np.polyfit(x_, y_, 1)[0],
                         "R2_lineal": 1 - (r1 ** 2).sum() / sst, "R2_cuadratico": 1 - (r2 ** 2).sum() / sst,
                         "p_termino_cuadratico(F)": 1 - stats.f.cdf(F, 1, n_ - 3),
                         "Pearson": stats.pearsonr(x_, y_)[0], "Spearman": stats.spearmanr(x_, y_)[0],
                         "p_Spearman": stats.spearmanr(x_, y_)[1]})
    guarda_tab(pd.DataFrame(filas_la), "Tab06e_relacion_lluvia_altura.csv")

    # ---- Correlograma espacial de la lluvia diaria (solo años de entrenamiento 1981-1996): define el
    #      radio de búsqueda, la banda de altura y el modelo del kriging (sin fuga hacia el test).
    Vt = np.where(TRAIN[None, :], V, np.nan)
    CORR = np.full((ne, ne), np.nan)
    filas_c = []
    for a_i, i in enumerate(ig):
        for j in ig[a_i + 1:]:
            if DIST[i, j] > 150:
                continue
            m_ = np.isfinite(Vt[i]) & np.isfinite(Vt[j])
            if m_.sum() < 365:
                continue
            r_ = np.corrcoef(Vt[i, m_], Vt[j, m_])[0, 1]
            CORR[i, j] = CORR[j, i] = r_
            filas_c.append((DIST[i, j], abs(alt_e[i] - alt_e[j]), r_, region_e[i] == region_e[j]))
    cc = pd.DataFrame(filas_c, columns=["dist_km", "dalt_m", "r", "misma_region"])
    BANDA_ALT = 400.0
    cb = cc[cc.dalt_m <= BANDA_ALT]
    mejor = (np.inf, None, None)
    for a_ in np.arange(1, 300, 0.5):
        e_ = np.exp(-cb.dist_km.values / a_); c0 = (e_ @ cb.r.values) / (e_ @ e_)
        sse = ((cb.r.values - c0 * e_) ** 2).sum()
        if sse < mejor[0]:
            mejor = (sse, a_, c0)
    _, A_RANGO, C0 = mejor
    C0 = min(C0, 0.999)
    bins = [0, 10, 20, 30, 50, 75, 100, 150]
    cc["dist_bin"] = pd.cut(cc.dist_km, bins); cc["banda_alt"] = pd.cut(cc.dalt_m, [-1, 400, 900, 5000], labels=["<=400 m", "400-900 m", ">900 m"])
    tabc = cc.groupby(["dist_bin", "banda_alt"], observed=True).r.agg(["mean", "size"]).reset_index()
    # Radio de acción: mayor límite de clase de distancia en que la correlación media de pares con
    # |Δz| <= 400 m sigue >= 0,30 (más allá la vecina explica < 10 % de la varianza diaria).
    rb = tabc[(tabc.banda_alt == "<=400 m")]
    R_IDW = float(max([b.right for b, m in zip(rb.dist_bin, rb["mean"]) if m >= 0.30] + [10]))
    R_KRIG = 2 * R_IDW
    tabc["dist_bin"] = tabc.dist_bin.astype(str)
    tabc = pd.concat([tabc, pd.DataFrame([{"dist_bin": "MODELO", "banda_alt": f"r(h)={C0:.3f}·exp(-h/{A_RANGO:.1f} km); radio IDW/CSW={R_IDW:.1f} km; radio kriging={R_KRIG:.1f} km; banda={BANDA_ALT:.0f} m"}])])
    guarda_tab(tabc, "Tab06f_correlograma_distancia_altura.csv")
    log(f"correlograma: c0={C0:.3f} a={A_RANGO:.1f} km -> R_IDW={R_IDW:.1f} km, R_KRIG={R_KRIG:.1f} km")

    # ---- Imputación. Discriminación previa: 1) diferencia de altura |Δz| <= 400 m (distancia en el eje
    #      vertical); 2) distancia radial horizontal <= radio. Donantes = estaciones con coordenadas que
    #      tengan dato ESE MISMO DÍA (los periodos secos/lluviosos son concurrentes). Máx. 6 donantes.
    #      IDW: w = 1/d². CSW: w = r² (correlación diaria 1981-1996, incorpora la similitud real).
    #      Kriging ordinario: covarianza C(h) = c0·exp(-h/a) con C(0) = 1 (pepita 1-c0), radio doble.
    NMAX = 6

    def donantes(i, radio):
        c_ = [j for j in ig if j != i and DIST[i, j] <= radio and abs(alt_e[j] - alt_e[i]) <= BANDA_ALT]
        return np.array(sorted(c_, key=lambda j: DIST[i, j]), int)

    def pesos(i, us, metodo):
        if metodo == "IDW":
            w = 1 / np.maximum(DIST[i, us], 0.1) ** 2
        elif metodo == "CSW":
            w = np.clip(np.nan_to_num(CORR[i, us]), 0, None) ** 2
            if w.sum() <= 0:
                return None
        else:
            Cm = C0 * np.exp(-DIST[np.ix_(us, us)] / A_RANGO); np.fill_diagonal(Cm, 1.0)
            n_ = len(us); A_ = np.ones((n_ + 1, n_ + 1)); A_[:n_, :n_] = Cm; A_[n_, n_] = 0
            b_ = np.append(C0 * np.exp(-DIST[i, us] / A_RANGO), 1.0)
            w = np.linalg.solve(A_, b_)[:n_]
            return w
        return w / w.sum()

    def imputa(i, cols, metodo):
        out = np.full(len(cols), np.nan)
        cand = donantes(i, R_KRIG if metodo == "kriging" else R_IDW)
        if len(cand) == 0 or len(cols) == 0:
            return out
        Vc = V[np.ix_(cand, cols)]
        A_ = np.isfinite(Vc)
        use = A_ & (np.cumsum(A_, 0) <= NMAX)
        pat, inv_ = np.unique(use.T, axis=0, return_inverse=True)
        for k, p_ in enumerate(pat):
            if not p_.any():
                continue
            w = pesos(i, cand[p_], metodo)
            if w is None:
                continue
            kk = np.where(inv_.ravel() == k)[0]
            out[kk] = np.maximum(w @ Vc[np.ix_(np.where(p_)[0], kk)], 0.0)
        return out

    # Validación (dejar-uno-fuera en el tiempo de entrenamiento): se predice cada día OBSERVADO de las
    # estaciones elegidas en 1981-1996 con los donantes de ese día (la propia estación nunca es donante).
    METODOS = ("IDW", "CSW", "kriging")
    val = []
    for i in np.where(sel_est)[0]:
        cols = np.where(TRAIN & np.isfinite(V[i]))[0]
        pr = {m: imputa(i, cols, m) for m in METODOS}
        val.append(pd.DataFrame({"i": i, "col": cols, "obs": V[i, cols], **pr}))
    val = pd.concat(val, ignore_index=True)
    val["region"] = region_e[val.i.values]
    filas_v = []
    comunes = val.dropna(subset=list(METODOS))
    comunes = comunes.assign(bloque=comunes.i.astype(np.int64) * 10**6 + comunes.col // 30)
    agg = comunes.groupby("bloque").agg(n=("obs", "size"), obs=("obs", "sum"), **{m: (m, "sum") for m in METODOS}, region=("region", "first"))
    agg = agg[agg.n == 30]
    for reg in ("todas", "Andina (Cauca)", "Pacífico"):
        s_ = comunes if reg == "todas" else comunes[comunes.region == reg]
        a_ = agg if reg == "todas" else agg[agg.region == reg]
        for m in METODOS:
            e_ = s_[m] - s_.obs; e30 = a_[m] - a_.obs
            filas_v.append({"region": reg, "metodo": m, "dias_validados": len(s_), "MAE_mm_dia": e_.abs().mean(),
                            "RMSE_mm_dia": np.sqrt((e_ ** 2).mean()), "sesgo_mm_dia": e_.mean(), "r_diaria": s_[m].corr(s_.obs),
                            "acierto_seco_humedo_%": 100 * ((s_[m] > 1) == (s_.obs > 1)).mean(),
                            "bloques_30d": len(a_), "RMSE_total_30d_mm": np.sqrt((e30 ** 2).mean()),
                            "sesgo_total_30d_%": 100 * (a_[m].sum() / a_.obs.sum() - 1), "r_total_30d": a_[m].corr(a_.obs)})
    TV = pd.DataFrame(filas_v)
    TV["cobertura_%_dias_con_donantes"] = TV.metodo.map(lambda m: 100 * val[m].notna().mean())
    guarda_tab(TV, "Tab06g_validacion_imputacion_IDW_CSW_kriging.csv")
    tv0 = TV[TV.region == "todas"].set_index("metodo")
    METODO = tv0.RMSE_total_30d_mm.idxmin()
    log("imputación: método elegido por menor RMSE del total de 30 días =", METODO)

    # ---- P8: imputación de los años elegidos (método elegido; si no hay donantes en el radio,
    #      kriging con espacio de búsqueda doble; si tampoco, el día queda NaN y la fase 1 lo omite).
    Vimp = Vsel.copy(); FUENTE = np.zeros_like(V, np.int8)        # 0 observado/NaN, 1 método, 2 kriging ampliado
    for i in np.where(sel_est)[0]:
        cols = np.where(np.isnan(Vsel[i]) & np.repeat(sel_pair[i], DIAS))[0]
        v1 = imputa(i, cols, METODO)
        Vimp[i, cols] = v1; FUENTE[i, cols[np.isfinite(v1)]] = 1
        if METODO != "kriging":
            rest = cols[np.isnan(v1)]
            v2 = imputa(i, rest, "kriging")
            Vimp[i, rest] = v2; FUENTE[i, rest[np.isfinite(v2)]] = 2
    n_imp = int((FUENTE > 0).sum())
    registra_V(f"P8 imputación ({METODO}; respaldo kriging radio {R_KRIG:.0f} km)", Vimp,
               f"Días faltantes de los años elegidos imputados con {METODO} (|Δz|<=400 m, d<={R_IDW:.0f} km); sin donantes -> kriging d<={R_KRIG:.0f} km; si no, NaN",
               n_imp)
    filas_i = []
    for reg in ("Andina (Cauca)", "Pacífico"):
        r_ = sel_est & (region_e == reg)
        celdas = np.repeat(sel_pair & r_[:, None], DIAS, axis=1)
        filas_i.append({"region": reg, "estaciones": int(r_.sum()), "pares_estacion_anio": int((sel_pair & r_[:, None]).sum()),
                        "dias_en_pares": int(celdas.sum()), "imputados_metodo": int(((FUENTE == 1) & celdas).sum()),
                        "imputados_kriging_ampliado": int(((FUENTE == 2) & celdas).sum()),
                        "%_imputado": 100 * float(((FUENTE > 0) & celdas).sum()) / max(int(celdas.sum()), 1),
                        "dias_sin_imputar_NaN": int((np.isnan(Vimp) & celdas).sum())})
    guarda_tab(pd.DataFrame(filas_i), "Tab06h_imputacion_por_region.csv")

    T = pd.DataFrame(traz)
    guarda_tab(T, "Tab02_trazabilidad.csv")
    pi, py = np.where(sel_pair)
    Y = Vimp.reshape(ne, NA, DIAS)[pi, py]
    np.savez_compressed(os.path.join(CACHE, "fase0_matriz_diaria.npz"), Y=Y, station=est_all[pi], year=ANIO_INI + py,
                        imputado=(FUENTE.reshape(ne, NA, DIAS)[pi, py] > 0), Y_sin_imputar=Vy[pi, py], metodo=METODO)
    # copia en texto para el script de R (fases 1-3 en R parten de la misma matriz imputada)
    pd.concat([pd.DataFrame({"station": est_all[pi], "year": ANIO_INI + py}), pd.DataFrame(np.round(Y, 4), columns=[f"d{k+1}" for k in range(DIAS)])],
              axis=1).to_csv(os.path.join(CACHE, "fase0_matriz_diaria.csv.gz"), index=False)
    guarda_tab(pd.Series(BAN[BAN != ""]).value_counts().rename_axis("bandera").reset_index(name="n"), "Tab02b_banderas_qc.csv")

    # ---- Unidades del producto satelital: péntada = total (mm/péntada) o promedio (mm/día)?
    cubo, anio_s, pent_s, lon_s, lat_s = lee_satelite()
    filas_u = []
    for i in np.where(sel_est)[0]:
        fi_, co_ = np.abs(lat_s - lat_e[i]).argmin(), np.abs(lon_s - lon_e[i]).argmin()
        for y in range(NA):
            if not sel_pair[i, y]:
                continue
            sat = cubo[(anio_s == ANIO_INI + y), fi_, co_][np.argsort(pent_s[anio_s == ANIO_INI + y])]
            if len(sat) != 72:
                continue
            for k in range(72):
                v_ = Vy[i, y, PENT_INI[k]:PENT_INI[k] + PENT_LON[k]]
                if np.isfinite(v_).all():
                    filas_u.append((sat[k], v_.sum(), v_.mean()))
    U = np.array(filas_u)
    guarda_tab(pd.DataFrame([{"pentadas_comparadas": len(U), "media_satelite": U[:, 0].mean(),
                              "media_suma_estacion_mm_pentada": U[:, 1].mean(), "media_promedio_estacion_mm_dia": U[:, 2].mean(),
                              "razon_sat/suma": U[:, 0].mean() / U[:, 1].mean(), "razon_sat/promedio": U[:, 0].mean() / U[:, 2].mean(),
                              "valores_-9999_en_cubo": int((cubo == -9999).sum()), "nan_en_cubo": int(np.isnan(cubo).sum()),
                              "valor_max_satelite": float(np.nanmax(cubo)), "%_valores_iguales_al_max": 100 * float((cubo == np.nanmax(cubo)).mean()),
                              "todos_enteros": bool((np.nan_to_num(cubo) % 1 == 0).all())}]),
              "Tab06i_unidades_satelite.csv")

    # ---- Fig. 1: inventario (% faltantes estación-año) y trazabilidad
    orden_e = np.lexsort((-pct_datos, region_e != "Pacífico", ~es_geo))
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 1.4]})
    im_ = ax[0].imshow(pfal[orden_e], aspect="auto", interpolation="nearest", cmap="magma_r", vmin=0, vmax=100,
                       extent=[ANIO_INI - 0.5, ANIO_EST_FIN + 0.5, ne, 0])
    yy, xx = np.where(sel_pair[orden_e])
    ax[0].scatter(ANIO_INI + xx, yy + 0.5, s=1.2, color="#2b83ba", marker="s", lw=0)
    n_pac = int((es_geo & (region_e == "Pacífico")).sum()); n_geo = int(es_geo.sum())
    ax[0].axhline(n_pac, color="w", lw=0.8); ax[0].axhline(n_geo, color="w", lw=0.8)
    ax[0].text(ANIO_EST_FIN + 0.7, n_pac / 2, "Pacífico", rotation=90, va="center", fontsize=7)
    ax[0].text(ANIO_EST_FIN + 0.7, (n_pac + n_geo) / 2, "Andina", rotation=90, va="center", fontsize=7)
    ax[0].text(ANIO_EST_FIN + 0.7, (n_geo + ne) / 2, "sin coordenadas", rotation=90, va="center", fontsize=7)
    plt.colorbar(im_, ax=ax[0], label="% de días faltantes en el año", fraction=0.04)
    ax[0].set_xlabel("Año"); ax[0].set_ylabel("Estación"); ax[0].set_title("Inventario: % faltantes por estación y año (punto azul = par usado)", fontsize=8)
    ax[1].barh(range(len(T)), T.pares_estacion_anio, color=C_EST)
    ax[1].set_yticks(range(len(T))); ax[1].set_yticklabels([p.split(" ")[0] for p in T.paso])
    ax[1].invert_yaxis(); ax[1].set_xlabel("Pares estación-año"); ax[1].set_title("Trazabilidad")
    for k, v_ in enumerate(T.pares_estacion_anio):
        ax[1].text(v_, k, f" {v_}", va="center", fontsize=7)
    guarda_fig(fig, "Fig01_inventario_faltantes_trazabilidad.png")

    # ---- Fig. 2: una racha declarada falla y una declarada sequía real, con sus vecinas
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a_, dec in zip(ax, ("falla_sensor", "sequia_real")):
        s_ = Tr[Tr.decision == dec]
        if not len(s_):
            continue
        e = s_.sort_values("lluvia_vecinas_mm_dia", ascending=(dec == "sequia_real")).iloc[0]
        i = np.searchsorted(est_all, e.STATION_ID)
        y0 = int(e.inicio[:4]); c0_ = (y0 - ANIO_INI) * DIAS
        tt = np.arange(c0_, min(c0_ + 2 * DIAS, NA * DIAS))
        for j in [int(v) for v in str(e.vecinas).split(";") if v and v != "nan"]:
            jj = np.searchsorted(est_all, j)
            a_.plot(tt - c0_, V[jj, tt], lw=0.6, alpha=0.8, label=f"vecina {j} ({DIST[i, jj]:.0f} km)")
        a_.plot(tt - c0_, V0[i, tt], color="k", lw=0.8, label=f"estación {e.STATION_ID}")
        ini_c = int(e.col_inicio) - c0_
        a_.axvspan(ini_c, ini_c + e.longitud_dias, color="#f4a582", alpha=0.4)
        a_.set_title(f"{dec}: {e.longitud_dias} días desde {e.inicio}", fontsize=8); a_.set_xlabel(f"día desde 1-ene-{y0}")
        a_.legend(fontsize=6)
    ax[0].set_ylabel("mm/día")
    guarda_fig(fig, "Fig02_rachas_cero_vecinas.png")

    # ---- Fig. 2b: estaciones retenidas por cada regla (mapa)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for a_, (nom, ra, rp, sel_e, _) in zip(ax, reglas[:3]):
        a_.scatter(lon_e[es_geo], lat_e[es_geo], s=8, color="#cccccc", label="descartada")
        for reg, col_ in (("Andina (Cauca)", C_EST), ("Pacífico", "#2e8b57")):
            m_ = sel_e & es_geo & (region_e == reg)
            a_.scatter(lon_e[m_], lat_e[m_], s=10, color=col_, label=f"{reg}: {m_.sum()}")
        a_.set_title(f"{nom}: {int((sel_e & es_geo).sum())} estaciones", fontsize=8); a_.legend(fontsize=6); a_.set_xlabel("lon")
    ax[0].set_ylabel("lat")
    guarda_fig(fig, "Fig02b_reglas_estaciones_mapa.png")

    # ---- Fig. 2c: validación de la altitud SRTM y relación lluvia-altura
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(va.alt_catalogo, va.alt_srtm90m, s=8, color=C_EST); m_ = [0, va.alt_catalogo.max()]
    ax[0].plot(m_, m_, "k--", lw=0.7); ax[0].set_xlabel("altitud catálogo (m)"); ax[0].set_ylabel("altitud SRTM 90 m (m)")
    ax[0].set_title(f"SRTM vs catálogo (n={len(va)})", fontsize=8)
    for reg, col_ in (("Andina (Cauca)", C_EST), ("Pacífico", "#2e8b57")):
        s_ = la[la.region == reg]; ax[1].scatter(s_.alt, s_.total, s=10, color=col_, label=reg)
    xs_ = np.linspace(la.alt.min(), la.alt.max(), 100)
    ax[1].plot(xs_, np.polyval(np.polyfit(la.alt, la.total, 1), xs_), "k-", lw=0.8, label="lineal")
    ax[1].plot(xs_, np.polyval(np.polyfit(la.alt, la.total, 2), xs_), "k--", lw=0.8, label="cuadrática")
    ax[1].set_xlabel("altitud (m)"); ax[1].set_ylabel("total anual medio (mm)"); ax[1].legend(fontsize=6)
    ax[1].set_title("Relación lluvia-altura (estaciones elegidas)", fontsize=8)
    guarda_fig(fig, "Fig02c_altitud_srtm_lluvia_altura.png")

    # ---- Fig. 2d: correlograma por banda de altura y validación de la imputación
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for b_, col_ in (("<=400 m", C_EST), ("400-900 m", "#fdae61"), (">900 m", "#d7191c")):
        s_ = cc[cc.banda_alt == b_]; ax[0].scatter(s_.dist_km, s_.r, s=3, alpha=0.4, color=col_, label=f"|Δz| {b_}")
    hh = np.linspace(0, 150, 200); ax[0].plot(hh, C0 * np.exp(-hh / A_RANGO), "k-", lw=1, label="modelo kriging")
    ax[0].axvline(R_IDW, color="k", ls=":", lw=0.8); ax[0].set_xlabel("distancia (km)"); ax[0].set_ylabel("r lluvia diaria (1981-1996)")
    ax[0].legend(fontsize=6); ax[0].set_title("Correlograma espacial", fontsize=8)
    t_ = TV[TV.region == "todas"]
    ax[1].bar(np.arange(3) - 0.2, t_.RMSE_mm_dia, 0.4, color=C_EST, label="RMSE diario (mm/día)")
    ax2 = ax[1].twinx(); ax2.bar(np.arange(3) + 0.2, t_.RMSE_total_30d_mm, 0.4, color=C_SAT, label="RMSE total 30 d (mm)")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(t_.metodo); ax[1].set_title("Validación de la imputación (1981-1996)", fontsize=8)
    ax[1].legend(loc="upper left", fontsize=6); ax2.legend(loc="upper right", fontsize=6)
    guarda_fig(fig, "Fig02d_correlograma_validacion_imputacion.png")
    return T




# =====================================================================================
# FASE 1. Construcción de los datos funcionales PASO A PASO (sin librerías)
# =====================================================================================
# Notación: y_i (n_i x 1) observaciones de la curva i en los tiempos t_ij; Φ (n_i x K) matriz de
# la base B-spline evaluada en esos tiempos; ĉ_i = (Φ'Φ)^{-1} Φ' y_i ; S = Φ (Φ'Φ)^{-1} Φ'.

ORDEN = 4                                   # B-splines cúbicos (orden 4), como en clase


def nodos_bspline(K, a=0.0, b=float(DIAS), orden=ORDEN):
    """Vector de nodos: extremos repetidos 'orden' veces y K-orden nodos interiores equiespaciados."""
    interiores = np.linspace(a, b, K - orden + 2)[1:-1]
    return np.concatenate([np.repeat(a, orden), interiores, np.repeat(b, orden)])


def base_bspline(t, K, orden=ORDEN, deriv=0):
    """Matriz Φ (len(t) x K) por la recursión de Cox-de Boor, programada a mano.
    deriv=1 o 2 devuelve la derivada de las funciones base (fórmula de derivada de B-splines)."""
    t = np.asarray(t, float)
    tau = nodos_bspline(K, orden=orden)
    b = tau[-1]
    # orden 1: indicadoras de los intervalos [tau_i, tau_{i+1}); el extremo derecho se asigna al último
    n_int = len(tau) - 1
    B = np.zeros((len(t), n_int))
    for i in range(n_int):
        if tau[i] < tau[i + 1]:
            B[:, i] = (t >= tau[i]) & (t < tau[i + 1])
    ult = np.max(np.where(tau[:-1] < tau[1:])[0])
    B[t == b, ult] = 1.0
    # recursión: B_{i,k}(t) = (t-tau_i)/(tau_{i+k-1}-tau_i) B_{i,k-1} + (tau_{i+k}-t)/(tau_{i+k}-tau_{i+1}) B_{i+1,k-1}
    for k in range(2, orden - deriv + 1):
        Bn = np.zeros((len(t), n_int - k + 1))
        for i in range(n_int - k + 1):
            d1 = tau[i + k - 1] - tau[i]
            d2 = tau[i + k] - tau[i + 1]
            izq = (t - tau[i]) / d1 * B[:, i] if d1 > 0 else 0.0
            der = (tau[i + k] - t) / d2 * B[:, i + 1] if d2 > 0 else 0.0
            Bn[:, i] = izq + der
        B = Bn
    # derivadas: B'_{i,k} = (k-1) [B_{i,k-1}/(tau_{i+k-1}-tau_i) - B_{i+1,k-1}/(tau_{i+k}-tau_{i+1})]
    for k in range(orden - deriv + 1, orden + 1):
        Bn = np.zeros((len(t), n_int - k + 1))
        for i in range(n_int - k + 1):
            d1 = tau[i + k - 1] - tau[i]
            d2 = tau[i + k] - tau[i + 1]
            izq = B[:, i] / d1 if d1 > 0 else 0.0
            der = B[:, i + 1] / d2 if d2 > 0 else 0.0
            Bn[:, i] = (k - 1) * (izq - der)
        B = Bn
    assert B.shape[1] == K
    return B


def cholesky_mano(A):
    """Factorización A = L L' (A simétrica definida positiva), programada a mano."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for j in range(n):
        s = A[j, j] - np.dot(L[j, :j], L[j, :j])
        if s <= 0:
            raise np.linalg.LinAlgError("matriz no definida positiva")
        L[j, j] = np.sqrt(s)
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


def adelante(L, B):
    """Resuelve L X = B (L triangular inferior) por sustitución hacia adelante."""
    B = np.asarray(B, float)
    X = np.zeros_like(B)
    for i in range(L.shape[0]):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]
    return X


def atras(L, B):
    """Resuelve L' X = B por sustitución hacia atrás."""
    B = np.asarray(B, float)
    X = np.zeros_like(B)
    n = L.shape[0]
    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
    return X


def mco_mano(Phi, Y):
    """Mínimos cuadrados: ĉ = (Φ'Φ)^{-1}Φ'Y vía ecuaciones normales + Cholesky a mano.
    Y puede ser vector (n) o matriz (n x N, una curva por columna).
    Devuelve coeficientes, L (factor de Φ'Φ) y la diagonal de S (palancas h_jj)."""
    A = Phi.T @ Phi
    L = cholesky_mano(A)
    C = atras(L, adelante(L, Phi.T @ Y))
    Z = adelante(L, Phi.T)                     # Z = L^{-1} Φ'  ->  S = Z'Z, h_jj = ||Z[:,j]||^2
    h = (Z ** 2).sum(0)
    return C, L, h


def criterios(y, yhat, h, K):
    """ECM, VC (dejando uno fuera, fórmula exacta con la palanca) y VCG para un ajuste lineal."""
    n = len(y)
    r = y - yhat
    sse = float(r @ r)
    ecm = sse / n
    vc = float(np.mean((r / (1 - h)) ** 2))
    vcg = (sse / n) / ((n - K) / n) ** 2        # misma definición que fda::smooth.basis()$gcv
    return ecm, vc, vcg


def lee_satelite():
    """Cubo de péntadas (archivo, fila, columna) desde los GeoTIFF; cache/satelite_cubo.npz."""
    ruta = os.path.join(CACHE, "satelite_cubo.npz")
    if os.path.exists(ruta):
        z = np.load(ruta)
        return z["cubo"], z["anio"], z["pent"], z["lon"], z["lat"]
    from PIL import Image
    carpeta = os.path.join(CACHE, "sat_tif", "ImagenesSatelitePentadal")
    if not os.path.isdir(carpeta):
        import zipfile
        with zipfile.ZipFile(os.path.join(DATOS, "ImagenesSatelitePentadal.zip")) as zf:
            for nombre in zf.namelist():
                if "/._" not in nombre and nombre.endswith(".tif"):
                    zf.extract(nombre, os.path.join(CACHE, "sat_tif"))
    archivos = sorted(glob.glob(os.path.join(carpeta, "ppt*.tif")))
    cubo, anio, pent = [], [], []
    for f in archivos:
        im = Image.open(f)
        a = np.array(im).astype(float)
        a[a == -9999] = np.nan
        cubo.append(a)
        b = os.path.basename(f)
        anio.append(int(b[3:7])); pent.append(int(b[7:9]))
        geo = im.tag_v2.get(33922); res = im.tag_v2.get(33550)
    cubo = np.array(cubo)
    nf, nr, nc = cubo.shape[0], cubo.shape[1], cubo.shape[2]
    lon = geo[3] + res[0] * (np.arange(nc) + 0.5)
    lat = geo[4] - res[1] * (np.arange(nr) + 0.5)
    np.savez_compressed(ruta, cubo=cubo, anio=np.array(anio), pent=np.array(pent), lon=lon, lat=lat)
    return cubo, np.array(anio), np.array(pent), lon, lat


def matriz_satelite():
    """Curvas satelitales: una fila por (píxel, año) con 72 tasas en mm/día (mm por péntada / días
    de la péntada). Solo años completos (1981-2012); 2013 trae 24 péntadas y se excluye."""
    cubo, anio, pent, lon, lat = lee_satelite()
    anios = [a for a in np.unique(anio) if (anio == a).sum() == 72]
    nr, nc = cubo.shape[1], cubo.shape[2]
    filas, meta = [], []
    for a in anios:
        sel = np.where(anio == a)[0]
        sel = sel[np.argsort(pent[sel])]
        X = cubo[sel].reshape(72, -1).T          # (píxeles x 72) en mm/péntada
        bis = (a % 4 == 0)
        lon_p = PENT_LON.astype(float).copy()
        if bis:
            lon_p[11] += 1                       # péntada 12 (26-feb a fin de mes) tiene 4 días en años bisiestos
        filas.append(X / lon_p)
        ii, jj = np.divmod(np.arange(nr * nc), nc)
        meta.append(pd.DataFrame({"pixel": np.arange(nr * nc), "fila": ii, "col": jj,
                                  "lon": lon[jj], "lat": lat[ii], "year": a}))
    return np.vstack(filas), pd.concat(meta, ignore_index=True)


def fase1():
    log("FASE 1: datos funcionales paso a paso")
    z = np.load(os.path.join(CACHE, "fase0_matriz_diaria.npz"))
    Yd, est, anio_e = z["Y"], z["station"], z["year"]          # (N_e x 365) mm/día con NaN
    Xs, meta_s = matriz_satelite()                              # (N_s x 72) mm/día
    log("curvas estaciones", Yd.shape, "curvas satélite", Xs.shape)

    # ---- Verificación de la base construida a mano (Tab. 7): partición de la unidad y comparación
    #      externa con scipy (solo verificación; la construcción no la usa)
    from scipy.interpolate import BSpline
    ver = []
    for K in (6, 12, 20):
        Phi = base_bspline(T_FINO, K)
        ref_ = BSpline.design_matrix(T_FINO, nodos_bspline(K), ORDEN - 1).toarray()
        d1 = base_bspline(T_FINO, K, deriv=1)
        ref1 = np.column_stack([BSpline(nodos_bspline(K), np.eye(K)[k], ORDEN - 1).derivative()(T_FINO) for k in range(K)])
        ver.append({"K": K, "max|sum_k phi_k(t) - 1|": np.abs(Phi.sum(1) - 1).max(),
                    "max|Phi_mano - Phi_scipy|": np.abs(Phi - ref_).max(),
                    "max|Dphi_mano - Dphi_scipy|": np.abs(d1 - ref1).max()})
    guarda_tab(pd.DataFrame(ver), "Tab07_verificacion_base.csv")

    # ---- Selección de K con ECM, VC y VCG (Tab. 8, Fig. 4)
    K_grid = list(range(5, 41))
    # Satélite: todas las curvas comparten los 72 tiempos -> una sola Φ y una sola S
    res_s = []
    for K in K_grid:
        Phi = base_bspline(T_PENT, K)
        C, L, h = mco_mano(Phi, Xs.T)
        R = Xs.T - Phi @ C
        n = 72
        sse = (R ** 2).sum(0)
        res_s.append({"K": K, "ECM": np.mean(sse / n),
                      "VC": np.mean(((R / (1 - h)[:, None]) ** 2).mean(0)),
                      "VCG": np.mean((sse / n) / ((n - K) / n) ** 2)})
    res_s = pd.DataFrame(res_s)
    # Estaciones: cada curva tiene sus propios días observados -> Φ_i distinta por curva
    rng = np.random.default_rng(SEMILLA)
    res_e = []
    Phi_full = {K: base_bspline(T_FINO, K) for K in K_grid}
    for K in K_grid:
        acum = np.zeros(3); no_ident = 0
        for i in range(Yd.shape[0]):
            obs = np.isfinite(Yd[i])
            Phi = Phi_full[K][obs]
            y = Yd[i, obs]
            try:
                c, L, h = mco_mano(Phi, y)
            except np.linalg.LinAlgError:
                no_ident += 1          # un hueco cubre un intervalo entre nodos: Φ'Φ singular
                continue
            acum += criterios(y, Phi @ c, h, K)
        if no_ident:                   # el criterio no es comparable si faltan curvas
            res_e.append({"K": K, "ECM": np.nan, "VC": np.nan, "VCG": np.nan, "curvas_no_identificables": no_ident})
        else:
            res_e.append({"K": K, "ECM": acum[0] / Yd.shape[0], "VC": acum[1] / Yd.shape[0],
                          "VCG": acum[2] / Yd.shape[0], "curvas_no_identificables": 0})
        log("K", K, "VCG est", round(res_e[-1]["VCG"], 4), "VCG sat", round(res_s.VCG[res_s.K == K].item(), 4))
    res_e = pd.DataFrame(res_e)
    K_est = int(res_e.K[res_e.VCG.idxmin()]); K_sat = int(res_s.K[res_s.VCG.idxmin()])
    K_est_vc = int(res_e.K[res_e.VC.idxmin()]); K_sat_vc = int(res_s.K[res_s.VC.idxmin()])
    tabK = pd.concat([res_e.assign(fuente="estaciones"), res_s.assign(fuente="satelite")])
    guarda_tab(tabK, "Tab08_seleccion_K.csv")
    elec = pd.DataFrame([{"fuente": "estaciones (diaria, n<=365)", "K_VCG": K_est, "K_VC": K_est_vc,
                          "VCG_min": res_e.VCG.min(), "curvas": Yd.shape[0]},
                         {"fuente": "satelite (pentadal, n=72)", "K_VCG": K_sat, "K_VC": K_sat_vc,
                          "VCG_min": res_s.VCG.min(), "curvas": Xs.shape[0]}])
    guarda_tab(elec, "Tab09_K_elegido.csv")
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.4))
    for f, (rr, Kopt, col, nom) in enumerate([(res_e, K_est, C_EST, "Estaciones"), (res_s, K_sat, C_SAT, "Satélite")]):
        for j, crit in enumerate(["ECM", "VC", "VCG"]):
            ax[f, j].plot(rr.K, rr[crit], "o-", ms=3, color=col)
            ax[f, j].axvline(Kopt, ls="--", color="k", lw=0.8)
            ax[f, j].set_title(f"{nom}: {crit}" + (f" (mín. en K={int(rr.K[rr[crit].idxmin()])})" if crit != "ECM" else ""))
            ax[f, j].set_xlabel("K (número de funciones base)")
    ax[0, 0].set_ylabel("(mm/día)²"); ax[1, 0].set_ylabel("(mm/día)²")
    fig.tight_layout()
    guarda_fig(fig, "Fig04_seleccion_K_ECM_VC_VCG.png")

    # ---- Fig. 3: base B-spline elegida
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.8))
    for a_, K, nom in ((ax[0], K_est, "estaciones"), (ax[1], K_sat, "satélite")):
        a_.plot(T_FINO, base_bspline(T_FINO, K), lw=1)
        for nd in np.unique(nodos_bspline(K)):
            a_.axvline(nd, color="grey", lw=0.4, ls=":")
        a_.set_title(f"Base B-spline cúbica, K={K} ({nom})"); eje_meses(a_)
    guarda_fig(fig, "Fig03_base_bspline.png")

    # ---- Ajuste final con el K elegido
    Phi_e = Phi_full[K_est]
    C_e = np.zeros((Yd.shape[0], K_est)); res_e_m = np.full_like(Yd, np.nan)
    verS = []
    for i in range(Yd.shape[0]):
        obs = np.isfinite(Yd[i])
        c, L, h = mco_mano(Phi_e[obs], Yd[i, obs])
        C_e[i] = c
        res_e_m[i, obs] = Yd[i, obs] - Phi_e[obs] @ c
    Phi_s = base_bspline(T_PENT, K_sat)
    C_s, L_s, h_s = mco_mano(Phi_s, Xs.T); C_s = C_s.T
    res_s_m = Xs - C_s @ Phi_s.T

    # ---- Verificación numérica de la matriz de suavizado (Tab. 10)
    def verifica_S(Phi, nombre):
        A = Phi.T @ Phi
        L = cholesky_mano(A)
        Ainv = atras(L, adelante(L, np.eye(A.shape[0])))
        S = Phi @ Ainv @ Phi.T
        return {"caso": nombre, "n": Phi.shape[0], "K": Phi.shape[1], "traza(S)": np.trace(S),
                "max|SS - S|": np.abs(S @ S - S).max(), "max|S - S'|": np.abs(S - S.T).max(),
                "max|S·1 - 1|": np.abs(S @ np.ones(S.shape[0]) - 1).max(),
                "cond(Φ'Φ)": np.linalg.cond(A)}
    verS.append(verifica_S(Phi_s, "satélite (72 péntadas)"))
    verS.append(verifica_S(Phi_e, "estación sin faltantes (365 días)"))
    i_peor = int(np.argmin(np.isfinite(Yd).sum(1)))
    verS.append(verifica_S(Phi_e[np.isfinite(Yd[i_peor])], f"estación con más faltantes ({int(np.isfinite(Yd[i_peor]).sum())} días)"))
    guarda_tab(pd.DataFrame(verS), "Tab10_verificacion_S.csv")

    # ---- Conservación del total anual: como Σ_k φ_k = 1, el residuo es ortogonal a la constante y
    #      Σ_j ŷ_j = Σ_j y_j en los días observados. Se reporta además ∫x̂ vs total observado.
    tot_obs_e = np.nansum(Yd, 1) * DIAS / np.isfinite(Yd).sum(1)
    tot_fun_e = (C_e @ Phi_e.T).sum(1)            # integral por regla del punto medio diaria
    tot_obs_s = (Xs * np.where(True, PENT_LON, 1)).sum(1)
    tot_fun_s = (C_s @ base_bspline(T_FINO, K_sat).T).sum(1)
    conserva = pd.DataFrame([
        {"fuente": "estaciones", "total_obs_medio_mm": tot_obs_e.mean(), "integral_curva_media_mm": tot_fun_e.mean(),
         "error_rel_medio_%": 100 * np.mean((tot_fun_e - tot_obs_e) / tot_obs_e),
         "max_|Σŷ-Σy|_dias_obs": max(abs((Phi_e[np.isfinite(Yd[i])] @ C_e[i]).sum() - np.nansum(Yd[i])) for i in range(0, Yd.shape[0], 50))},
        {"fuente": "satelite", "total_obs_medio_mm": tot_obs_s.mean(), "integral_curva_media_mm": tot_fun_s.mean(),
         "error_rel_medio_%": 100 * np.mean((tot_fun_s - tot_obs_s) / tot_obs_s),
         "max_|Σŷ-Σy|_dias_obs": np.abs((C_s @ Phi_s.T).sum(1) - Xs.sum(1)).max()}])
    conserva["%_malla_con_valor_negativo"] = [100 * np.mean(C_e @ Phi_e.T < 0),
                                              100 * np.mean(C_s @ base_bspline(T_FINO, K_sat).T < 0)]
    guarda_tab(conserva, "Tab11_conservacion_total_anual.csv")

    # ---- Covarianza residual (Fig. 6) y bandas de confianza punto a punto (Fig. 5)
    # Var(ĉ_i) = (Φ'Φ)^{-1} Φ' Σ_e Φ (Φ'Φ)^{-1};  Var(x̂_i(t)) = φ(t)' Var(ĉ_i) φ(t).
    # Σ_e se estima con los residuos: la matriz de correlación residual (todas las curvas) muestra
    # que la dependencia entre tiempos distintos es débil, así que se usa Σ_e diagonal con varianza
    # que cambia en el año y entre sitios (heterocedástica): σ²_s(t) = media de r² de ese sitio
    # (estación o píxel) en ese tiempo, suavizada con ventana móvil (±15 días / ±1 péntada).
    def cor_residual(R):
        Mk = np.isfinite(R).astype(float); R0 = np.nan_to_num(R)
        cov = (R0.T @ R0) / np.maximum(Mk.T @ Mk, 1)
        sd = np.sqrt(np.diag(cov))
        return cov, cov / np.outer(sd, sd)
    cov_re, cor_re = cor_residual(res_e_m)
    cov_rs, cor_rs = cor_residual(res_s_m)
    lag1_e = np.mean(np.diag(cor_re, 1)); lag1_s = np.mean(np.diag(cor_rs, 1))
    fuera_e = np.abs(cor_re[np.triu_indices(DIAS, 2)]).mean(); fuera_s = np.abs(cor_rs[np.triu_indices(72, 2)]).mean()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    im = ax[0].imshow(cor_re, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower", extent=[0, DIAS, 0, DIAS])
    ax[0].set_title(f"Correlación residual: estaciones\nr(1 día)={lag1_e:.2f}; |r| medio (≥2 días)={fuera_e:.3f}", fontsize=8)
    ax[1].imshow(cor_rs, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower", extent=[0, DIAS, 0, DIAS])
    ax[1].set_title(f"Correlación residual: satélite\nr(1 péntada)={lag1_s:.2f}; |r| medio (≥2 pént.)={fuera_s:.3f}", fontsize=8)
    plt.colorbar(im, ax=ax[:2], shrink=0.8)
    ax[2].plot(T_FINO, np.sqrt(np.diag(cov_re)), color=C_EST, label="estaciones (diario)")
    ax[2].plot(T_PENT, np.sqrt(np.diag(cov_rs)), color=C_SAT, label="satélite (péntada)")
    ax[2].set_title("Desviación estándar residual σ_e(t)"); ax[2].set_ylabel("mm/día"); ax[2].legend()
    for a_ in ax:
        eje_meses(a_)
    ax[0].set_yticks([]); ax[1].set_yticks([])
    guarda_fig(fig, "Fig06_covarianza_residual.png")

    def suaviza_circular(v, w):
        k = np.ones(2 * w + 1) / (2 * w + 1)
        return np.convolve(np.concatenate([v[-w:], v, v[:w]]), k, mode="valid")

    def banda(Phi_obs, Phi_eval, sig2_obs):
        A = Phi_obs.T @ Phi_obs
        L = cholesky_mano(A)
        Ainv = atras(L, adelante(L, np.eye(A.shape[0])))
        H = Ainv @ Phi_obs.T                       # K x n
        Vc = (H * sig2_obs) @ H.T                  # (Φ'Φ)^{-1}Φ' Σ_e Φ(Φ'Φ)^{-1}, Σ_e diagonal
        return np.sqrt(np.einsum("tk,kl,tl->t", Phi_eval, Vc, Phi_eval)), Vc

    # σ²(t) por estación y por píxel
    df_e = pd.DataFrame({"station": est})
    sig2_est = {}
    for e_, idx in df_e.groupby("station").indices.items():
        v = np.nanmean(res_e_m[idx] ** 2, 0)
        sig2_est[e_] = suaviza_circular(np.nan_to_num(v, nan=np.nanmean(v)), 15)
    pix = meta_s.pixel.values
    sig2_pix = np.zeros((pix.max() + 1, 72))
    for p_ in range(pix.max() + 1):
        sig2_pix[p_] = suaviza_circular((res_s_m[pix == p_] ** 2).mean(0), 1)
    # ancho medio de la banda en todas las curvas (Tab. 12)
    Phi_fs = base_bspline(T_FINO, K_sat)
    se_e = np.zeros(Yd.shape[0]); se_s_med = []
    for i in range(Yd.shape[0]):
        obs = np.isfinite(Yd[i])
        se, _ = banda(Phi_e[obs], Phi_e, sig2_est[est[i]][obs])
        se_e[i] = se.mean()
    for p_ in range(0, pix.max() + 1):
        se, _ = banda(Phi_s, Phi_fs, sig2_pix[p_])
        se_s_med.append(se.mean())
    guarda_tab(pd.DataFrame([{"fuente": "estaciones", "EE_medio_mm_dia": se_e.mean(), "EE_p90": np.quantile(se_e, .9),
                              "nivel_medio_curvas_mm_dia": (C_e @ Phi_e.T).mean()},
                             {"fuente": "satelite", "EE_medio_mm_dia": np.mean(se_s_med), "EE_p90": np.quantile(se_s_med, .9),
                              "nivel_medio_curvas_mm_dia": (C_s @ Phi_fs.T).mean()}]), "Tab12_bandas_EE.csv")
    # Fig. 5: ejemplo de una estación-año y del píxel que la contiene
    cvc = pd.read_csv(os.path.join(DATOS, "CVC_Wheather_Stations_Long_Lat.csv"), encoding="latin-1")
    cand = [(i, e_) for i, e_ in enumerate(est) if e_ in set(cvc.CODIGO) and anio_e[i] == 1999]
    i_ej = cand[0][0] if cand else 0
    fila_c = cvc[cvc.CODIGO == est[i_ej]].iloc[0] if cand else None
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    obs = np.isfinite(Yd[i_ej])
    se, _ = banda(Phi_e[obs], Phi_e, sig2_est[est[i_ej]][obs])
    yh = Phi_e @ C_e[i_ej]
    ax[0].bar(T_FINO, np.nan_to_num(Yd[i_ej]), width=1, color="#aaaaaa", label="observado (mm/día)")
    ax[0].plot(T_FINO, yh, color=C_EST, lw=2, label=f"x̂(t), K={K_est}")
    ax[0].fill_between(T_FINO, yh - 2 * se, yh + 2 * se, color=C_EST, alpha=0.25, label="±2 EE punto a punto")
    ax[0].set_ylim(0, max(10, np.nanpercentile(Yd[i_ej], 99)))
    ax[0].set_title(f"Estación {est[i_ej]} ({fila_c.STATION_NAME if fila_c is not None else ''}), {anio_e[i_ej]}")
    ax[0].legend(fontsize=7); eje_meses(ax[0])
    if fila_c is not None:
        j_ = int(np.floor((fila_c.Lon - (-77.55)) / 0.05)); i_ = int(np.floor((5.0 - fila_c.lat) / 0.05))
        p_ = i_ * 37 + j_
        r_ = np.where((meta_s.pixel.values == p_) & (meta_s.year.values == anio_e[i_ej]))[0][0]
        se, _ = banda(Phi_s, Phi_fs, sig2_pix[p_])
        xh = Phi_fs @ C_s[r_]
        ax[1].bar(T_PENT, Xs[r_], width=PENT_LON, color="#e8c9a8", label="observado (mm/día por péntada)")
        ax[1].plot(T_FINO, xh, color=C_SAT, lw=2, label=f"x̂(t), K={K_sat}")
        ax[1].fill_between(T_FINO, xh - 2 * se, xh + 2 * se, color=C_SAT, alpha=0.3, label="±2 EE punto a punto")
        ax[1].set_title(f"Píxel que contiene la estación ({fila_c.Lon:.2f}, {fila_c.lat:.2f}), {anio_e[i_ej]}")
        ax[1].legend(fontsize=7); eje_meses(ax[1])
    guarda_fig(fig, "Fig05_ajuste_y_bandas.png")

    # ---- Fig. 7: curvas de ambas fuentes en el mismo dominio anual [0, 365]
    Ye = C_e @ Phi_e.T; Xf = C_s @ Phi_fs.T
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    rs = np.random.default_rng(SEMILLA)
    for a_, M_, col, nom in ((ax[0], Ye, C_EST, f"Estaciones: {Ye.shape[0]} curvas estación-año (1981-2003), K={K_est}"),
                             (ax[1], Xf, C_SAT, f"Satélite: {Xf.shape[0]} curvas píxel-año (1981-2012), K={K_sat}")):
        sub = rs.choice(M_.shape[0], size=min(600, M_.shape[0]), replace=False)
        a_.plot(T_FINO, M_[sub].T, color="grey", lw=0.3, alpha=0.4)
        a_.plot(T_FINO, M_.mean(0), color=col, lw=2.5, label="media")
        a_.set_title(nom, fontsize=8); eje_meses(a_); a_.legend()
    ax[0].set_ylabel("Precipitación (mm/día)"); ax[0].set_ylim(0, np.percentile(np.r_[Ye.ravel(), Xf.ravel()], 99.7))
    guarda_fig(fig, "Fig07_curvas_dominio_comun.png")

    np.savez_compressed(os.path.join(CACHE, "fase1_coef_estaciones.npz"), C=C_e, station=est, year=anio_e,
                        K=K_est, nodos=nodos_bspline(K_est), orden=ORDEN)
    np.savez_compressed(os.path.join(CACHE, "fase1_coef_satelite.npz"), C=C_s, pixel=meta_s.pixel.values,
                        fila=meta_s.fila.values, col=meta_s.col.values, lon=meta_s.lon.values,
                        lat=meta_s.lat.values, year=meta_s.year.values, K=K_sat, nodos=nodos_bspline(K_sat), orden=ORDEN)
    # Para comparar fuentes (fase 3) ambas deben tener el MISMO suavizado: el satélite se reajusta
    # también con K = K_est (misma base que las estaciones) y las estaciones con K = K_sat.
    Phi_s_c = base_bspline(T_PENT, K_est)
    C_s_c, _, _ = mco_mano(Phi_s_c, Xs.T)
    np.savez_compressed(os.path.join(CACHE, "fase1_coef_satelite_Kcomun.npz"), C=C_s_c.T, K=K_est)
    C_e_alt = np.zeros((Yd.shape[0], K_sat)); P_alt = base_bspline(T_FINO, K_sat)
    for i in range(Yd.shape[0]):
        obs = np.isfinite(Yd[i])
        C_e_alt[i] = mco_mano(P_alt[obs], Yd[i, obs])[0]
    np.savez_compressed(os.path.join(CACHE, "fase1_coef_estaciones_Ksat.npz"), C=C_e_alt, K=K_sat)
    json.dump({"K_est": K_est, "K_sat": K_sat, "lag1_res_est": lag1_e, "lag1_res_sat": lag1_s},
              open(os.path.join(CACHE, "fase1_resumen.json"), "w"), indent=1)
    return elec



# =====================================================================================
# FASE 2. Análisis exploratorio de datos funcionales (se permiten librerías: scikit-fda)
# =====================================================================================

def carga_coef():
    ze = np.load(os.path.join(CACHE, "fase1_coef_estaciones.npz"))
    zs = np.load(os.path.join(CACHE, "fase1_coef_satelite.npz"))
    return ze, zs


def lee_oni():
    """ONI de NOAA-CPC (oni.ascii.txt). Fase ENSO del año civil: media de las 12 estaciones
    trimestrales (DJF..NDJ) >= +0.5 El Niño, <= -0.5 La Niña, resto Neutral."""
    o = pd.read_csv(os.path.join(EXT, "oni.ascii.txt"), sep=r"\s+")
    a = o.groupby("YR").ANOM.agg(["mean", "size"]).reset_index()
    a = a[a["size"] == 12]
    a["fase_enso"] = np.where(a["mean"] >= 0.5, "El Niño", np.where(a["mean"] <= -0.5, "La Niña", "Neutral"))
    return a.rename(columns={"YR": "year", "mean": "oni_anual"})[["year", "oni_anual", "fase_enso"]]


def catalogo_estaciones():
    """Une coordenadas (CVC) y altitud (catálogo de elevación) por código. Si la estación no trae
    altitud en el catálogo se usa la del SRTM 90 m (datos_externos/srtm90m_estaciones.csv, consultada
    en OpenTopoData con las coordenadas CVC); la validación SRTM vs catálogo está en Tab06d."""
    cvc = pd.read_csv(os.path.join(DATOS, "CVC_Wheather_Stations_Long_Lat.csv"), encoding="latin-1")
    el = pd.read_excel(os.path.join(DATOS, "CatalogoEstacionesElevacion.xlsx")).drop_duplicates("codigo")
    cat = cvc.rename(columns={"CODIGO": "station", "Lon": "lon"}).merge(
        el.rename(columns={"codigo": "station"})[["station", "alt", "cuenca"]], on="station", how="left")
    srtm = pd.read_csv(os.path.join(EXT, "srtm90m_estaciones.csv"))[["station", "alt_srtm90m"]].drop_duplicates("station")
    cat = cat.merge(srtm, on="station", how="left")
    cat["alt_catalogo"] = cat["alt"]
    cat["alt_fuente"] = np.where(cat.alt.notna(), "catálogo", np.where(cat.alt_srtm90m.notna(), "SRTM 90 m", "sin altitud"))
    cat["alt"] = cat.alt.fillna(cat.alt_srtm90m)
    return cat


def profundidad_fm(X):
    """Profundidad de Fraiman-Muniz como en fda.usc::depth.FM: integral (promedio en la malla) de
    1 - |1/2 - F_t(x_i(t))|, con F_t la función de distribución empírica en cada t."""
    N = X.shape[0]
    F = (np.argsort(np.argsort(X, axis=0), axis=0) + 1) / N        # rango/N
    return (1 - np.abs(0.5 - F)).mean(1)


def profundidad_mbd(X):
    """Profundidad de bandas modificada (J=2) [López-Pintado y Romo, 2009] por rangos:
    en cada t, la curva con rango r queda dentro de (r-1)(N-r) bandas de pares + (N-1) bandas que
    ella misma forma; se divide entre C(N,2) y se promedia en t."""
    N = X.shape[0]
    r = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    dentro = (r - 1) * (N - r) + (N - 1)
    return (dentro / (N * (N - 1) / 2)).mean(1)


def boxplot_funcional(X, prof, factor=1.5):
    """Boxplot funcional (Sun y Genton, 2011) como fda::fbplot: región central = envolvente del 50 %
    más profundo; bigotes = región central inflada 'factor' veces su rango; atípica = curva que sale
    de los bigotes en algún t."""
    N = X.shape[0]
    orden = np.argsort(-prof)
    centro = X[orden[:int(np.ceil(N / 2))]]
    inf_, sup_ = centro.min(0), centro.max(0)
    rango = sup_ - inf_
    bi, bs = inf_ - factor * rango, sup_ + factor * rango
    atip = np.where(((X < bi) | (X > bs)).any(1))[0]
    return {"mediana": X[orden[0]], "inf": inf_, "sup": sup_, "bigote_inf": bi, "bigote_sup": bs, "atipicas": atip}


def gram(K, orden=ORDEN, deriv=0, n_cuad=20):
    """W = integral de phi(t) phi(t)^T dt por cuadratura de Gauss-Legendre en cada intervalo entre
    nodos (exacta para polinomios a trozos). deriv=2 da la matriz de rugosidad R (segundas derivadas)."""
    nod = np.unique(nodos_bspline(K, orden=orden))
    x, w = np.polynomial.legendre.leggauss(n_cuad)
    tq, wq = [], []
    for a, b in zip(nod[:-1], nod[1:]):
        tq.append((b - a) / 2 * x + (a + b) / 2); wq.append((b - a) / 2 * w)
    tq, wq = np.concatenate(tq), np.concatenate(wq)
    B = base_bspline(tq, K, orden=orden, deriv=deriv)
    return (B * wq[:, None]).T @ B


def fpca_coef(C, W):
    """FPCA sobre coeficientes (Ramsay y Silverman, cap. 8): problema propio de W^1/2 Sigma_c W^1/2.
    Devuelve media de coeficientes, coeficientes b_m de las funciones propias psi_m (normalizadas en
    L2: b_m^T W b_m = 1), valores propios, varianza explicada y puntajes s_im = int (x_i - xbar) psi_m."""
    cbar = C.mean(0)
    Cc = C - cbar
    Sigma = Cc.T @ Cc / (C.shape[0] - 1)
    ev, U = np.linalg.eigh(W)
    W12 = U @ np.diag(np.sqrt(ev)) @ U.T
    W12i = U @ np.diag(1 / np.sqrt(ev)) @ U.T
    lam, V = np.linalg.eigh(W12 @ Sigma @ W12)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    B = W12i @ V
    scores = Cc @ W @ B
    return {"cbar": cbar, "coef_psi": B, "valores_propios": lam, "var_exp": lam / lam.sum(), "puntajes": scores}


def fase2():
    log("FASE 2: AED funcional")
    ze, zs = carga_coef()
    K_e, K_s = int(ze["K"]), int(zs["K"])
    C_e, C_s = ze["C"], zs["C"]
    est, anio_e = ze["station"], ze["year"]
    Phi_e = base_bspline(T_FINO, K_e); Phi_s = base_bspline(T_FINO, K_s)
    D1_e = base_bspline(T_FINO, K_e, deriv=1); D1_s = base_bspline(T_FINO, K_s, deriv=1)
    oni = lee_oni(); cat = catalogo_estaciones()
    salidas = {}

    # ---- Tab. 13: media funcional paso a paso x̄(t) = c̄'φ(t) y verificación con scikit-fda
    from skfda.representation.basis import BSplineBasis
    from skfda.representation import FDataBasis
    ver = []
    medias, sds = {}, {}
    for nom, C, Phi, K in (("estaciones", C_e, Phi_e, K_e), ("satelite", C_s, Phi_s, K_s)):
        cbar = np.zeros(K)
        for i in range(C.shape[0]):           # acumulador explícito
            cbar += C[i]
        cbar /= C.shape[0]
        media_paso = Phi @ cbar
        fd = FDataBasis(BSplineBasis(domain_range=(0, DIAS), n_basis=K, order=ORDEN), C)
        media_lib = fd.mean()(T_FINO)[0, :, 0]
        # varianza funcional por coeficientes: Var(t) = φ(t)' Σ_c φ(t); se verifica contra la varianza
        # punto a punto (ddof=1) de las curvas evaluadas en la malla
        Cc = C - cbar
        Sigma_c = Cc.T @ Cc / (C.shape[0] - 1)
        var_t = np.einsum("tk,kl,tl->t", Phi, Sigma_c, Phi)
        var_malla = np.var(C @ Phi.T, axis=0, ddof=1)
        ver.append({"fuente": nom, "N_curvas": C.shape[0], "K": K,
                    "max|media_paso - media_skfda|": np.abs(media_paso - media_lib).max(),
                    "max|var_coef - var_malla|": np.abs(var_t - var_malla).max()})
        medias[nom], sds[nom] = media_paso, np.sqrt(var_t)
        salidas[nom] = {"cbar": cbar, "Sigma_c": Sigma_c}
    guarda_tab(pd.DataFrame(ver), "Tab13_verificacion_media_varianza.csv")

    # ---- Fig. 8: media, DE y CV funcional
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for nom, col, et in (("estaciones", C_EST, "Estaciones"), ("satelite", C_SAT, "Satélite")):
        ax[0].plot(T_FINO, medias[nom], color=col, lw=2, label=et)
        ax[0].fill_between(T_FINO, np.maximum(medias[nom] - 2 * sds[nom], 0), medias[nom] + 2 * sds[nom], color=col, alpha=0.12)
        ax[1].plot(T_FINO, sds[nom], color=col, lw=2, label=et)
        ax[2].plot(T_FINO, sds[nom] / medias[nom], color=col, lw=2, label=et)
    ax[0].set_title("Media funcional (sombra: media ± 2 DE)"); ax[0].set_ylabel("mm/día")
    ax[1].set_title("Desviación estándar funcional"); ax[2].set_title("Coeficiente de variación DE(t)/x̄(t)")
    for a_ in ax:
        eje_meses(a_); a_.legend(fontsize=7)
    guarda_fig(fig, "Fig08_media_DE_CV.png")
    resumen = []
    for nom in ("estaciones", "satelite"):
        m_ = medias[nom]
        resumen.append({"fuente": nom, "total_anual_medio_mm": m_.sum(), "media_min_mm_dia": m_.min(),
                        "fecha_min": fecha_txt(T_FINO[m_.argmin()]), "media_max_mm_dia": m_.max(),
                        "fecha_max": fecha_txt(T_FINO[m_.argmax()]), "DE_max": sds[nom].max(),
                        "fecha_DE_max": fecha_txt(T_FINO[sds[nom].argmax()]), "CV_medio": np.mean(sds[nom] / m_)})
    guarda_tab(pd.DataFrame(resumen), "Tab14_resumen_media_DE.csv")

    # ---- Fig. 9: covarianza y correlación funcional por fuente: v(s,t) = φ(s)' Σ_c φ(t)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    cors = {}
    for j, (nom, Phi) in enumerate((("estaciones", Phi_e), ("satelite", Phi_s))):
        V = Phi @ salidas[nom]["Sigma_c"] @ Phi.T
        sd = np.sqrt(np.diag(V)); R = V / np.outer(sd, sd); cors[nom] = R
        cs = ax[0, j].contourf(T_FINO, T_FINO, V, levels=15, cmap="viridis")
        plt.colorbar(cs, ax=ax[0, j], shrink=0.8)
        ax[0, j].set_title(f"Covarianza v(s,t) — {nom} ((mm/día)²)")
        cs = ax[1, j].contourf(T_FINO, T_FINO, R, levels=np.linspace(-1, 1, 21), cmap="RdBu_r")
        ax[1, j].contour(T_FINO, T_FINO, R, levels=[0, 0.5], colors="k", linewidths=0.5)
        plt.colorbar(cs, ax=ax[1, j], shrink=0.8)
        ax[1, j].set_title(f"Correlación r(s,t) — {nom}")
        for a_ in (ax[0, j], ax[1, j]):
            eje_meses(a_); a_.set_yticks(INICIO_MES + DIAS_MES / 2); a_.set_yticklabels(list("EFMAMJJASOND")); a_.set_ylim(0, DIAS)
    fig.tight_layout()
    guarda_fig(fig, "Fig09_covarianza_correlacion.png")
    guarda_tab(pd.DataFrame([{"fuente": n, "r(15-abr, 28-oct)": cors[n][104, 300], "r(15-abr, 15-jul)": cors[n][104, 195],
                              "r(15-ene, 15-jul)": cors[n][14, 195], "r(15-abr, 15-may)": cors[n][104, 134]} for n in cors]),
               "Tab15_correlaciones_clave.csv")

    # ---- Fig. 10: derivadas. D x̄(t) = c̄' Dφ(t). Picos/valles = ceros de D x̄; inicio de temporada =
    #      máximo de D x̄ antes de cada pico (subida más rápida); fin = mínimo de D x̄ después del pico.
    filas_t = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for nom, D1, col in (("estaciones", D1_e, C_EST), ("satelite", D1_s, C_SAT)):
        dm = D1 @ salidas[nom]["cbar"]
        m_ = medias[nom]
        ax[0].plot(T_FINO, m_, color=col, lw=2, label=nom)
        ax[1].plot(T_FINO, dm, color=col, lw=2, label=nom)
        ceros = np.where(np.diff(np.sign(dm)) != 0)[0]
        for c0 in ceros:
            tipo = "pico (máximo)" if dm[c0] > 0 else "valle (mínimo)"
            filas_t.append({"fuente": nom, "evento": tipo, "fecha": fecha_txt(T_FINO[c0]), "dia": int(T_FINO[c0]),
                            "valor_mm_dia": m_[c0]})
            ax[0].axvline(T_FINO[c0], color=col, ls=":", lw=0.8)
        picos = [c0 for c0 in ceros if dm[c0] > 0 and 40 < c0 < 345]
        for pk in picos:
            ant = [c for c in ceros if c < pk]; sig = [c for c in ceros if c > pk]
            a0 = ant[-1] if ant else 0; b0 = sig[0] if sig else DIAS - 1
            ini = a0 + int(np.argmax(dm[a0:pk + 1])); fin = pk + int(np.argmin(dm[pk:b0 + 1]))
            filas_t.append({"fuente": nom, "evento": "inicio temporada (máx. D)", "fecha": fecha_txt(T_FINO[ini]), "dia": ini, "valor_mm_dia": m_[ini]})
            filas_t.append({"fuente": nom, "evento": "fin temporada (mín. D)", "fecha": fecha_txt(T_FINO[fin]), "dia": fin, "valor_mm_dia": m_[fin]})
            ax[1].plot([T_FINO[ini], T_FINO[fin]], [dm[ini], dm[fin]], "v", color=col)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[0].set_title("Media funcional y sus puntos críticos (D x̄ = 0)"); ax[0].set_ylabel("mm/día")
    ax[1].set_title("Primera derivada de la media D x̄(t) (triángulos: inicio/fin)"); ax[1].set_ylabel("mm/día por día")
    for a_ in ax:
        eje_meses(a_); a_.legend(fontsize=7)
    guarda_fig(fig, "Fig10_derivadas_temporadas.png")
    guarda_tab(pd.DataFrame(filas_t).sort_values(["fuente", "dia"]), "Tab16_temporadas_derivada.csv")
    dmed = (C_e @ D1_e.T).mean(0)
    log("verificación: max|D(media) - media(D)| estaciones =", np.abs(dmed - D1_e @ salidas["estaciones"]["cbar"]).max())

    # ---- Profundidad FM y MBD, boxplot funcional (malla cada 5 días: 73 puntos)
    t5 = np.arange(2.5, DIAS, 5.0)
    P5e = base_bspline(t5, K_e); P5s = base_bspline(t5, K_s)
    Ye5 = C_e @ P5e.T
    # (a) curvas en bruto (nivel + forma): atípicas espaciales
    fm_e, mbd_e = profundidad_fm(Ye5), profundidad_mbd(Ye5)
    bx_e = boxplot_funcional(Ye5, mbd_e)
    # (b) anomalías respecto a la media de cada estación: atípicas interanuales (ENSO)
    dfe = pd.DataFrame({"station": est, "year": anio_e})
    An = Ye5.copy()
    for e_, idx in dfe.groupby("station").indices.items():
        An[idx] -= Ye5[idx].mean(0)
    mbd_a = profundidad_mbd(An); fm_a = profundidad_fm(An)
    bx_a = boxplot_funcional(An, mbd_a)
    from skfda.representation import FDataGrid
    from skfda.exploratory.depth import ModifiedBandDepth
    import itertools
    sub = np.arange(0, len(Ye5), 32)                     # submuestra para la verificación O(N²)
    Xv = Ye5[sub]; Nv = len(sub)
    mbd_fb = np.zeros(Nv)                                # definición literal: promedio sobre pares de bandas
    for a, b in itertools.combinations(range(Nv), 2):
        lo = np.minimum(Xv[a], Xv[b]); hi = np.maximum(Xv[a], Xv[b])
        mbd_fb += ((Xv >= lo) & (Xv <= hi)).mean(1)
    mbd_fb /= Nv * (Nv - 1) / 2
    mbd_lib = ModifiedBandDepth()(FDataGrid(Xv, t5))
    guarda_tab(pd.DataFrame([{"curvas_submuestra": Nv,
                              "max|MBD_rangos - MBD_fuerza_bruta|": np.abs(profundidad_mbd(Xv) - mbd_fb).max(),
                              "Spearman(MBD_mano, MBD_skfda)": pd.Series(profundidad_mbd(Xv)).corr(pd.Series(mbd_lib), method="spearman"),
                              "Spearman(FM, MBD) todas las curvas": pd.Series(fm_e).corr(pd.Series(mbd_e), method="spearman")}]),
               "Tab17_verificacion_profundidad.csv")
    prof = dfe.assign(FM=fm_e, MBD=mbd_e, FM_anom=fm_a, MBD_anom=mbd_a).merge(oni, on="year", how="left")
    prof["atipica_bruta"] = False; prof.loc[bx_e["atipicas"], "atipica_bruta"] = True
    prof["atipica_anomalia"] = False; prof.loc[bx_a["atipicas"], "atipica_anomalia"] = True
    prof["total_anual_mm"] = (C_e @ Phi_e.T).sum(1)
    prof = prof.merge(cat[["station", "STATION_NAME", "alt", "lon", "lat"]], on="station", how="left")
    prof.to_csv(os.path.join(CACHE, "fase2_profundidades_estaciones.csv"), index=False)
    at = prof[prof.atipica_bruta | prof.atipica_anomalia]
    guarda_tab(at.sort_values(["atipica_bruta", "MBD"], ascending=[False, True]), "Tab18_atipicas_estaciones.csv")
    orden = np.argsort(-fm_e)
    med_fm = Phi_e @ C_e[orden[0]]
    trim = Phi_e @ C_e[orden[: int(0.75 * len(orden))]].mean(0)
    resA = (prof.groupby("fase_enso").agg(curvas=("year", "size"), atipicas_anomalia=("atipica_anomalia", "sum"))
            .assign(porcentaje=lambda d: 100 * d.atipicas_anomalia / d.curvas).reset_index())
    guarda_tab(resA, "Tab19_atipicas_por_fase_enso.csv")

    # Satélite: profundidad en todas las curvas píxel-año (por rangos)
    Xs5 = C_s @ P5s.T
    mbd_s = profundidad_mbd(Xs5)
    bx_s = boxplot_funcional(Xs5, mbd_s)
    pix = pd.DataFrame({"pixel": zs["pixel"], "lon": zs["lon"], "lat": zs["lat"], "year": zs["year"]})
    Asat = Xs5.copy()
    for p_, idx in pix.groupby("pixel").indices.items():
        Asat[idx] -= Xs5[idx].mean(0)
    mbd_sa = profundidad_mbd(Asat); bx_sa = boxplot_funcional(Asat, mbd_sa)
    pix["atipica_bruta"] = False; pix.loc[bx_s["atipicas"], "atipica_bruta"] = True
    pix["atipica_anomalia"] = False; pix.loc[bx_sa["atipicas"], "atipica_anomalia"] = True
    pix = pix.merge(oni, on="year", how="left")
    sat_at = (pix.groupby("year").agg(atipicas_anomalia=("atipica_anomalia", "sum"), atipicas_bruto=("atipica_bruta", "sum"),
                                      oni=("oni_anual", "first"), fase=("fase_enso", "first")).reset_index())
    guarda_tab(sat_at, "Tab20_atipicas_satelite_por_anio.csv")
    # Con N = 46.176 curvas la envolvente del 50 % central es tan ancha que ninguna curva sale de los
    # bigotes. Para leer años atípicos se aplica el boxplot a la curva REGIONAL de cada año (media de
    # los 1.443 píxeles; 32 curvas) y a la media regional de estaciones (23 curvas).
    anios_s = np.unique(zs["year"])
    Rs = np.array([Xs5[zs["year"] == a].mean(0) for a in anios_s])
    anios_e = np.unique(anio_e)
    Re = np.array([Ye5[anio_e == a].mean(0) for a in anios_e])
    bx_rs = boxplot_funcional(Rs, profundidad_mbd(Rs)); bx_re = boxplot_funcional(Re, profundidad_mbd(Re))
    reg = pd.concat([pd.DataFrame({"fuente": "satelite (media de la malla)", "year": anios_s, "MBD": profundidad_mbd(Rs),
                                   "FM": profundidad_fm(Rs), "atipica": np.isin(np.arange(len(anios_s)), bx_rs["atipicas"])}),
                     pd.DataFrame({"fuente": "estaciones (media de estaciones)", "year": anios_e, "MBD": profundidad_mbd(Re),
                                   "FM": profundidad_fm(Re), "atipica": np.isin(np.arange(len(anios_e)), bx_re["atipicas"])})]
                    ).merge(oni, on="year", how="left")
    guarda_tab(reg.sort_values(["fuente", "MBD"]), "Tab20b_profundidad_anios_regionales.csv")

    # ---- Fig. 11: profundidad y medidas de centralidad (estaciones)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    grises = plt.cm.Greys(np.linspace(0.15, 0.9, len(orden)))       # menos profunda = más clara
    for k in range(len(orden) - 1, -1, -6):
        ax[0].plot(T_FINO, Phi_e @ C_e[orden[k]], color=grises[len(orden) - 1 - k], lw=0.4)
    ax[0].set_title("Curvas de estaciones ordenadas por profundidad FM\n(oscuro = más profunda)"); ax[0].set_ylim(0, 30)
    ax[1].plot(T_FINO, medias["estaciones"], color="blue", lw=2, label="media")
    ax[1].plot(T_FINO, med_fm, color="red", lw=2, label=f"mediana FM ({est[orden[0]]}, {anio_e[orden[0]]})")
    ax[1].plot(T_FINO, trim, color="darkgreen", lw=2, ls="--", label="media recortada 25 % (FM)")
    ax[1].set_title("Medidas de centralidad"); ax[1].legend(fontsize=7)
    for a_ in ax:
        eje_meses(a_); a_.set_ylabel("mm/día")
    guarda_fig(fig, "Fig11_profundidad_centralidad.png")

    # ---- Fig. 12: boxplots funcionales (MBD)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax = ax.ravel()
    for a_, X, bx, tit in ((ax[0], Ye5, bx_e, f"(a) Estaciones, curva bruta: {len(bx_e['atipicas'])} atípicas de {len(Ye5)}"),
                           (ax[1], An, bx_a, f"(b) Estaciones, anomalía: {len(bx_a['atipicas'])} atípicas de {len(An)}"),
                           (ax[2], Asat, bx_sa, f"(c) Satélite, anomalía: {len(bx_sa['atipicas'])} atípicas de {len(Asat)}"),
                           (ax[3], Rs, bx_rs, f"(d) Satélite, media regional por año: {len(bx_rs['atipicas'])} atípicas de {len(Rs)}")):
        for i in bx["atipicas"][:400]:
            a_.plot(t5, X[i], color="red", lw=0.4, alpha=0.5)
        a_.fill_between(t5, bx["inf"], bx["sup"], color="#b39ddb", alpha=0.8, label="50 % central")
        a_.plot(t5, bx["bigote_inf"], "b-", lw=1); a_.plot(t5, bx["bigote_sup"], "b-", lw=1, label="bigotes (1,5×)")
        a_.plot(t5, bx["mediana"], "k-", lw=1.5, label="mediana MBD")
        a_.set_title(tit, fontsize=8); eje_meses(a_)
    for k, a in enumerate(anios_s):                   # etiqueta de años atípicos / extremos del panel (d)
        if k in bx_rs["atipicas"]:
            ax[3].text(t5[np.argmax(np.abs(Rs[k] - bx_rs["mediana"]))], Rs[k][np.argmax(np.abs(Rs[k] - bx_rs["mediana"]))], str(a), fontsize=7, color="red")
    ax[0].set_ylabel("mm/día"); ax[2].set_ylabel("mm/día"); ax[0].legend(fontsize=6)
    fig.tight_layout()
    guarda_fig(fig, "Fig12_boxplot_funcional.png")

    # ---- Fig. 13: orografía y ENSO en las curvas de estaciones
    tot = pd.DataFrame({"station": est, "year": anio_e, "total": (C_e @ Phi_e.T).sum(1)}).merge(oni, on="year")
    te = tot.groupby("station").total.mean().reset_index().merge(cat[["station", "alt", "lon"]], on="station", how="left")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    v = te.dropna(subset=["alt"])
    zona = np.where(v.station // 10**8 >= 50, "Pacífico (53/54)", "Cauca (26)")
    for z_, col in (("Cauca (26)", C_EST), ("Pacífico (53/54)", "#2e8b57")):
        ax[0].scatter(v.alt[zona == z_], v.total[zona == z_], s=10, color=col, label=z_)
    ax[0].set_xlabel("Altitud (m s.n.m.)"); ax[0].set_ylabel("Total anual medio (mm)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Orografía: total anual por estación")
    for f_, col in (("El Niño", "#d62728"), ("Neutral", "grey"), ("La Niña", "#1f77b4")):
        idx = np.isin(anio_e, oni.year[oni.fase_enso == f_])
        ax[1].plot(T_FINO, Phi_e @ C_e[idx].mean(0), color=col, lw=2, label=f"{f_} (n={idx.sum()})")
    ax[1].set_title("Media funcional de estaciones por fase ENSO"); eje_meses(ax[1]); ax[1].legend(fontsize=7)
    ax[1].set_ylabel("mm/día")
    an_tot = tot.assign(anom=lambda d: d.total / d.groupby("station").total.transform("mean") - 1)
    g = an_tot.groupby("year").agg(anom=("anom", "mean"), oni=("oni_anual", "first")).reset_index()
    ax[2].bar(g.year, 100 * g.anom, color=np.where(g.oni >= 0.5, "#d62728", np.where(g.oni <= -0.5, "#1f77b4", "grey")))
    ax[2].set_title("Anomalía media del total anual (%) por año\nrojo = El Niño, azul = La Niña"); ax[2].set_ylabel("%")
    guarda_fig(fig, "Fig13_orografia_enso_estaciones.png")
    guarda_tab(pd.DataFrame([{"Spearman(anomalía total anual, ONI) 1981-2003": pd.Series(g.anom).corr(g.oni, method="spearman"),
                              "años": len(g),
                              "Spearman(total medio, altitud) estaciones": v.total.corr(v.alt, method="spearman"),
                              "Spearman(total medio, altitud) solo Cauca": v.total[zona == "Cauca (26)"].corr(v.alt[zona == "Cauca (26)"], method="spearman"),
                              "estaciones_con_altitud": len(v)}]), "Tab21_enso_orografia.csv")
    g.to_csv(os.path.join(TAB, "Tab21b_anomalia_anual_oni.csv"), index=False)

    # ---- Tab. 22: representatividad de 1981-2003 (periodo con estaciones) dentro de 1981-2012 (satélite)
    Xf_m1 = Phi_s @ C_s[zs["year"] <= 2003].mean(0); Xf_m2 = Phi_s @ C_s[zs["year"] > 2003].mean(0)
    guarda_tab(pd.DataFrame([{"media_sat_1981_2003_total_mm": Xf_m1.sum(), "media_sat_2004_2012_total_mm": Xf_m2.sum(),
                              "dif_relativa_L2_%": 100 * np.sqrt(((Xf_m1 - Xf_m2) ** 2).sum() / (Xf_m1 ** 2).sum()),
                              "max_dif_mm_dia": np.abs(Xf_m1 - Xf_m2).max(),
                              "fecha_max_dif": fecha_txt(T_FINO[np.abs(Xf_m1 - Xf_m2).argmax()])}]),
               "Tab22_representatividad_periodo.csv")

    # ---- Fig. 14 / Tab. 23: FPCA de las curvas de estaciones (describe la variabilidad; insumo fase 4)
    W = gram(K_e)
    fp = fpca_coef(C_e, W)
    guarda_tab(pd.DataFrame({"componente": np.arange(1, len(fp["var_exp"]) + 1), "varianza_explicada": fp["var_exp"],
                             "acumulada": np.cumsum(fp["var_exp"])}), "Tab23_fpca_estaciones.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for m in range(3):
        ax[0].plot(T_FINO, Phi_e @ fp["coef_psi"][:, m], lw=2, label=f"ψ{m+1} ({100*fp['var_exp'][m]:.1f} %)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(fontsize=7); eje_meses(ax[0]); ax[0].set_title("Funciones propias (estaciones)")
    ax[1].plot(np.arange(1, K_e + 1), np.cumsum(fp["var_exp"]) * 100, "o-", color=C_EST)
    ax[1].set_xlabel("M componentes"); ax[1].set_ylabel("% varianza acumulada"); ax[1].set_title("Varianza explicada")
    guarda_fig(fig, "Fig14_fpca_estaciones.png")

    # ---- Tab. 23b / Fig. 14b: FPCA SEPARADO POR RÉGIMEN ENSO (asesoría 20-nov). El FPCA global mezcla
    #      Niño, Niña y neutro y sesga valores/vectores propios; aquí se estima uno por régimen (ONI del
    #      año, NOAA-CPC) y se compara con el global: varianza explicada y |coseno| entre funciones propias
    #      (producto interno L2 con W; 1 = misma forma). Solo años de entrenamiento 1981-1996.
    fase_c = pd.Series(anio_e).map(oni.set_index("year").fase_enso).values
    tr_ = anio_e <= ANIO_TRAIN_FIN
    filas_fp = []
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    fp_g = fpca_coef(C_e[tr_], W)
    for k_, (reg, col_) in enumerate((("El Niño", "#d62728"), ("Neutral", "grey"), ("La Niña", "#1f77b4"))):
        s_ = tr_ & (fase_c == reg)
        fp_r = fpca_coef(C_e[s_], W)
        for m in range(3):
            cos_ = abs(fp_r["coef_psi"][:, m] @ W @ fp_g["coef_psi"][:, m])
            filas_fp.append({"regimen": reg, "curvas_entrenamiento": int(s_.sum()), "anios": int(len(np.unique(anio_e[s_]))),
                             "componente": m + 1, "var_exp_%": 100 * fp_r["var_exp"][m],
                             "var_exp_acum_%": 100 * np.cumsum(fp_r["var_exp"])[m], "var_exp_global_%": 100 * fp_g["var_exp"][m],
                             "|coseno| con psi global": cos_, "valor_propio": fp_r["valores_propios"][m],
                             "valor_propio_global": fp_g["valores_propios"][m]})
            ax[m].plot(T_FINO, Phi_e @ fp_r["coef_psi"][:, m] * np.sign(fp_r["coef_psi"][:, m] @ W @ fp_g["coef_psi"][:, m]),
                       color=col_, lw=1.8, label=f"{reg} ({100*fp_r['var_exp'][m]:.0f} %)")
    for m in range(3):
        ax[m].plot(T_FINO, Phi_e @ fp_g["coef_psi"][:, m], "k--", lw=1, label="global (mezcla)")
        ax[m].axhline(0, color="k", lw=0.4); ax[m].set_title(f"ψ{m+1} por régimen ENSO (1981-1996)", fontsize=8)
        ax[m].legend(fontsize=6); eje_meses(ax[m])
    guarda_fig(fig, "Fig14b_fpca_por_regimen_enso.png")
    guarda_tab(pd.DataFrame(filas_fp), "Tab23b_fpca_por_regimen_enso.csv")
    return pd.DataFrame(resumen)



# =====================================================================================
# FASE 3. Comparación funcional estación vs satélite (se permiten librerías)
# =====================================================================================

LON0, LAT1, RES, NCOL, NFIL = -77.55, 5.00, 0.05, 37, 39       # malla del GeoTIFF (esquina NO)


def intervalos(indicador, t=T_FINO):
    """Intervalos contiguos del año donde 'indicador' es True (como Intervalos() del script t-funcional)."""
    ini, lon = rachas(np.asarray(indicador, bool))
    return [{"desde": fecha_txt(t[a]), "hasta": fecha_txt(t[a + l - 1]), "dias": int(l)} for a, l in zip(ini, lon)]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def fase3():
    log("FASE 3: comparación funcional")
    rng = np.random.default_rng(SEMILLA)
    ze, zs = carga_coef()
    K_e, K_s = int(ze["K"]), int(zs["K"])
    Phi_e = base_bspline(T_FINO, K_e); Phi_s = base_bspline(T_FINO, K_s)
    cat = catalogo_estaciones(); oni = lee_oni()

    # ---- Emparejamiento estación-píxel (Tab. 24, Fig. 15)
    # Regla: cada estación se asigna al píxel de 0,05° que CONTIENE su coordenada (CVC). No se usa el
    # vecino más correlacionado (eso usaría la respuesta para elegir el predictor).
    est_u = np.unique(ze["station"])
    m = cat[cat.station.isin(est_u)].copy()
    m = m[(m.lon > LON0) & (m.lon < LON0 + NCOL * RES) & (m.lat < LAT1) & (m.lat > LAT1 - NFIL * RES)]
    m["col"] = np.floor((m.lon - LON0) / RES).astype(int)
    m["fila"] = np.floor((LAT1 - m.lat) / RES).astype(int)
    m["pixel"] = m.fila * NCOL + m.col
    m["lon_pixel"] = LON0 + (m.col + 0.5) * RES
    m["lat_pixel"] = LAT1 - (m.fila + 0.5) * RES
    m["dist_km_centro"] = haversine_km(m.lon, m.lat, m.lon_pixel, m.lat_pixel)
    m["estaciones_en_pixel"] = m.groupby("pixel").station.transform("size")
    m["zona"] = np.where(m.station // 10**8 >= 50, "Pacífico (53/54)", "Cauca (26)")
    m["clase_altitud"] = pd.cut(m.alt, [-1, 1200, 2000, 5000], labels=["<1200 m", "1200-2000 m", ">2000 m"]).astype(str)
    m.loc[m.alt.isna(), "clase_altitud"] = "sin altitud"
    guarda_tab(m[["station", "STATION_NAME", "Municipality", "lon", "lat", "alt", "zona", "fila", "col", "pixel",
                  "lon_pixel", "lat_pixel", "dist_km_centro", "estaciones_en_pixel"]].sort_values(["pixel", "station"]),
               "Tab24_emparejamiento_estacion_pixel.csv")
    compartidos = m[m.estaciones_en_pixel > 1].groupby("pixel").station.apply(lambda s: ", ".join(map(str, s))).reset_index()
    guarda_tab(compartidos, "Tab25_pixeles_compartidos.csv")

    # Pares estación-año con su curva satelital del mismo año y píxel
    dfe = pd.DataFrame({"station": ze["station"], "year": ze["year"], "ie": np.arange(len(ze["year"]))})
    dfs = pd.DataFrame({"pixel": zs["pixel"], "year": zs["year"], "is_": np.arange(len(zs["year"]))})
    pares = dfe.merge(m[["station", "pixel", "alt", "zona", "clase_altitud", "lon", "lat"]], on="station").merge(dfs, on=["pixel", "year"])
    pares = pares.merge(oni, on="year", how="left").reset_index(drop=True)
    # Misma base para ambas fuentes (K común = K elegido para estaciones): la diferencia d(t) no puede
    # deberse a que una fuente se suavizó más que la otra.
    zsc = np.load(os.path.join(CACHE, "fase1_coef_satelite_Kcomun.npz"))
    Y = ze["C"][pares.ie.values] @ Phi_e.T          # estación (mm/día), K = K_est
    X = zsc["C"][pares.is_.values] @ Phi_e.T        # satélite (mm/día), K = K_est
    D = Y - X                                        # d_i(t) = estación - satélite
    n = len(pares)
    # Sensibilidad: ambas fuentes con K = K_sat
    zea = np.load(os.path.join(CACHE, "fase1_coef_estaciones_Ksat.npz"))
    D_alt = zea["C"][pares.ie.values] @ Phi_s.T - zs["C"][pares.is_.values] @ Phi_s.T
    D_mix = Y - zs["C"][pares.is_.values] @ Phi_s.T   # lo que pasaría con K distintos (6 vs 11)
    sens = []
    for nom, DD in ((f"ambas K={K_e} (principal)", D), (f"ambas K={K_s}", D_alt), (f"estación K={K_e} vs satélite K={K_s} (no recomendado)", D_mix)):
        Dsx = pd.DataFrame(DD).groupby(pares.station.values).mean()
        mu = Dsx.mean(axis=0).values; se = Dsx.std(axis=0, ddof=1).values / np.sqrt(len(Dsx))
        sens.append({"caso": nom, "d_integrado_mm": mu.sum(), "dias_sat>est": int((mu + 2 * se < 0).sum()),
                     "dias_est>sat": int((mu - 2 * se > 0).sum()),
                     "intervalos_sat>est": "; ".join(f"{i['desde']} a {i['hasta']}" for i in intervalos(mu + 2 * se < 0)),
                     "intervalos_est>sat": "; ".join(f"{i['desde']} a {i['hasta']}" for i in intervalos(mu - 2 * se > 0))})
    guarda_tab(pd.DataFrame(sens), "Tab26b_sensibilidad_K.csv")
    log("pares", n, "estaciones", pares.station.nunique(), "píxeles", pares.pixel.nunique())

    # Mapa (Fig. 15)
    Cs = zs["C"]; Xmean = (Cs @ Phi_s.T).sum(1)
    mapa = pd.DataFrame({"pixel": zs["pixel"], "tot": Xmean}).groupby("pixel").tot.mean()
    A = np.full(NFIL * NCOL, np.nan); A[mapa.index.values] = mapa.values
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    im = ax[0].imshow(A.reshape(NFIL, NCOL), extent=[LON0, LON0 + NCOL * RES, LAT1 - NFIL * RES, LAT1], cmap="YlGnBu")
    plt.colorbar(im, ax=ax[0], shrink=0.8, label="Total anual medio satélite 1981-2012 (mm)")
    for z_, mk in (("Cauca (26)", "o"), ("Pacífico (53/54)", "^")):
        s_ = m[m.zona == z_]
        ax[0].scatter(s_.lon, s_.lat, s=14, marker=mk, c=np.where(s_.estaciones_en_pixel > 1, "red", "k"), label=z_)
    ax[0].set_title(f"{len(m)} estaciones emparejadas (rojo: comparten píxel)"); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_xlabel("Longitud"); ax[0].set_ylabel("Latitud")
    ax[1].hist(m.dist_km_centro, bins=20, color="grey"); ax[1].set_xlabel("Distancia estación - centro del píxel (km)")
    ax[1].set_title(f"Distancia al centro del píxel (máx. teórico {haversine_km(0, 4, .025, 4.025):.1f} km)")
    guarda_fig(fig, "Fig15_emparejamiento_mapa.png")

    # ---- d(t) = estación - satélite con banda d̄ ± 2 EE (Fig. 16, Tab. 26)
    dbar = D.mean(0); ee = D.std(0, ddof=1) / np.sqrt(n)
    # versión por estación (la unidad independiente es la estación; los años de una estación no lo son)
    Dst = pd.DataFrame(D).groupby(pares.station.values).mean()
    ns = len(Dst); dbar_s = Dst.mean(axis=0).values; ee_s = Dst.std(axis=0, ddof=1).values / np.sqrt(ns)
    filas = []
    for nom, mu, se in (("pares estación-año", dbar, ee), ("estaciones (media por estación)", dbar_s, ee_s)):
        for signo, ind in (("estación > satélite", mu - 2 * se > 0), ("estación < satélite", mu + 2 * se < 0)):
            for itv in intervalos(ind):
                filas.append({"unidad": nom, "signo": signo, **itv})
    guarda_tab(pd.DataFrame(filas), "Tab26_intervalos_banda_excluye_cero.csv")
    # prueba pareada por cambio de signo (script t-funcional), sobre las medias por estación
    B = 5000
    Sg = rng.choice([-1.0, 1.0], size=(B, ns))
    Dm = Dst.values
    Tobs = np.abs(Dm.mean(0)) / (Dm.std(0, ddof=1) / np.sqrt(ns))
    Tmax = np.zeros(B)
    for b in range(B):
        Db = Dm * Sg[b][:, None]
        Tmax[b] = (np.abs(Db.mean(0)) / (Db.std(0, ddof=1) / np.sqrt(ns))).max()
    cmax = np.quantile(Tmax, 0.95); p_glob = (np.sum(Tmax >= Tobs.max()) + 1) / (B + 1)
    sig = intervalos(Tobs > cmax)
    guarda_tab(pd.DataFrame([{"estaciones": ns, "permutaciones": B, "T_max_obs": Tobs.max(), "c_max_95": cmax,
                              "p_global": p_glob, "intervalos_significativos": "; ".join(f"{i['desde']} a {i['hasta']}" for i in sig)}]),
               "Tab27_prueba_pareada_signos.csv")
    tot_y = Y.sum(1); tot_x = X.sum(1)
    pares["total_est_mm"] = tot_y; pares["total_sat_mm"] = tot_x
    pares["sesgo_rel_sat_%"] = 100 * (tot_x - tot_y) / tot_y
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(T_FINO, Y.mean(0), color=C_EST, lw=2, label="estaciones")
    ax[0].plot(T_FINO, X.mean(0), color=C_SAT, lw=2, label="satélite (píxel emparejado)")
    ax[0].set_title(f"Medias de {n} pares estación-año (misma base K={K_e})", fontsize=8); ax[0].set_ylabel("mm/día"); ax[0].legend(fontsize=7)
    sub = rng.choice(n, 300, replace=False)
    ax[1].plot(T_FINO, D[sub].T, color="grey", lw=0.3, alpha=0.4)
    for mu, se, col, lab in ((dbar, ee, "darkred", f"pares (n={n})"), (dbar_s, ee_s, "navy", f"estaciones (n={ns})")):
        ax[1].plot(T_FINO, mu, color=col, lw=2, label=f"d̄(t) {lab}")
        ax[1].fill_between(T_FINO, mu - 2 * se, mu + 2 * se, color=col, alpha=0.25)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylim(np.percentile(D, 1), np.percentile(D, 99))
    ax[1].set_title("d(t) = estación − satélite, d̄ ± 2 EE", fontsize=8); ax[1].legend(fontsize=7)
    ax[2].plot(T_FINO, Tobs, color="red", lw=2, label="|t(t)| observado")
    ax[2].axhline(cmax, color="k", ls="--", label=f"valor crítico máximo ({cmax:.2f})")
    ax[2].set_title(f"Prueba pareada por cambio de signo\n(B={B}, unidad = estación), p global = {p_glob:.4f}", fontsize=8); ax[2].legend(fontsize=7)
    for a_ in ax:
        eje_meses(a_)
    guarda_fig(fig, "Fig16_diferencia_d_t.png")

    # ---- Correlación cruzada funcional (Fig. 17, Tab. 28)
    # (a) curvas brutas: mezcla diferencias entre sitios y entre años, como en clase
    # (b) anomalías (curva - media de su estación / de su píxel): solo variación interanual
    def corr_cruzada(Xm, Ym):
        Xc = Xm - Xm.mean(0); Yc = Ym - Ym.mean(0)
        cov = Xc.T @ Yc / (len(Xm) - 1)
        return cov / np.outer(Xc.std(0, ddof=1), Yc.std(0, ddof=1))
    R_b = corr_cruzada(X, Y)
    Xa = X - pd.DataFrame(X).groupby(pares.station.values).transform("mean").values
    Ya = Y - pd.DataFrame(Y).groupby(pares.station.values).transform("mean").values
    R_a = corr_cruzada(Xa, Ya)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a_, R_, tit in ((ax[0], R_b, "(a) curvas brutas"), (ax[1], R_a, "(b) anomalías interanuales")):
        cs = a_.contourf(T_FINO, T_FINO, R_, levels=np.linspace(-1, 1, 21), cmap="RdBu_r")
        a_.plot([0, DIAS], [0, DIAS], "k:", lw=0.8); eje_meses(a_)
        a_.set_yticks(INICIO_MES + DIAS_MES / 2); a_.set_yticklabels(list("EFMAMJJASOND")); a_.set_ylim(0, DIAS)
        a_.set_xlabel("s: satélite"); a_.set_ylabel("t: estación"); a_.set_title(f"Correlación cruzada {tit}")
    plt.colorbar(cs, ax=ax[:2], shrink=0.8)
    ax[2].plot(T_FINO, np.diag(R_b), color="k", lw=2, label="diagonal r(t,t) bruta")
    ax[2].plot(T_FINO, np.diag(R_a), color="purple", lw=2, label="diagonal r(t,t) anomalías")
    ax[2].set_ylim(0, 1); ax[2].set_title("Diagonal de la correlación cruzada"); ax[2].legend(fontsize=7); eje_meses(ax[2])
    guarda_fig(fig, "Fig17_correlacion_cruzada.png")
    dg = pd.DataFrame({"t": T_FINO, "r_bruta": np.diag(R_b), "r_anomalia": np.diag(R_a)})
    guarda_tab(pd.DataFrame([{"tipo": c, "media": dg[c].mean(), "min": dg[c].min(), "fecha_min": fecha_txt(dg.t[dg[c].idxmin()]),
                              "max": dg[c].max(), "fecha_max": fecha_txt(dg.t[dg[c].idxmax()])} for c in ("r_bruta", "r_anomalia")]),
               "Tab28_diagonal_correlacion_cruzada.csv")

    # ---- Comparación de covarianzas (Fig. 18, Tab. 29)
    def cov(M_):
        Mc = M_ - M_.mean(0)
        return Mc.T @ Mc / (len(M_) - 1)
    Vy, Vx = cov(Y), cov(X)
    Vya, Vxa = cov(Ya), cov(Xa)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    lev = np.linspace(0, max(Vy.max(), Vx.max()), 16)
    for a_, V_, tit in ((ax[0], Vy, "Covarianza estaciones"), (ax[1], Vx, "Covarianza satélite (mismos pares)")):
        cs = a_.contourf(T_FINO, T_FINO, V_, levels=lev, cmap="viridis"); eje_meses(a_)
        a_.set_yticks(INICIO_MES + DIAS_MES / 2); a_.set_yticklabels(list("EFMAMJJASOND")); a_.set_title(tit + " (mm/día)²")
    plt.colorbar(cs, ax=ax[:2], shrink=0.8)
    ax[2].plot(T_FINO, np.sqrt(np.diag(Vy)), color=C_EST, lw=2, label="DE estaciones (bruta)")
    ax[2].plot(T_FINO, np.sqrt(np.diag(Vx)), color=C_SAT, lw=2, label="DE satélite (bruta)")
    ax[2].plot(T_FINO, np.sqrt(np.diag(Vya)), color=C_EST, lw=1.5, ls="--", label="DE estaciones (anomalía)")
    ax[2].plot(T_FINO, np.sqrt(np.diag(Vxa)), color=C_SAT, lw=1.5, ls="--", label="DE satélite (anomalía)")
    ax[2].set_title("Desviación estándar funcional por fuente"); ax[2].legend(fontsize=7); eje_meses(ax[2])
    guarda_fig(fig, "Fig18_comparacion_covarianzas.png")
    def eig1(V_):
        lam, U = np.linalg.eigh(V_); return lam[::-1], U[:, ::-1]
    ly, Uy = eig1(Vy); lx, Ux = eig1(Vx); lya, Uya = eig1(Vya); lxa, Uxa = eig1(Vxa)
    guarda_tab(pd.DataFrame([
        {"componente": "bruta", "var_total_est": np.trace(Vy), "var_total_sat": np.trace(Vx), "razon_sat/est": np.trace(Vx) / np.trace(Vy),
         "%var_CP1_est": 100 * ly[0] / ly.sum(), "%var_CP1_sat": 100 * lx[0] / lx.sum(), "|coseno(CP1_est, CP1_sat)|": abs(Uy[:, 0] @ Ux[:, 0]),
         "dist_relativa_Frobenius": np.linalg.norm(Vx - Vy) / np.linalg.norm(Vy)},
        {"componente": "anomalía interanual", "var_total_est": np.trace(Vya), "var_total_sat": np.trace(Vxa), "razon_sat/est": np.trace(Vxa) / np.trace(Vya),
         "%var_CP1_est": 100 * lya[0] / lya.sum(), "%var_CP1_sat": 100 * lxa[0] / lxa.sum(), "|coseno(CP1_est, CP1_sat)|": abs(Uya[:, 0] @ Uxa[:, 0]),
         "dist_relativa_Frobenius": np.linalg.norm(Vxa - Vya) / np.linalg.norm(Vya)}]), "Tab29_comparacion_covarianzas.csv")

    # ---- Por subregión / altitud y por fase ENSO (Fig. 19, Tab. 30, Tab. 31)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    res_g = []
    grupos = [("zona", g) for g in ("Cauca (26)", "Pacífico (53/54)")] + \
             [("clase_altitud", g) for g in ("<1200 m", "1200-2000 m", ">2000 m", "sin altitud")] + \
             [("fase_enso", g) for g in ("El Niño", "Neutral", "La Niña")]
    col_g = {"Cauca (26)": C_EST, "Pacífico (53/54)": "#2e8b57", "<1200 m": "#fdae61", "1200-2000 m": "#d7191c",
             ">2000 m": "#2b83ba", "sin altitud": "grey", "El Niño": "#d62728", "Neutral": "grey", "La Niña": "#1f77b4"}
    for var, g in grupos:
        idx = (pares[var] == g).values
        if idx.sum() < 5:
            continue
        st = pares.station[idx]
        Dg = pd.DataFrame(D[idx]).groupby(st.values).mean()       # unidad = estación
        mu = Dg.mean(axis=0).values; se = Dg.std(axis=0, ddof=1).values / np.sqrt(len(Dg))
        a_ = ax[{"zona": 0, "clase_altitud": 1, "fase_enso": 2}[var]]
        a_.plot(T_FINO, mu, color=col_g[g], lw=2, label=f"{g} ({len(Dg)} est., {idx.sum()} pares)")
        a_.fill_between(T_FINO, mu - 2 * se, mu + 2 * se, color=col_g[g], alpha=0.15)
        exc = intervalos(mu + 2 * se < 0)
        res_g.append({"variable": var, "grupo": g, "estaciones": len(Dg), "pares": int(idx.sum()),
                      "total_est_mm": tot_y[idx].mean(), "total_sat_mm": tot_x[idx].mean(),
                      "sesgo_rel_mediano_sat_%": np.median(pares["sesgo_rel_sat_%"][idx]),
                      "d_integrado_mm": mu.sum(),
                      "dias_sat>est_significativo": int((mu + 2 * se < 0).sum()),
                      "dias_est>sat_significativo": int((mu - 2 * se > 0).sum()),
                      "corr_diag_media_anomalia": np.mean(np.diag(corr_cruzada(Xa[idx], Ya[idx]))) if idx.sum() > 30 else np.nan})
    for a_, tit in zip(ax, ("Por subregión (zona hidrográfica)", "Por altitud", "Por fase ENSO (ONI anual)")):
        a_.axhline(0, color="k", lw=0.6); a_.set_title(f"d̄(t) ± 2EE — {tit}", fontsize=8); a_.legend(fontsize=6); eje_meses(a_)
    ax[0].set_ylabel("estación − satélite (mm/día)")
    guarda_fig(fig, "Fig19_diferencia_por_grupo.png")
    guarda_tab(pd.DataFrame(res_g), "Tab30_comparacion_por_grupo.csv")
    # sesgo relativo del total anual vs altitud (estación)
    se_st = pares.groupby("station").agg(alt=("alt", "first"), zona=("zona", "first"), sesgo=("sesgo_rel_sat_%", "median"),
                                         tot=("total_est_mm", "mean")).dropna(subset=["alt"])
    rho_alt = se_st.sesgo.corr(se_st.alt, method="spearman")
    # ANOVA funcional de d(t) por fase ENSO (script anova_funcional): F(t) puntual y calibración por
    # permutación de las fases ENTRE AÑOS (los pares de un mismo año comparten la fase)
    fases = pares.fase_enso.values
    anios = pares.year.values
    fase_anio = pd.Series(fases, index=anios).groupby(level=0).first()
    def F_t(lab):
        gm = D.mean(0); ssr = np.zeros(DIAS); sse = np.zeros(DIAS); k = 0
        for g in np.unique(lab):
            Dg = D[lab == g]; mg = Dg.mean(0); k += 1
            ssr += len(Dg) * (mg - gm) ** 2; sse += ((Dg - mg) ** 2).sum(0)
        return (ssr / (k - 1)) / (sse / (len(D) - k))
    Fobs = F_t(fases)
    Bp = 1000; Fmax = np.zeros(Bp)
    pos_anio = {a: np.where(anios == a)[0] for a in fase_anio.index}
    for b in range(Bp):
        perm = pd.Series(rng.permutation(fase_anio.values), index=fase_anio.index)
        Fmax[b] = F_t(perm.loc[anios].values).max()
    p_F = (np.sum(Fmax >= Fobs.max()) + 1) / (Bp + 1)
    guarda_tab(pd.DataFrame([{"F_max_obs": Fobs.max(), "fecha_F_max": fecha_txt(T_FINO[Fobs.argmax()]),
                              "c_max_95_perm_anios": np.quantile(Fmax, .95), "p_global": p_F, "permutaciones": Bp,
                              "intervalos_F>c": "; ".join(f"{i['desde']} a {i['hasta']}" for i in intervalos(Fobs > np.quantile(Fmax, .95))),
                              "Spearman(sesgo rel. total, altitud)": rho_alt, "estaciones_con_altitud": len(se_st)}]),
               "Tab31_anova_enso_y_altitud.csv")
    # Fig. 20: sesgo relativo vs altitud y medias por fase ENSO de cada fuente
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for z_, col in (("Cauca (26)", C_EST), ("Pacífico (53/54)", "#2e8b57")):
        s_ = se_st[se_st.zona == z_]
        ax[0].scatter(s_.alt, s_.sesgo, s=12, color=col, label=z_)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("Altitud de la estación (m)")
    ax[0].set_ylabel("Sesgo relativo del total anual del satélite (%)"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"Sesgo del satélite vs altitud (Spearman = {rho_alt:.2f})")
    for f_, col in (("El Niño", "#d62728"), ("Neutral", "grey"), ("La Niña", "#1f77b4")):
        idx = fases == f_
        ax[1].plot(T_FINO, Y[idx].mean(0), color=col, lw=2, label=f"estación — {f_}")
        ax[1].plot(T_FINO, X[idx].mean(0), color=col, lw=1.5, ls="--", label=f"satélite — {f_}")
    ax[1].set_title("Medias por fase ENSO (continua: estación; discontinua: satélite)", fontsize=8)
    ax[1].legend(fontsize=6, ncol=2); eje_meses(ax[1]); ax[1].set_ylabel("mm/día")
    guarda_fig(fig, "Fig20_sesgo_altitud_enso.png")
    pares.drop(columns=["ie", "is_"]).to_csv(os.path.join(CACHE, "fase3_pares.csv"), index=False)
    guarda_tab(pd.DataFrame([{"pares": n, "estaciones": ns, "pixeles": pares.pixel.nunique(),
                              "total_est_medio_mm": tot_y.mean(), "total_sat_medio_mm": tot_x.mean(),
                              "sesgo_rel_mediano_%": np.median(pares["sesgo_rel_sat_%"]),
                              "pares_sat>est_%": 100 * np.mean(tot_x > tot_y)}]), "Tab32_resumen_comparacion.csv")
    return pd.DataFrame(res_g)



def fase3b():
    """ANOVA funcional por SUBREGIONES con varias estaciones cada una (asesoría 19-nov: ningún grupo
    puede tener una sola curva) y comparación de pares de subregiones con t de Welch funcional.
    Unidad experimental = estación (curva media de sus años 1981-2003): las estaciones de subregiones
    distintas son poblaciones INDEPENDIENTES, sin emparejamiento natural (ni siquiera tienen el mismo
    tamaño), por eso no aplica la t pareada; Welch no supone varianzas iguales (Pacífico vs Andina
    difieren mucho en varianza y en n). Calibración por permutación de etiquetas de estación (máx. en t)."""
    log("FASE 3b: ANOVA funcional por subregiones + Welch")
    rng = np.random.default_rng(SEMILLA)
    ze, zs = carga_coef()
    Phi_e = base_bspline(T_FINO, int(ze["K"])); Phi_s = base_bspline(T_FINO, int(zs["K"]))
    pares = pd.read_csv(os.path.join(CACHE, "fase3_pares.csv"))
    dfe = pd.DataFrame({"station": ze["station"], "year": ze["year"], "ie": np.arange(len(ze["year"]))})
    dfs = pd.DataFrame({"pixel": zs["pixel"], "year": zs["year"], "is_": np.arange(len(zs["year"]))})
    pares = pares.merge(dfe, on=["station", "year"]).merge(dfs, on=["pixel", "year"])
    Y = ze["C"][pares.ie.values] @ Phi_e.T; X = zs["C"][pares.is_.values] @ Phi_s.T
    pares["subregion"] = np.where(pares.zona.str.startswith("Pac"), "Pacífico",
                                  np.where(pares.alt < 1200, "Andina valle (<1200 m)",
                                           np.where(pares.alt <= 2000, "Andina ladera (1200-2000 m)", "Andina alta (>2000 m)")))
    sub_st = pares.groupby("station").subregion.first()
    grupos = ["Pacífico", "Andina valle (<1200 m)", "Andina ladera (1200-2000 m)", "Andina alta (>2000 m)"]
    n_g = sub_st.value_counts()
    assert all(n_g.get(g, 0) >= 5 for g in grupos), n_g
    curvas = {"lluvia estación Y(t)": Y, "sesgo d(t)=estación-satélite": Y - X}

    def F_t(Z, lab):
        gm = Z.mean(0); ssr = np.zeros(Z.shape[1]); sse = np.zeros(Z.shape[1]); k = len(np.unique(lab))
        for g in np.unique(lab):
            Zg = Z[lab == g]; mg = Zg.mean(0); ssr += len(Zg) * (mg - gm) ** 2; sse += ((Zg - mg) ** 2).sum(0)
        return (ssr / (k - 1)) / (sse / (len(Z) - k))

    def welch_t(Z1, Z2):
        v1, v2 = Z1.var(0, ddof=1) / len(Z1), Z2.var(0, ddof=1) / len(Z2)
        return (Z1.mean(0) - Z2.mean(0)) / np.sqrt(v1 + v2), (v1 + v2) ** 2 / (v1 ** 2 / (len(Z1) - 1) + v2 ** 2 / (len(Z2) - 1))

    B = 2000
    filas_a, filas_w = [], []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    col_g = dict(zip(grupos, ["#2e8b57", "#fdae61", "#d7191c", "#2b83ba"]))
    for k_, (nom, Z) in enumerate(curvas.items()):
        Zs = pd.DataFrame(Z).groupby(pares.station.values).mean()
        lab = sub_st.loc[Zs.index].values; Zs = Zs.values
        Fobs = F_t(Zs, lab)
        Fmax = np.array([F_t(Zs, rng.permutation(lab)).max() for _ in range(B)])
        c95 = np.quantile(Fmax, 0.95)
        filas_a.append({"variable": nom, **{f"n_{g}": int((lab == g).sum()) for g in grupos},
                        "F_max_obs": Fobs.max(), "fecha_F_max": fecha_txt(T_FINO[Fobs.argmax()]), "c_max_95_perm": c95,
                        "p_global_perm": (np.sum(Fmax >= Fobs.max()) + 1) / (B + 1), "permutaciones": B,
                        "intervalos_F>c95": "; ".join(f"{i['desde']} a {i['hasta']}" for i in intervalos(Fobs > c95))})
        for g in grupos:
            Zg = Zs[lab == g]; mu = Zg.mean(0); se = Zg.std(0, ddof=1) / np.sqrt(len(Zg))
            ax[k_].plot(T_FINO, mu, color=col_g[g], lw=1.8, label=f"{g} (n={len(Zg)})")
            ax[k_].fill_between(T_FINO, mu - 2 * se, mu + 2 * se, color=col_g[g], alpha=0.12)
        ax[k_].set_title(f"{nom}: F máx={Fobs.max():.1f}, p={filas_a[-1]['p_global_perm']:.4f}", fontsize=8); eje_meses(ax[k_])
        ax[k_].legend(fontsize=6)
        ax[2].plot(T_FINO, Fobs, label=f"F(t) {nom.split(' ')[0]}"); ax[2].axhline(c95, ls=":", lw=0.8)
        for a_i, g1 in enumerate(grupos):
            for g2 in grupos[a_i + 1:]:
                Z1, Z2 = Zs[lab == g1], Zs[lab == g2]
                tw, df_ = welch_t(Z1, Z2)
                Zc = np.vstack([Z1, Z2]); n1 = len(Z1)
                Tmax = np.zeros(B)
                for b in range(B):
                    p_ = rng.permutation(len(Zc)); Tmax[b] = np.abs(welch_t(Zc[p_[:n1]], Zc[p_[n1:]])[0]).max()
                c_ = np.quantile(Tmax, 0.95)
                razon_var = (Z1.var(0, ddof=1) / Z2.var(0, ddof=1))
                filas_w.append({"variable": nom, "grupo_1": g1, "grupo_2": g2, "n1": n1, "n2": len(Z2),
                                "max|t_Welch|": np.abs(tw).max(), "fecha_max": fecha_txt(T_FINO[np.abs(tw).argmax()]),
                                "gl_Satterthwaite_en_max": df_[np.abs(tw).argmax()], "c95_perm_max|t|": c_,
                                "p_global_perm": (np.sum(Tmax >= np.abs(tw).max()) + 1) / (B + 1),
                                "razon_varianzas_min": razon_var.min(), "razon_varianzas_max": razon_var.max(),
                                "dias_significativos": int((np.abs(tw) > c_).sum()),
                                "intervalos": "; ".join(f"{i['desde']} a {i['hasta']}" for i in intervalos(np.abs(tw) > c_))})
    ax[2].set_title("F(t) puntual y umbral 95 % por permutación", fontsize=8); ax[2].legend(fontsize=6); eje_meses(ax[2])
    ax[0].set_ylabel("mm/día (media por estación ± 2EE)")
    guarda_fig(fig, "Fig21_anova_funcional_subregiones.png")
    guarda_tab(pd.DataFrame(filas_a), "Tab33_anova_funcional_subregiones.csv")
    guarda_tab(pd.DataFrame(filas_w), "Tab34_welch_funcional_subregiones.csv")
    return pd.DataFrame(filas_a)


# =====================================================================================
# INTERFAZ PARA LA FASE 4 (red funcional: cabezas KAN, MLP y lineal)
# =====================================================================================

def interfaz_fase4():
    """Escribe interfaz_fase4/ con todo lo que la fase 4 necesita sin rehacer las fases 0-3."""
    log("Interfaz fase 4")
    out = os.path.join(AQUI, "interfaz_fase4")
    os.makedirs(out, exist_ok=True)
    ze, zs = carga_coef()
    zsc = np.load(os.path.join(CACHE, "fase1_coef_satelite_Kcomun.npz"))
    K = int(ze["K"])
    pares = pd.read_csv(os.path.join(CACHE, "fase3_pares.csv"))
    dfe = pd.DataFrame({"station": ze["station"], "year": ze["year"], "ie": np.arange(len(ze["year"]))})
    dfs = pd.DataFrame({"pixel": zs["pixel"], "year": zs["year"], "is_": np.arange(len(zs["year"]))})
    pp = pares.merge(dfe, on=["station", "year"]).merge(dfs, on=["pixel", "year"])
    # coeficientes en escala sqrt (la fase 4 modela sqrt(P)); misma base y mismo K
    z0 = np.load(os.path.join(CACHE, "fase0_matriz_diaria.npz"))
    Xs, meta_s = matriz_satelite()
    Phi_d = base_bspline(T_FINO, K); Phi_p = base_bspline(T_PENT, K)
    Csq_e = np.zeros((len(pp), K)); Csq_s = np.zeros((len(pp), K))
    for r, (ie, is_) in enumerate(zip(pp.ie.values, pp.is_.values)):
        obs = np.isfinite(z0["Y"][ie])
        Csq_e[r] = mco_mano(Phi_d[obs], np.sqrt(z0["Y"][ie, obs]))[0]
    Csq_s = mco_mano(Phi_p, np.sqrt(Xs[pp.is_.values]).T)[0].T
    W = gram(K); Rr = gram(K, deriv=2)
    np.savez_compressed(os.path.join(out, "pares_coeficientes.npz"),
                        C_est_mm=ze["C"][pp.ie.values], C_sat_mm=zsc["C"][pp.is_.values],
                        C_est_sqrt=Csq_e, C_sat_sqrt=Csq_s, W=W, R=Rr, K=K, nodos=nodos_bspline(K), orden=ORDEN,
                        station=pp.station.values, year=pp.year.values, pixel=pp.pixel.values,
                        alt=pp.alt.values, oni=pp.oni_anual.values)
    pp[["station", "year", "pixel", "lon", "lat", "alt", "zona", "oni_anual", "fase_enso"]].to_csv(
        os.path.join(out, "pares_metadatos.csv"), index=False)
    # Particiones: oficial del enunciado y la de la fe de erratas (procedencia por confirmar)
    part = pd.DataFrame({"year": np.arange(1981, 2014)})
    part["oficial_1981_2005_2006_2009_2010_2013"] = np.where(part.year <= 2005, "entrenamiento", np.where(part.year <= 2009, "validacion", "prueba"))
    part["fe_erratas_1981_1995_1996_2000_2001_2003"] = np.where(part.year <= 1995, "entrenamiento", np.where(part.year <= 2000, "validacion", np.where(part.year <= 2003, "prueba", "sin_datos_estacion")))
    part["pares_disponibles"] = part.year.map(pp.groupby("year").size()).fillna(0).astype(int)
    part.to_csv(os.path.join(out, "particiones.csv"), index=False)
    # Verificación numérica de W: ∫ x(t) dt de una curva = 1' W c ... (aquí: ∫φ_k = suma de columnas de W porque Σφ=1)
    chk = {"K": K, "max|W-W'|": float(np.abs(W - W.T).max()), "sum(W)=365": float(W.sum()),
           "min_valor_propio_W": float(np.linalg.eigvalsh(W).min()), "pares": int(len(pp))}
    json.dump(chk, open(os.path.join(out, "verificacion_W.json"), "w"), indent=1)
    return pd.DataFrame([chk])


# =====================================================================================
# FASE 4. Red funcional (codificador funcional + cabeza lineal/KAN/MLP + decodificador FPCA por
# régimen ENSO). Partición de la asesoría 20-nov: 1981-1996 entrenamiento+validación (validación
# cruzada temporal, sin fuga) y 1997-2003 prueba en tres tramos (Niño 97-98, Niña 99-00, neutro 01-03).
# PyTorch se permite en este ítem (enunciado). La KAN se programa aquí (B-splines en las aristas).
# =====================================================================================

DIAS_SECOS = np.r_[0:59, 151:243, 334:365]          # dic-feb y jun-ago (temporadas secas)
DIAS_HUMEDOS = np.r_[59:151, 243:334]               # mar-may y sep-nov (temporadas lluviosas)


def fase4():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    from scipy import stats
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    log("FASE 4: red funcional, ablación, evaluación e interpretación")
    rng = np.random.default_rng(SEMILLA)
    Z = np.load(os.path.join(AQUI, "interfaz_fase4", "pares_coeficientes.npz"))
    K = int(Z["K"]); W = Z["W"]; Rr = Z["R"]
    # Escala de modelación: raíz de la CURVA suavizada en mm/día (no la curva ajustada a la raíz de los
    # datos diarios: al elevar al cuadrado esa curva se pierde la varianza diaria (desigualdad de Jensen)
    # y el total anual cae ~60 %). sqrt(max(x(t),0)) se reproyecta en la misma base (K=6) con MCO a mano.
    Phi0 = base_bspline(T_FINO, K)
    Ce = mco_mano(Phi0, np.sqrt(np.maximum(Z["C_est_mm"] @ Phi0.T, 0)).T)[0].T
    Cs = mco_mano(Phi0, np.sqrt(np.maximum(Z["C_sat_mm"] @ Phi0.T, 0)).T)[0].T
    st, yr, alt = Z["station"], Z["year"], Z["alt"].astype(float)
    oni = lee_oni().set_index("year")
    oni_v = oni.oni_anual.reindex(yr).values
    REGS = ["El Niño", "Neutral", "La Niña"]
    ireg = np.array([REGS.index(f) for f in oni.fase_enso.reindex(yr).values])
    Phi = base_bspline(T_FINO, K)
    N = len(yr)
    # Estaciones reservadas (el docente no publicó lista): 10 % de las estaciones por sorteo con semilla
    est_u = np.unique(st)
    reserv = np.sort(rng.choice(est_u, size=int(round(0.1 * len(est_u))), replace=False))
    es_res = np.isin(st, reserv)
    TRAMOS = {"T1 Niño 1997-1998": (1997, 1998), "T2 Niña 1999-2000": (1999, 2000), "T3 neutro 2001-2003": (2001, 2003)}
    tramo = np.full(N, "entrenamiento 1981-1996", dtype=object)
    for nom, (a, b) in TRAMOS.items():
        tramo[(yr >= a) & (yr <= b)] = nom
    TR = (yr <= ANIO_TRAIN_FIN) & ~es_res
    TE = (yr > ANIO_TRAIN_FIN) & ~es_res
    FOLDS = [(1988, (1989, 1990)), (1990, (1991, 1992)), (1992, (1993, 1994)), (1994, (1995, 1996))]

    # ---- M (número de funciones propias): menor M en [4, 6] con >= 95 % de varianza en los tres regímenes
    acum = {r: np.cumsum(fpca_coef(Ce[TR & (ireg == r)], W)["var_exp"]) for r in range(3)}
    M = next((m for m in (4, 5, 6) if min(acum[r][m - 1] for r in range(3)) >= 0.95), 6)
    log("M =", M, {REGS[r]: round(100 * acum[r][M - 1], 2) for r in range(3)})

    def decodificador(idx, por_regimen=True):
        cb = np.zeros((N, K)); Bm = np.zeros((N, K, M)); ve = {}
        for r in range(3):
            fp = fpca_coef(Ce[idx & (ireg == r)] if por_regimen else Ce[idx], W)
            cb[ireg == r] = fp["cbar"]; Bm[ireg == r] = fp["coef_psi"][:, :M]; ve[REGS[r]] = 100 * np.cumsum(fp["var_exp"])[M - 1]
        return torch.tensor(cb), torch.tensor(Bm), ve

    Wt = torch.tensor(W / DIAS); Rt = torch.tensor(Rr)
    esc_rug = np.trace(W) / np.trace(Rr)
    Ct_e = torch.tensor(Ce)

    COVSETS = {"satélite solo": (True, []), "altitud sola": (False, ["alt"]), "satélite + altitud": (True, ["alt"]),
               "satélite + ONI": (True, ["oni"]), "satélite + altitud + ONI": (True, ["alt", "oni"])}
    COMPLETO = "satélite + altitud + ONI"
    NF = 6                                                        # neuronas funcionales (6-8 del enunciado)
    H_KAN, G_KAN, K_SPL = 3, 5, 3

    class KANCapa(nn.Module):
        """phi_ij(x) = w_b·SiLU(x) + w_s·Σ_g c_g B_g(x), B-splines cúbicas en malla uniforme sobre
        [-3, 3] (x se comprime con 3·tanh(x/3) para quedar en la malla)."""
        def __init__(s, nin, nout, G=G_KAN, k=K_SPL, lim=3.0):
            super().__init__()
            h = 2 * lim / G
            s.k, s.lim = k, lim
            s.register_buffer("nod", torch.arange(-k, G + k + 1, dtype=torch.float64) * h - lim)
            s.c = nn.Parameter(0.1 * torch.randn(nin, nout, G + k))
            s.wb = nn.Parameter(torch.randn(nin, nout) / np.sqrt(nin))
            s.ws = nn.Parameter(torch.ones(nin, nout))
            s.register_buffer("mask", torch.ones(nin, nout))

        def base(s, x):
            x = x.unsqueeze(-1); t = s.nod
            B = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
            for p in range(1, s.k + 1):
                B = (x - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * B[..., :-1] + (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * B[..., 1:]
            return B

        def phi(s, x):
            xs = s.lim * torch.tanh(x / s.lim)
            spl = torch.einsum("nib,iob->nio", s.base(xs), s.c)
            return s.mask * (s.wb * Fnn.silu(x)[..., None] + s.ws * spl)

        def forward(s, x):
            f = s.phi(x); s._ult = f
            return f.sum(1)

        def reg(s):
            a = s._ult.abs().mean(0); l1 = a.sum(); p = a / (l1 + 1e-12)
            return l1 + (-(p * torch.log(p + 1e-12)).sum())

    class KAN(nn.Module):
        def __init__(s, nin, nout):
            super().__init__(); s.l1 = KANCapa(nin, H_KAN); s.l2 = KANCapa(H_KAN, nout); s.b = nn.Parameter(torch.zeros(nout))
        def forward(s, x):
            return s.l2(s.l1(x)) + s.b
        def reg(s):
            return s.l1.reg() + s.l2.reg()

    def cabeza(tipo, nin, h_mlp=None):
        if tipo == "lineal":
            return nn.Linear(nin, M)
        if tipo == "KAN":
            return KAN(nin, M)
        return nn.Sequential(nn.Linear(nin, h_mlp), nn.GELU(), nn.Linear(h_mlp, M))

    def n_par(m):
        return sum(p.numel() for p in m.parameters())

    def h_igualado(nin):
        nk = n_par(KAN(nin, M))
        return min(range(1, 400), key=lambda h: abs(n_par(cabeza("MLP", nin, h)) - nk))

    class Red(nn.Module):
        def __init__(s, tipo, covs, usa_sat):
            super().__init__()
            s.usa_sat, s.covs = usa_sat, covs
            nin = (NF if usa_sat else 0) + len(covs)
            s.A = nn.Parameter(0.3 * torch.randn(K, NF)) if usa_sat else None
            s.h = cabeza(tipo, nin, h_igualado(nin) if tipo == "MLP" else None)
            s.tipo = tipo

        def entradas(s, X):
            partes = []
            if s.usa_sat:
                partes.append(X["cs"] @ Wt @ s.A)                  # z_k = ∫ w_k(t) (x(t) - x̄(t)) dt / 365
            for c in s.covs:
                partes.append(X[c][:, None])
            return torch.cat(partes, 1)

        def forward(s, X, cb, Bm):
            sc = s.h(s.entradas(X))
            return cb + torch.einsum("nkm,nm->nk", Bm, sc), sc

        def rugosidad(s):
            return 0.0 if s.A is None else esc_rug * torch.trace(s.A.T @ Rt @ s.A) / DIAS

    def tensores(idx_tr):
        mu_c = Cs[idx_tr].mean(0)
        return {"cs": torch.tensor(Cs - mu_c),
                "alt": torch.tensor((alt - alt[idx_tr].mean()) / alt[idx_tr].std()),
                "oni": torch.tensor((oni_v - oni_v[idx_tr].mean()) / oni_v[idx_tr].std())}

    def perdida_L2(pred, idx):
        e = pred[idx] - Ct_e[idx]
        return (e @ (Wt) * e).sum(1).mean()

    LAMBDA_RUG = 1e-2
    HP = {"lineal": [{"lr": 0.03}, {"lr": 0.01}, {"lr": 0.003}],
          "MLP": [{"lr": 0.01, "wd": 0.0}, {"lr": 0.01, "wd": 1e-4}, {"lr": 0.01, "wd": 1e-3}],
          "KAN": [{"lr": 0.01, "lam": 0.0}, {"lr": 0.01, "lam": 1e-4}, {"lr": 0.01, "lam": 1e-3}]}
    EPOCAS_MAX, PACIENCIA, CADA = 2000, 200, 10

    def entrena(tipo, covset, hp, idx_tr, idx_va, semilla, epocas=EPOCAS_MAX, cb_Bm=None, modelo=None):
        torch.manual_seed(semilla)
        usa_sat, covs = COVSETS[covset]
        X = tensores(idx_tr)
        cb, Bm = cb_Bm if cb_Bm is not None else decodificador(idx_tr)[:2]
        m = modelo if modelo is not None else Red(tipo, covs, usa_sat)
        opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp.get("wd", 0.0))
        itr = torch.tensor(np.where(idx_tr)[0]); iva = torch.tensor(np.where(idx_va)[0]) if idx_va is not None else None
        mejor, mejor_ep, estado, hist = np.inf, epocas, None, []
        for ep in range(1, epocas + 1):
            opt.zero_grad()
            pred, _ = m(X, cb, Bm)
            loss = perdida_L2(pred, itr) + LAMBDA_RUG * m.rugosidad()
            if tipo == "KAN" and hp.get("lam", 0) > 0:
                loss = loss + hp["lam"] * m.h.reg()
            loss.backward(); opt.step()
            if iva is not None and ep % CADA == 0:
                with torch.no_grad():
                    v = float(perdida_L2(m(X, cb, Bm)[0], iva))
                hist.append(v)
                if v < mejor - 1e-6:
                    mejor, mejor_ep, estado = v, ep, {k: t.clone() for k, t in m.state_dict().items()}
                elif ep - mejor_ep >= PACIENCIA:
                    break
        if estado is not None:
            m.load_state_dict(estado)
        return m, mejor, mejor_ep, X, cb, Bm

    def poda(m, X, cb, Bm, idx_tr, hp, umbral=0.02, ajuste=300):
        """Poda (Liu et al., 2024): importancia de arista = media |phi_ij| en entrenamiento; se anulan
        las aristas con importancia < umbral·máx de su capa y se re-ajusta con L1 + entropía."""
        with torch.no_grad():
            m.h(m.entradas(X)[torch.tensor(np.where(idx_tr)[0])])
            n0 = 0; n1 = 0
            for capa in (m.h.l1, m.h.l2):
                imp = capa._ult.abs().mean(0)
                n0 += int(capa.mask.sum()); capa.mask[imp < umbral * imp.max()] = 0; n1 += int(capa.mask.sum())
        hp2 = dict(hp); hp2["lam"] = max(hp.get("lam", 0), 1e-4); hp2["lr"] = hp["lr"] / 2
        m, *_ = entrena("KAN", m.covset, hp2, idx_tr, None, SEMILLA, epocas=ajuste, cb_Bm=(cb, Bm), modelo=m)
        return m, n0, n1

    # ---- 0) Verificación de la KAN con una función sintética (antes de usar datos reales)
    torch.manual_seed(SEMILLA)
    xs_ = torch.tensor(rng.uniform(-1, 1, (2000, 2)))
    fs_ = torch.exp(torch.sin(np.pi * xs_[:, 0]) + xs_[:, 1] ** 2)
    ks = nn.Sequential(KANCapa(2, 1, G=20), KANCapa(1, 1, G=20))
    opt = torch.optim.Adam(ks.parameters(), lr=0.01)
    for ep in range(6000):
        opt.zero_grad(); l_ = ((ks(xs_[:1500]).squeeze() - fs_[:1500]) ** 2).mean(); l_.backward(); opt.step()
    opt = torch.optim.LBFGS(ks.parameters(), lr=0.5, max_iter=2000, line_search_fn="strong_wolfe")
    def _c():
        opt.zero_grad(); l_ = ((ks(xs_[:1500]).squeeze() - fs_[:1500]) ** 2).mean(); l_.backward(); return l_
    opt.step(_c)
    with torch.no_grad():
        rmse_sint = float(torch.sqrt(((ks(xs_[1500:]).squeeze() - fs_[1500:]) ** 2).mean()))
    log("KAN sintética exp(sin(pi x1)+x2^2), RMSE prueba =", rmse_sint)

    # ---- 1) Validación cruzada temporal (ventana creciente) para hiperparámetros, en 1981-1996
    filas_cv = []
    for tipo in ("lineal", "MLP", "KAN"):
        for j, hp in enumerate(HP[tipo]):
            for f, (fin_tr, (v0, v1)) in enumerate(FOLDS):
                itr = (yr <= fin_tr) & ~es_res; iva = (yr >= v0) & (yr <= v1) & ~es_res
                m, v, ep, X, cb, Bm = entrena(tipo, COMPLETO, hp, itr, iva, SEMILLA)
                fila = {"cabeza": tipo, "config": j, "hp": json.dumps(hp), "fold": f + 1, "entrena_hasta": fin_tr,
                        "valida": f"{v0}-{v1}", "L2_val": v, "mejor_epoca": ep}
                if tipo == "KAN":
                    m.covset = COMPLETO
                    m, n0, n1 = poda(m, X, cb, Bm, itr, hp)
                    with torch.no_grad():
                        fila.update({"L2_val_tras_poda": float(perdida_L2(m(X, cb, Bm)[0], torch.tensor(np.where(iva)[0]))),
                                     "aristas_antes": n0, "aristas_despues": n1})
                filas_cv.append(fila)
            log("CV", tipo, hp, np.mean([r["L2_val"] for r in filas_cv[-4:]]))
    CV = pd.DataFrame(filas_cv)
    guarda_tab(CV, "Tab35_cv_temporal_hiperparametros.csv")
    res_cv = CV.groupby(["cabeza", "config", "hp"]).agg(CV_media=("L2_val", "mean"), CV_de=("L2_val", "std"),
                                                       epoca_mediana=("mejor_epoca", "median")).reset_index()
    elegido = {t: res_cv[res_cv.cabeza == t].sort_values("CV_media").iloc[0] for t in ("lineal", "MLP", "KAN")}

    # ---- 2) Entrenamiento final (1981-1996 completo), 3 semillas, 5 conjuntos de covariables
    SEMILLAS = (SEMILLA, SEMILLA + 1, SEMILLA + 2)
    preds, modelos, filas_par = {}, {}, []
    dec_tr = decodificador(TR)
    for tipo in ("lineal", "MLP", "KAN"):
        hp = json.loads(elegido[tipo].hp); ep = int(elegido[tipo].epoca_mediana)
        for cs_ in COVSETS:
            for s in SEMILLAS:
                m, *_r, X, cb, Bm = entrena(tipo, cs_, hp, TR, None, s, epocas=ep, cb_Bm=dec_tr[:2])
                with torch.no_grad():
                    preds[(tipo, cs_, s)] = m(X, cb, Bm)[0].numpy()
                modelos[(tipo, cs_, s)] = (m, X, cb, Bm)
            filas_par.append({"cabeza": tipo, "covariables": cs_, "parametros_cabeza": n_par(m.h),
                              "parametros_totales": n_par(m), "hp": elegido[tipo].hp, "epocas": ep})
        log("entrenada cabeza", tipo)
    # KAN podada (modelo completo) y decodificador global (mezcla de regímenes) para comparar
    for s in SEMILLAS:
        m, X, cb, Bm = modelos[("KAN", COMPLETO, s)]
        import copy
        mp = copy.deepcopy(m); mp.covset = COMPLETO
        mp, n0, n1 = poda(mp, X, cb, Bm, TR, json.loads(elegido["KAN"].hp))
        with torch.no_grad():
            preds[("KAN podada", COMPLETO, s)] = mp(X, cb, Bm)[0].numpy()
        modelos[("KAN podada", COMPLETO, s)] = (mp, X, cb, Bm)
    dec_gl = decodificador(TR, por_regimen=False)
    for tipo in ("lineal", "MLP", "KAN"):
        for s in SEMILLAS:
            m, *_r = entrena(tipo, COMPLETO, json.loads(elegido[tipo].hp), TR, None, s,
                             epocas=int(elegido[tipo].epoca_mediana), cb_Bm=dec_gl[:2])
            X = tensores(TR)
            with torch.no_grad():
                preds[(tipo + " (FPCA global)", COMPLETO, s)] = m(X, *dec_gl[:2])[0].numpy()
    TP = pd.DataFrame(filas_par)
    nk = TP[(TP.cabeza == "KAN") & (TP.covariables == COMPLETO)].parametros_cabeza.iloc[0]
    nm = TP[(TP.cabeza == "MLP") & (TP.covariables == COMPLETO)].parametros_cabeza.iloc[0]
    TP["dif_%_KAN_vs_MLP"] = np.nan
    TP.loc[(TP.cabeza == "MLP") & (TP.covariables == COMPLETO), "dif_%_KAN_vs_MLP"] = 100 * abs(nk - nm) / nk
    guarda_tab(TP, "Tab36_parametros_y_presupuesto.csv")

    # ---- 3) Métricas en mm (reconstrucción no negativa: (max(ŷ_sqrt, 0))²)
    obs = np.maximum(Z["C_est_mm"] @ Phi.T, 0)                   # curva observada de la estación (mm/día)
    sat = np.maximum(Z["C_sat_mm"] @ Phi.T, 0)                   # curva satelital sin corregir (mm/día)

    def mm(C):
        y = C @ Phi.T
        return np.maximum(y, 0) ** 2, 100 * (y < 0).mean()

    def picos(Yc):
        return np.c_[59 + Yc[:, 59:181].argmax(1), 243 + Yc[:, 243:365].argmax(1)]

    def desfase(P, O, L=60):
        Pa = P - P.mean(1, keepdims=True); Oa = O - O.mean(1, keepdims=True)
        lags = np.arange(-L, L + 1)
        cc = np.array([(np.roll(Pa, -l, axis=1) * Oa).sum(1) for l in lags])
        return lags[cc.argmax(0)]                                   # > 0: la predicción va ATRASADA

    def metricas_curva(P, O):
        l2 = np.linalg.norm(P - O, axis=1) / np.linalg.norm(O, axis=1)
        r = np.array([np.corrcoef(p, o)[0, 1] for p, o in zip(P, O)])
        a = P.std(1) / O.std(1); b = P.mean(1) / O.mean(1)
        kge = 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
        sesgo = 100 * (P.sum(1) / O.sum(1) - 1)
        ep = np.abs(picos(P) - picos(O)).mean(1)
        return pd.DataFrame({"L2_rel": l2, "KGE": kge, "sesgo_acum_%": sesgo, "err_picos_dias": ep,
                             "atenuacion_sd_pred/sd_obs": a, "desfase_dias": desfase(P, O)})

    base_df = pd.DataFrame({"station": st, "year": yr, "tramo": tramo, "reservada": es_res, "regimen": [REGS[r] for r in ireg]})
    filas_m, por_curva = [], {}
    m_sat = metricas_curva(sat, obs)
    por_curva[("satélite sin corregir", "-")] = m_sat
    por_curva[("piso de representación sqrt K=6", "-")] = metricas_curva(np.maximum(Ce @ Phi.T, 0) ** 2, obs)
    for (tipo, cs_, s), C in preds.items():
        P, pct = mm(C)
        por_curva.setdefault((tipo, cs_), []).append(metricas_curva(P, obs).assign(pct_recortado=pct))
    for k_ in list(por_curva):
        if isinstance(por_curva[k_], list):
            L_ = por_curva[k_]
            por_curva[k_] = sum(L_) / len(L_)                       # media de las 3 semillas por curva
            por_curva[k_ + ("sd_semillas",)] = L_
    conjuntos = {"prueba 1997-2003": TE, **{t: TE & (tramo == t) for t in TRAMOS},
                 "reservadas 1997-2003": es_res & (yr > ANIO_TRAIN_FIN), "reservadas 1981-1996": es_res & (yr <= ANIO_TRAIN_FIN)}
    for k_, dfm in por_curva.items():
        if len(k_) == 3:
            continue
        for nom, idx in conjuntos.items():
            fila = {"modelo": k_[0], "covariables": k_[1], "conjunto": nom, "curvas": int(idx.sum()),
                    "estaciones": int(len(np.unique(st[idx])))}
            for c in ("L2_rel", "KGE", "sesgo_acum_%", "err_picos_dias", "atenuacion_sd_pred/sd_obs", "desfase_dias"):
                fila[c] = dfm[c][idx].mean() if c != "KGE" else dfm[c][idx].median()
            fila["|desfase|>15d_%"] = 100 * (dfm["desfase_dias"][idx].abs() > 15).mean()
            if k_ + ("sd_semillas",) in por_curva:
                fila["L2_rel_sd_semillas"] = np.std([d_["L2_rel"][idx].mean() for d_ in por_curva[k_ + ("sd_semillas",)]], ddof=1)
                fila["pct_recortado"] = por_curva[k_ + ("sd_semillas",)][0]["pct_recortado"].iloc[0]
            filas_m.append(fila)
    TM = pd.DataFrame(filas_m)
    guarda_tab(TM[TM.covariables.isin([COMPLETO, "-"])], "Tab37_metricas_prueba_tramos.csv")
    abl = TM[(TM.conjunto == "prueba 1997-2003") & TM.modelo.isin(["lineal", "MLP", "KAN"])]
    guarda_tab(abl.pivot_table(index="covariables", columns="modelo", values=["L2_rel", "KGE", "sesgo_acum_%"]).round(4)
               .reset_index(), "Tab39_ablacion_covariables_prueba.csv", index=False)
    guarda_tab(TM[TM.modelo.isin(["lineal", "MLP", "KAN"]) & TM.conjunto.str.startswith(("prueba", "T"))], "Tab39b_ablacion_covariables_tramos.csv")

    # ---- 4) Bootstrap (B = 2000, unidad = estación) y Wilcoxon pareada por estación
    B = 2000
    filas_b = []

    def compara(a, b, idx, nom_conj, metrica="L2_rel"):
        da = por_curva[a][metrica][idx].groupby(st[idx]).mean(); db = por_curva[b][metrica][idx].groupby(st[idx]).mean()
        dif = (da - db).values
        bs = np.array([dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        try:
            w = stats.wilcoxon(dif)
        except ValueError:
            w = type("w", (), {"statistic": np.nan, "pvalue": np.nan})()
        filas_b.append({"conjunto": nom_conj, "metrica": metrica, "A": f"{a[0]} [{a[1]}]", "B": f"{b[0]} [{b[1]}]",
                        "estaciones": len(dif), "media_A": da.mean(), "media_B": db.mean(), "dif_media_A-B": dif.mean(),
                        "IC95_boot_inf": lo, "IC95_boot_sup": hi, "IC_excluye_0": bool(lo > 0 or hi < 0),
                        "p_bootstrap_bilateral": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                        "Wilcoxon_W": w.statistic, "Wilcoxon_p": w.pvalue,
                        "A_mejor_en_%_estaciones": 100 * (dif < 0).mean() if metrica != "KGE" else 100 * (dif > 0).mean()})
    S0 = ("satélite sin corregir", "-")
    for nom, idx in [("prueba 1997-2003", TE)] + [(t, TE & (tramo == t)) for t in TRAMOS]:
        compara(("KAN", COMPLETO), ("MLP", COMPLETO), idx, nom)
        compara(("KAN", COMPLETO), ("lineal", COMPLETO), idx, nom)
        compara(("MLP", COMPLETO), ("lineal", COMPLETO), idx, nom)
        for t_ in ("lineal", "MLP", "KAN"):
            compara((t_, COMPLETO), S0, idx, nom)
    compara(("KAN", COMPLETO), ("MLP", COMPLETO), TE, "prueba 1997-2003", "KGE")
    compara(("KAN podada", COMPLETO), ("KAN", COMPLETO), TE, "prueba 1997-2003")
    for t_ in ("lineal", "MLP", "KAN"):
        compara((t_, COMPLETO), (t_ + " (FPCA global)", COMPLETO), TE, "prueba 1997-2003")
        for cs_ in COVSETS:
            if cs_ != COMPLETO:
                compara((t_, cs_), (t_, COMPLETO), TE, "prueba 1997-2003 (ablación)")
    TB = pd.DataFrame(filas_b)
    guarda_tab(TB, "Tab38_bootstrap_wilcoxon.csv")

    # ---- 5) Verificación: cabeza lineal ≡ regresión funcional β(s,t) = Σ_k Σ_m w_k(s) B_km ψ_m(t)
    m, X, cb, Bm = modelos[("lineal", COMPLETO, SEMILLA)]
    A_ = m.A.detach().numpy(); Bl = m.h.weight.detach().numpy(); b0 = m.h.bias.detach().numpy()
    nod = np.unique(nodos_bspline(K)); xg, wg = np.polynomial.legendre.leggauss(20)
    sq = np.concatenate([(b - a) / 2 * xg + (a + b) / 2 for a, b in zip(nod[:-1], nod[1:])])
    wq = np.concatenate([(b - a) / 2 * wg for a, b in zip(nod[:-1], nod[1:])])
    Phs = base_bspline(sq, K)
    wks = Phs @ A_                                                     # w_k(s) en la malla de cuadratura
    xc = (X["cs"].numpy()) @ Phs.T                                     # x_i(s) - x̄(s)
    dif_max = 0.0
    with torch.no_grad():
        yred = (m(X, cb, Bm)[0].numpy()) @ Phi.T
    for r in range(3):
        idx = np.where(ireg == r)[0]
        psi = Phi @ Bm[idx[0]].numpy()                                  # (365, M)
        beta = wks @ Bl[:, :NF].T @ psi.T                               # β(s,t) (n_s x 365)
        integ = (xc[idx] * wq) @ beta / DIAS
        cov = np.c_[X["alt"].numpy()[idx], X["oni"].numpy()[idx]] @ Bl[:, NF:].T @ psi.T
        alfa = Phi @ cb[idx[0]].numpy() + psi @ b0
        dif_max = max(dif_max, np.abs(alfa + integ + cov - yred[idx]).max())
    # ---- 6) Interpretación: w_k(t), activaciones KAN, ajuste simbólico, sequía/lluvia, MLP + SAE
    mk, Xk, cbk, Bmk = modelos[("KAN", COMPLETO, SEMILLA)]
    nombres_in = [f"z{k+1}" for k in range(NF)] + ["altitud", "ONI"]
    with torch.no_grad():
        Ein = mk.entradas(Xk).numpy()
    itr = np.where(TR)[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for k in range(NF):
        ax[0].plot(T_FINO, Phi @ mk.A.detach().numpy()[:, k], lw=1.3, label=f"w{k+1}(t)")
        ax[1].plot(T_FINO, Phi @ A_[:, k], lw=1.3)
    for a_, t_ in zip(ax, ("KAN", "lineal")):
        a_.axhline(0, color="k", lw=0.4); eje_meses(a_); a_.set_title(f"Funciones de peso w_k(t) — cabeza {t_}", fontsize=8)
    ax[0].legend(fontsize=6, ncol=2)
    guarda_fig(fig, "Fig22_funciones_peso_wk.png")
    # activaciones de la primera capa de la KAN en el rango de los datos
    capa = mk.h.l1
    fig, ax = plt.subplots(H_KAN, len(nombres_in), figsize=(14, 4.5), sharex="col")
    filas_s = []
    imp1 = None
    with torch.no_grad():
        capa(torch.tensor(Ein[itr])); imp1 = capa._ult.abs().mean(0).numpy()
        for i, nom in enumerate(nombres_in):
            xr = np.linspace(np.quantile(Ein[itr, i], 0.01), np.quantile(Ein[itr, i], 0.99), 200)
            xin = np.zeros((200, len(nombres_in))); xin[:, i] = xr
            ph = capa.phi(torch.tensor(xin)).numpy()[:, i, :]
            for j in range(H_KAN):
                ax[j, i].plot(xr, ph[:, j], color=C_EST if capa.mask[i, j] > 0 else "lightgrey", lw=1.2)
                ax[j, i].tick_params(labelsize=5)
                if j == 0:
                    ax[j, i].set_title(nom, fontsize=7)
                # ajuste simbólico: lineal, cuadrático, cúbico, tanh(a x + b)
                y_ = ph[:, j]; sst = ((y_ - y_.mean()) ** 2).sum() + 1e-12
                ajustes = {f"poli{g}": 1 - ((y_ - np.polyval(np.polyfit(xr, y_, g), xr)) ** 2).sum() / sst for g in (1, 2, 3)}
                try:
                    from scipy.optimize import curve_fit
                    pt, _ = curve_fit(lambda x, a, b, c, d: a * np.tanh(b * x + c) + d, xr, y_, p0=[np.ptp(y_) / 2, 1, 0, y_.mean()], maxfev=5000)
                    ajustes["a·tanh(bx+c)+d"] = 1 - ((y_ - (pt[0] * np.tanh(pt[1] * xr + pt[2]) + pt[3])) ** 2).sum() / sst
                except Exception:
                    pass
                mejor_f = max(ajustes, key=ajustes.get)
                filas_s.append({"entrada": nom, "neurona_oculta": j + 1, "importancia_|phi|": imp1[i, j], "activa": bool(capa.mask[i, j] > 0),
                                **{f"R2_{k}": v for k, v in ajustes.items()}, "mejor_forma": mejor_f,
                                "pendiente_media_signo": "+" if np.polyfit(xr, y_, 1)[0] > 0 else "-"})
    fig.suptitle("Activaciones φ_ij de la 1ª capa de la KAN (gris = podable / anulada)", fontsize=8)
    guarda_fig(fig, "Fig23_activaciones_kan.png")
    TS = pd.DataFrame(filas_s).sort_values("importancia_|phi|", ascending=False)
    guarda_tab(TS, "Tab45_ajuste_simbolico_activaciones_kan.csv")
    # ¿Captan las B-splines sequía y lluvia? Efecto parcial de cada entrada (p10 -> p90, resto en la
    # mediana) sobre la curva decodificada en mm/día, separado en temporadas secas y lluviosas.
    filas_e = []
    for tipo in ("KAN", "lineal", "MLP"):
        mm_, Xm, cbm, Bmm = modelos[(tipo, COMPLETO, SEMILLA)]
        with torch.no_grad():
            E = mm_.entradas(Xm).numpy()
            for r in range(3):
                i0 = np.where(TR & (ireg == r))[0][0]
                for i, nom in enumerate(nombres_in):
                    lo_, hi_ = np.quantile(E[itr, i], [0.1, 0.9])
                    base_in = np.median(E[itr], 0)
                    curvas_ = []
                    for v_ in (lo_, hi_):
                        e_ = base_in.copy(); e_[i] = v_
                        sc_ = mm_.h(torch.tensor(e_[None]))
                        c_ = cbm[i0] + Bmm[i0] @ sc_[0]
                        curvas_.append(np.maximum(c_.numpy() @ Phi.T, 0) ** 2)
                    dlt = curvas_[1] - curvas_[0]
                    filas_e.append({"cabeza": tipo, "regimen": REGS[r], "entrada": nom,
                                    "efecto_temporada_seca_mm_dia": dlt[DIAS_SECOS].mean(),
                                    "efecto_temporada_lluviosa_mm_dia": dlt[DIAS_HUMEDOS].mean(),
                                    "dias_aporte_(+)": int((dlt > 0.05).sum()), "dias_atenuacion_(-)": int((dlt < -0.05).sum())})
    TEf = pd.DataFrame(filas_e)
    guarda_tab(TEf, "Tab43_efecto_entradas_sequia_lluvia.csv")
    # MLP: activaciones ocultas y autoencoder disperso (SAE)
    mm_, Xm, cbm, Bmm = modelos[("MLP", COMPLETO, SEMILLA)]
    with torch.no_grad():
        Hact = Fnn.gelu(mm_.h[0](mm_.entradas(Xm))).numpy()
    seco_obs = obs[:, DIAS_SECOS].mean(1); hum_obs = obs[:, DIAS_HUMEDOS].mean(1)
    torch.manual_seed(SEMILLA)
    nh = Hact.shape[1]; nd = 2 * nh
    Hm, Hs = Hact[TR].mean(0), Hact[TR].std(0) + 1e-9
    Ht = torch.tensor((Hact - Hm) / Hs)
    enc = nn.Linear(nh, nd); dec = nn.Linear(nd, nh)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01)
    itr_t = torch.tensor(itr)
    for ep in range(3000):
        opt.zero_grad(); f_ = torch.relu(enc(Ht[itr_t])); l_ = ((dec(f_) - Ht[itr_t]) ** 2).mean() + 3e-3 * f_.abs().sum(1).mean()
        l_.backward(); opt.step()
    with torch.no_grad():
        Fs = torch.relu(enc(Ht)).numpy()
        rec_r2 = 1 - float(((dec(torch.tensor(Fs)) - Ht) ** 2).mean())
    filas_a = []
    for nom_, Mx in (("neurona MLP", Hact), ("rasgo SAE", Fs)):
        for k in range(Mx.shape[1]):
            v_ = Mx[TE, k]
            if v_.std() < 1e-9:
                continue
            filas_a.append({"unidad": nom_, "indice": k + 1, "activa_en_%_curvas_prueba": 100 * (np.abs(v_) > 1e-3).mean(),
                            "r_con_lluvia_temporada_seca": np.corrcoef(v_, seco_obs[TE])[0, 1],
                            "r_con_lluvia_temporada_lluviosa": np.corrcoef(v_, hum_obs[TE])[0, 1],
                            "r_con_ONI": np.corrcoef(v_, oni_v[TE])[0, 1]})
    TA = pd.DataFrame(filas_a)
    TA["R2_reconstruccion_SAE"] = rec_r2
    guarda_tab(TA, "Tab44_mlp_activaciones_y_sae.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    o_ = np.argsort(oni_v[TE])
    im_ = ax[0].imshow(Hact[TE][o_].T, aspect="auto", cmap="RdBu_r", vmin=-np.abs(Hact).max(), vmax=np.abs(Hact).max())
    ax[0].set_xlabel("curvas de prueba ordenadas por ONI (Niña → Niño)"); ax[0].set_ylabel("neurona oculta")
    ax[0].set_title("Activaciones GELU de la MLP", fontsize=8); plt.colorbar(im_, ax=ax[0])
    sae = TA[TA.unidad == "rasgo SAE"]
    ax[1].scatter(sae.r_con_lluvia_temporada_seca, sae.r_con_lluvia_temporada_lluviosa, s=12 + 0.5 * sae["activa_en_%_curvas_prueba"], color=C_SAT, label="rasgos SAE")
    mlp_ = TA[TA.unidad == "neurona MLP"]
    ax[1].scatter(mlp_.r_con_lluvia_temporada_seca, mlp_.r_con_lluvia_temporada_lluviosa, s=8, color=C_EST, label="neuronas MLP")
    ax[1].axhline(0, color="k", lw=0.4); ax[1].axvline(0, color="k", lw=0.4)
    ax[1].set_xlabel("r con lluvia de temporada seca"); ax[1].set_ylabel("r con lluvia de temporada lluviosa"); ax[1].legend(fontsize=6)
    ax[1].set_title("¿Qué codifica cada unidad? (prueba 1997-2003)", fontsize=8)
    guarda_fig(fig, "Fig24_mlp_activaciones_sae.png")

    # ---- 7) Figuras de evaluación: curvas medias por tramo, atenuación/desfase y peores casos
    Pm = {t_: np.mean([mm(preds[(t_, COMPLETO, s)])[0] for s in SEMILLAS], 0) for t_ in ("lineal", "MLP", "KAN")}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for a_, t in zip(ax, TRAMOS):
        idx = TE & (tramo == t)
        a_.plot(T_FINO, obs[idx].mean(0), "k", lw=2, label="estación (observada)")
        a_.plot(T_FINO, sat[idx].mean(0), color=C_SAT, lw=1.2, ls="--", label="satélite sin corregir")
        for t_, col_ in (("lineal", "grey"), ("MLP", "#2e8b57"), ("KAN", C_EST)):
            a_.plot(T_FINO, Pm[t_][idx].mean(0), color=col_, lw=1.3, label=t_)
        a_.set_title(f"{t} ({idx.sum()} curvas)", fontsize=8); eje_meses(a_)
    ax[0].legend(fontsize=6); ax[0].set_ylabel("mm/día (media de curvas)")
    guarda_fig(fig, "Fig25_prueba_tres_tramos.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for k_, (t_, col_) in enumerate((("lineal", "grey"), ("MLP", "#2e8b57"), ("KAN", C_EST))):
        d_ = TM[(TM.modelo == t_) & (TM.covariables == COMPLETO) & TM.conjunto.isin(list(TRAMOS))]
        ax[0].bar(np.arange(3) + 0.25 * k_ - 0.25, d_["atenuacion_sd_pred/sd_obs"], 0.25, color=col_, label=t_)
        ax[1].bar(np.arange(3) + 0.25 * k_ - 0.25, d_["desfase_dias"], 0.25, color=col_, label=t_)
    d_ = TM[(TM.modelo == "satélite sin corregir") & TM.conjunto.isin(list(TRAMOS))]
    ax[0].plot(range(3), d_["atenuacion_sd_pred/sd_obs"], "o", color=C_SAT, label="satélite")
    ax[1].plot(range(3), d_["desfase_dias"], "o", color=C_SAT, label="satélite")
    ax[0].axhline(1, color="k", lw=0.5); ax[1].axhline(0, color="k", lw=0.5)
    for a_, t_ in zip(ax, ("Atenuación: sd(pred)/sd(obs) (1 = sin atenuar)", "Desfase medio (días; >0 = atraso)")):
        a_.set_xticks(range(3)); a_.set_xticklabels([t.split(" ")[0] + " " + t.split(" ")[1] for t in TRAMOS]); a_.set_title(t_, fontsize=8)
    ax[0].legend(fontsize=6)
    guarda_fig(fig, "Fig26_atenuacion_desfase_tramos.png")
    dk = por_curva[("KAN", COMPLETO)]
    peores = dk["L2_rel"][TE].sort_values(ascending=False).index[:3]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), sharey=False)
    filas_p = []
    for a_, i in zip(ax, peores):
        a_.plot(T_FINO, obs[i], "k", lw=1.5, label="estación"); a_.plot(T_FINO, sat[i], color=C_SAT, ls="--", label="satélite")
        a_.plot(T_FINO, Pm["KAN"][i], color=C_EST, label="KAN"); a_.plot(T_FINO, Pm["MLP"][i], color="#2e8b57", label="MLP")
        a_.set_title(f"{st[i]} {yr[i]} ({REGS[ireg[i]]}), L2 rel KAN={dk['L2_rel'][i]:.2f}", fontsize=7); eje_meses(a_)
        filas_p.append({"station": st[i], "year": yr[i], "regimen": REGS[ireg[i]], "alt": alt[i], "L2_rel_KAN": dk["L2_rel"][i],
                        "L2_rel_satelite": m_sat["L2_rel"][i], "total_obs_mm": obs[i].sum(), "total_sat_mm": sat[i].sum(),
                        "total_KAN_mm": Pm["KAN"][i].sum()})
    ax[0].legend(fontsize=6)
    guarda_fig(fig, "Fig27_peores_casos_kan.png")
    guarda_tab(pd.DataFrame(filas_p), "Tab46_peores_casos.csv")
    # ablación de covariables (figura)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    cs_l = list(COVSETS)
    for k_, (t_, col_) in enumerate((("lineal", "grey"), ("MLP", "#2e8b57"), ("KAN", C_EST))):
        d_ = TM[(TM.modelo == t_) & (TM.conjunto == "prueba 1997-2003")].set_index("covariables").loc[cs_l]
        ax.bar(np.arange(len(cs_l)) + 0.25 * k_ - 0.25, d_.L2_rel, 0.25, yerr=d_.L2_rel_sd_semillas, color=col_, label=t_)
    ax.axhline(TM[(TM.modelo == "satélite sin corregir") & (TM.conjunto == "prueba 1997-2003")].L2_rel.iloc[0], color=C_SAT, ls="--", label="satélite sin corregir")
    ax.set_xticks(range(len(cs_l))); ax.set_xticklabels(cs_l, fontsize=7); ax.set_ylabel("L2 relativo (prueba 1997-2003)")
    ax.legend(fontsize=6); ax.set_title("Aporte de cada covariable (media ± DE de 3 semillas)", fontsize=8)
    guarda_fig(fig, "Fig28_ablacion_covariables.png")

    # ---- 8) Predicciones de las estaciones reservadas (modelo elegido por CV, no por la prueba)
    mejor_t = min(("lineal", "MLP", "KAN"), key=lambda t: elegido[t].CV_media)
    Pr = np.mean([mm(preds[(mejor_t, COMPLETO, s)])[0] for s in SEMILLAS], 0)
    ir = np.where(es_res)[0]
    pd.DataFrame({"station_id": np.repeat(st[ir], DIAS), "year": np.repeat(yr[ir], DIAS),
                  "day": np.tile(np.arange(1, DIAS + 1), len(ir)), "value_mm": Pr[ir].ravel().round(3)}).to_csv(
        os.path.join(AQUI, "predicciones_estaciones_reservadas.csv"), index=False)

    ver = {"M_funciones_propias": M, **{f"var_exp_%_{k}": v for k, v in dec_tr[2].items()},
           "RMSE_KAN_sintetica_exp(sin(pi x1)+x2^2)": rmse_sint, "max|lineal - regresion_funcional|": dif_max,
           "pct_recortado_KAN": TM[(TM.modelo == "KAN") & (TM.covariables == COMPLETO)].pct_recortado.iloc[0],
           "estaciones_reservadas": len(reserv), "curvas_entrenamiento": int(TR.sum()), "curvas_prueba": int(TE.sum()),
           "modelo_para_reservadas(elegido_por_CV)": mejor_t, "neuronas_funcionales": NF,
           "params_cabeza_KAN": int(nk), "params_cabeza_MLP": int(nm), "dif_params_%": 100 * abs(nk - nm) / nk,
           "lambda_rugosidad": LAMBDA_RUG, **{f"CV_{t}": float(elegido[t].CV_media) for t in elegido},
           **{f"hp_{t}": elegido[t].hp for t in elegido}}
    guarda_tab(pd.DataFrame([ver]).T.reset_index().rename(columns={"index": "verificacion", 0: "valor"}), "Tab42_verificaciones_fase4.csv")
    pd.DataFrame({"station": reserv}).to_csv(os.path.join(TAB, "Tab42b_estaciones_reservadas.csv"), index=False)
    guarda_tab(res_cv, "Tab35b_cv_resumen.csv")
    return pd.DataFrame([ver]).T



if __name__ == "__main__":
    fases = sys.argv[1:] or ["fase0", "fase1", "fase2", "fase3", "fase3b", "interfaz_fase4", "fase4"]
    for f in fases:
        r = globals()[f]()
        if isinstance(r, pd.DataFrame):
            print(r.to_string())
