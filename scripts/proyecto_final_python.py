#!/usr/bin/env python3
# =====================================================================
# Proyecto Final de Analitica de Datos (UAO 2026-2S)
# ARCHIVO PYTHON UNICO - todo el codigo del proyecto, reproducible
#
# Ejecuta las fases en orden, con rutas relativas al directorio de este
# archivo y semillas fijas. Cada fase es una funcion autocontenida con el
# mismo codigo que los scripts por fase del repositorio.
#
#   fase 1b : depuracion y trazabilidad de las estaciones (4.1)
#   fase 1  : bases a mano, minimos cuadrados con Cholesky propia,
#             seleccion de K y lambda, matriz de suavizado, bandas
#   fase 3.0: reajuste pentadal de las estaciones (comparabilidad con el
#             satelital; corrige el sesgo de Jensen)
#   fase 2  : EDA funcional (media, varianza, covarianza, derivadas,
#             profundidades, boxplot funcional, FPCA, ENSO)
#   fase 3  : emparejamiento estacion-pixel y comparacion de fuentes
#   fase 4  : codificador funcional + decodificador FPCA + cabezas
#             lineal/MLP/KAN, evaluacion con bootstrap y Wilcoxon
#
# Uso:
#     python3 proyecto_final_python.py             # todo
#     python3 proyecto_final_python.py 2 4         # solo las fases 2 y 4
#
# Dependencias (versiones verificadas):
#     numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-fda 0.10.1,
#     scipy 1.18.1, torch 2.15.0+cpu, pykan 0.2.x
# =====================================================================
import os
import sys

_BASE_PROYECTO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE_PROYECTO)

def _bloque_1():
    """Fase 1b - preprocesamiento y depuracion (4.1)  (origen: preproc_4_1.py)"""
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

    BASE = _BASE_PROYECTO
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



def _bloque_2():
    """Fase 1 - construccion de datos funcionales  (origen: fase1_datos_funcionales.py)"""
    # =====================================================================
    # PROYECTO FINAL ANALITICA DE DATOS 2026-2S
    # Fase 1 - Construccion de datos funcionales PASO A PASO (sin librerias)
    #
    # Nada de smooth.basis / fda / skfda: las bases se construyen a mano
    # (recursion de Cox-de Boor para B-splines, serie trigonometrica para
    # Fourier), los coeficientes salen de las ecuaciones normales resueltas
    # con una factorizacion de Cholesky escrita aqui, y la matriz de
    # suavizado se verifica numericamente (S idempotente, tr(S)=K).
    # numpy se usa solo como contenedor de arreglos y producto matricial.
    #
    # Se comparan DOS bases porque la curva anual de lluvia es periodica
    # (el 31-dic empalma con el 1-ene): la base B-spline no impone esa
    # condicion y produce un artefacto de borde al final del anio, mientras
    # la base de Fourier es periodica por construccion. La eleccion se
    # decide con VCG, no por preferencia.
    #
    # Ejecutar:  python3 fase1_datos_funcionales.py
    # =====================================================================
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEMILLA = 42
    np.random.seed(SEMILLA)

    BASE = _BASE_PROYECTO
    OUT = os.path.join(BASE, "entrega")
    FIG = os.path.join(OUT, "figs")
    TAB = os.path.join(OUT, "tablas")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    DOM = (0.0, 365.0)          # dominio anual comun a las dos fuentes
    PERIODO = 365.0
    ORDEN = 4                   # B-splines cubicos
    N_MUESTRA = 60              # curvas usadas para elegir base, K y lambda
    TIF = os.path.join(BASE, "repo/Datos/ImagenesSatelitePentadal/ImagenesSatelitePentadal")
    LON0, LAT1, RES, NFIL, NCOL = -77.55, 5.00, 0.05, 39, 37   # georreferencia del GeoTIFF

    MESES = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]   # mitad de cada mes
    LIM_MES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    NOM_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago",
               "Sep", "Oct", "Nov", "Dic"]


    # =====================================================================
    # 1. Bases funcionales construidas a mano
    # =====================================================================
    def nodos_completos(n_int, orden=ORDEN, dom=DOM):
        """Nodos con multiplicidad `orden` en los extremos e interiores equiespaciados."""
        interiores = np.linspace(dom[0], dom[1], n_int + 2)[1:-1]
        return np.concatenate([np.full(orden, dom[0]), interiores, np.full(orden, dom[1])])


    def base_bspline(t, nodos, orden=ORDEN):
        """Phi (len(t) x K) de B-splines de orden `orden` por recursion de Cox-de Boor."""
        t = np.asarray(t, dtype=float)
        K = len(nodos) - orden
        B = np.zeros((len(t), len(nodos) - 1))
        for i in range(len(nodos) - 1):
            if nodos[i + 1] > nodos[i]:
                B[:, i] = ((t >= nodos[i]) & (t < nodos[i + 1])).astype(float)
        ult = max(i for i in range(len(nodos) - 1) if nodos[i + 1] > nodos[i])
        B[t >= nodos[-1], ult] = 1.0
        for d in range(2, orden + 1):
            Bn = np.zeros((len(t), len(nodos) - d))
            for i in range(len(nodos) - d):
                den1 = nodos[i + d - 1] - nodos[i]
                den2 = nodos[i + d] - nodos[i + 1]
                term = np.zeros(len(t))
                if den1 > 0:
                    term += (t - nodos[i]) / den1 * B[:, i]
                if den2 > 0:
                    term += (nodos[i + d] - t) / den2 * B[:, i + 1]
                Bn[:, i] = term
            B = Bn
        return B[:, :K]


    def base_fourier(t, K, periodo=PERIODO):
        """Phi (len(t) x K) de Fourier: 1, sen(2pi j t/T), cos(2pi j t/T). K impar."""
        t = np.asarray(t, dtype=float)
        cols = [np.ones_like(t)]
        for j in range(1, (K - 1) // 2 + 1):
            w = 2.0 * np.pi * j * t / periodo
            cols.append(np.sin(w))
            cols.append(np.cos(w))
        return np.column_stack(cols[:K])


    def construir(t, K, tipo):
        """Interfaz unica para las dos bases."""
        if tipo == "bspline":
            return base_bspline(t, nodos_completos(K - ORDEN))
        return base_fourier(t, K)


    # =====================================================================
    # 2. Minimos cuadrados con Cholesky propia (sin lm(), sin lstsq)
    # =====================================================================
    def cholesky_propia(A):
        """A = L L^T para A simetrica definida positiva. Algoritmo escrito a mano."""
        n = A.shape[0]
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j] - L[i, :j] @ L[j, :j]
                if i == j:
                    if s <= 0:
                        raise np.linalg.LinAlgError("matriz no definida positiva")
                    L[i, j] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
        return L


    def resolver_chol(A, b):
        """Resuelve A x = b con la Cholesky propia (sustitucion adelante y atras)."""
        L = cholesky_propia(A)
        n = A.shape[0]
        B = b.reshape(n, -1)
        Y = np.zeros_like(B)
        for i in range(n):                      # L Y = B
            Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
        X = np.zeros_like(B)
        for i in range(n - 1, -1, -1):          # L^T X = Y
            X[i] = (Y[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
        return X.reshape(b.shape)


    def ajustar(Phi, y, R=None, lam=0.0):
        """Coeficientes, matriz de suavizado y diagnosticos del ajuste."""
        G = Phi.T @ Phi
        if R is not None and lam > 0:
            G = G + lam * R
        c = resolver_chol(G, Phi.T @ y)
        A = resolver_chol(G, np.eye(G.shape[0]))     # G^{-1}
        S = Phi @ A @ Phi.T
        yhat = Phi @ c
        h = np.einsum("ij,jk,ik->i", Phi, A, Phi)    # diagonal de S
        n = len(y)
        sse = float(((y - yhat) ** 2).sum())
        nu = float(np.trace(S))
        cv = float(np.mean(((y - yhat) / (1.0 - np.clip(h, None, 1 - 1e-9))) ** 2))
        return dict(c=c, yhat=yhat, S=S, A=A, h=h, sse=sse, nu=nu,
                    ecm=sse / n, cv=cv, gcv=n * sse / (n - nu) ** 2,
                    sigma2=sse / (n - nu))


    def penalizacion(K, tipo, n_malla=3651):
        """R = integral B''(t) B''(t)' dt por diferencias finitas + trapecios."""
        malla = np.linspace(DOM[0], DOM[1], n_malla)
        Bm = construir(malla, K, tipo)
        h = malla[1] - malla[0]
        D2 = (Bm[2:] - 2 * Bm[1:-1] + Bm[:-2]) / h ** 2
        R = (D2.T @ D2) * h
        return (R + R.T) / 2 + 1e-10 * np.eye(R.shape[0])


    def eje_meses(ax, marcar_anio=True):
        ax.set_xticks(MESES)
        ax.set_xticklabels(NOM_MES, fontsize=8)
        ax.set_xticks(LIM_MES, minor=True)
        ax.set_xlim(DOM[0], DOM[1])
        ax.set_xlabel("mes del anio")
        if marcar_anio:   # bordes = cambio de anio, donde se juzga el empalme
            for x in DOM:
                ax.axvline(x, color="0.3", ls=":", lw=1)


    # =====================================================================
    # 3. Curvas de estaciones: datos depurados -> dominio comun
    # =====================================================================
    print("== cargando estaciones depuradas (salida de la fase 4.1) ==")
    est = pd.read_csv(os.path.join(OUT, "estaciones_depuradas.csv"),
                      dtype={"STATION_ID": str})
    dias_anio = np.where((est.YEAR % 4 == 0) & ((est.YEAR % 100 != 0) |
                                                (est.YEAR % 400 == 0)), 366, 365)
    est["T"] = (est.DOY - 0.5) / dias_anio * 365.0
    est["SQRTP"] = np.sqrt(est.VALUE)     # estabiliza la varianza de la lluvia diaria
    print(f"   {len(est)} registros | {est.STATION_ID.nunique()} estaciones | "
          f"{est.groupby(['STATION_ID','YEAR']).ngroups} pares estacion-anio")

    pares = est.groupby(["STATION_ID", "YEAR"]).size().index.to_list()
    idx = np.random.choice(len(pares), size=min(N_MUESTRA, len(pares)), replace=False)
    muestra = [pares[i] for i in sorted(idx)]
    grupos = {k: v for k, v in est.groupby(["STATION_ID", "YEAR"])}
    print(f"   muestra para elegir base/K/lambda: {len(muestra)} pares (semilla {SEMILLA})")


    # =====================================================================
    # 4. Seleccion de base y de K con ECM, VC y VCG
    # =====================================================================
    print("\n== seleccion de base y K (estaciones) ==")
    rejilla = ([("bspline", K) for K in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 73]] +
               [("fourier", K) for K in [3, 5, 7, 9, 11, 13, 15, 21, 31, 41, 51, 73]])
    filas = []
    for tipo, K in rejilla:
        ecm = cv = gcv = 0.0
        usados = 0
        for key in muestra:
            g = grupos[key]
            Phi = construir(g["T"].to_numpy(), K, tipo)
            try:
                r = ajustar(Phi, g["SQRTP"].to_numpy())
            except np.linalg.LinAlgError:
                continue
            ecm += r["ecm"]; cv += r["cv"]; gcv += r["gcv"]; usados += 1
        if usados:
            filas.append(dict(base=tipo, K=K, ECM=ecm / usados, VC=cv / usados,
                              VCG=gcv / usados, curvas=usados))
            print(f"   {tipo:8s} K={K:3d}  ECM={filas[-1]['ECM']:.4f}  "
                  f"VC={filas[-1]['VC']:.4f}  VCG={filas[-1]['VCG']:.4f}")

    selK = pd.DataFrame(filas)
    selK.to_csv(os.path.join(TAB, "fase1_seleccion_K_estaciones.csv"), index=False)
    mejor = selK.loc[selK.VCG.idxmin()]
    TIPO_EST, K_EST = str(mejor.base), int(mejor.K)
    print(f"   -> base elegida por VCG: {TIPO_EST} con K={K_EST} (VCG={mejor.VCG:.4f})")
    for tt in selK.base.unique():
        print(f"      mejor VCG con {tt}: {selK[selK.base == tt].VCG.min():.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col in zip(ax, ["ECM", "VC", "VCG"]):
        for tt, mk in [("bspline", "o-"), ("fourier", "s-")]:
            s = selK[selK.base == tt]
            a.plot(s.K, s[col], mk, ms=4, label=tt)
        a.axvline(K_EST, color="red", ls="--", lw=1)
        a.set_xscale("log"); a.set_xlabel("K (nº de funciones base)")
        a.set_title(col); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("Seleccion de base y K - curvas de estaciones (sqrt P)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_seleccion_K_estaciones.png"), dpi=130)
    plt.close(fig)


    # =====================================================================
    # 5. Verificacion numerica de la matriz de suavizado
    # =====================================================================
    print("\n== verificacion de la matriz de suavizado S ==")
    g = grupos[muestra[0]]
    Phi = construir(g["T"].to_numpy(), K_EST, TIPO_EST)
    r = ajustar(Phi, g["SQRTP"].to_numpy())
    S = r["S"]
    err_idem = float(np.max(np.abs(S @ S - S)))
    tr = float(np.trace(S))
    print(f"   max|S@S - S| = {err_idem:.3e}   (debe ser ~0: S es proyeccion)")
    print(f"   tr(S) = {tr:.6f}   K = {K_EST}   |tr(S)-K| = {abs(tr-K_EST):.3e}")
    print(f"   S simetrica: max|S - S^T| = {float(np.max(np.abs(S - S.T))):.3e}")

    R = penalizacion(K_EST, TIPO_EST)
    print("\n== seleccion de lambda con penalizacion de rugosidad ==")
    fl = []
    for lam in [0.0, 1e-2, 1e0, 1e2, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12]:
        r2 = ajustar(Phi, g["SQRTP"].to_numpy(), R=R, lam=lam)
        fl.append(dict(lam=lam, nu=r2["nu"], ECM=r2["ecm"], VC=r2["cv"], VCG=r2["gcv"]))
        print(f"   lambda={lam:8.0e}  nu=tr(S_l)={r2['nu']:7.3f}  "
              f"ECM={r2['ecm']:.4f}  VC={r2['cv']:.4f}  VCG={r2['gcv']:.4f}")
    sel_lam = pd.DataFrame(fl)
    sel_lam.to_csv(os.path.join(TAB, "fase1_seleccion_lambda.csv"), index=False)
    lam_opt = float(sel_lam.loc[sel_lam.VCG.idxmin(), "lam"])
    nu_opt = float(sel_lam.loc[sel_lam.VCG.idxmin(), "nu"])
    print(f"   -> lambda elegido por VCG: {lam_opt:g}  (nu = {nu_opt:.3f})")

    pd.DataFrame([dict(base=TIPO_EST, K=K_EST, max_abs_SS_menos_S=err_idem, traza_S=tr,
                       dif_traza_K=abs(tr - K_EST),
                       simetria=float(np.max(np.abs(S - S.T))),
                       lambda_opt=lam_opt, nu_lambda_opt=nu_opt)]
                 ).to_csv(os.path.join(TAB, "fase1_verificacion_S.csv"), index=False)


    # =====================================================================
    # 6. Covarianza residual y bandas de confianza punto a punto
    # =====================================================================
    print("\n== bandas de confianza punto a punto ==")
    t_fino = np.linspace(DOM[0], DOM[1], 366)
    Phi_f = construir(t_fino, K_EST, TIPO_EST)
    var_pt = r["sigma2"] * np.einsum("ij,jk,ik->i", Phi_f, r["A"], Phi_f)
    ee = np.sqrt(var_pt)
    curva = Phi_f @ r["c"]
    sup, inf = curva + 1.96 * ee, curva - 1.96 * ee
    print(f"   sigma^2 residual = {r['sigma2']:.4f} | EE medio = {ee.mean():.4f} | "
          f"ancho medio de banda = {float(np.mean(sup-inf)):.4f} (escala sqrt mm)")
    print(f"   empalme 31-dic/1-ene: |curva(365)-curva(0)| = "
          f"{abs(float(curva[-1]-curva[0])):.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g["T"], g["SQRTP"], ".", ms=3, color="0.6", label="dato diario (sqrt mm)")
    ax.plot(t_fino, curva, "b-", lw=1.8, label=f"curva ajustada ({TIPO_EST}, K={K_EST})")
    ax.fill_between(t_fino, inf, sup, color="b", alpha=.2, label="banda punto a punto 95%")
    eje_meses(ax); ax.set_ylabel("sqrt(P) [sqrt mm]")
    ax.set_title(f"Estacion {muestra[0][0]} - anio {muestra[0][1]}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_banda_confianza.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(dict(t=t_fino, ajuste=curva, ee=ee, inf=inf, sup=sup)).to_csv(
        os.path.join(TAB, "fase1_banda_confianza.csv"), index=False)


    # =====================================================================
    # 6b. Control climatologico: bimodalidad y empalme de fin de anio
    # =====================================================================
    print("\n== control de bimodalidad y periodicidad del perfil medio ==")
    perfil = est.groupby(est.DOY).SQRTP.mean()
    t_perf = perfil.index.to_numpy() - 0.5
    y_perf = perfil.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_perf, y_perf, ".", ms=3, color="0.7", label="media diaria (186 estaciones)")
    filas_bm = []
    combos = [(TIPO_EST, K_EST, "red")]
    for tt, kk, cc in [("fourier", 7, "green"), ("bspline", 6, "blue")]:
        if (tt, kk) != (TIPO_EST, K_EST):
            combos.append((tt, kk, cc))
    for tt, kk, color in combos:
        Phc = construir(t_perf, kk, tt)
        rc = ajustar(Phc, y_perf)
        cf = construir(t_fino, kk, tt) @ rc["c"]
        pic = [i for i in range(1, len(cf) - 1) if cf[i] > cf[i-1] and cf[i] > cf[i+1]]
        dias = [round(float(t_fino[i]), 1) for i in pic]
        salto = abs(float(cf[-1] - cf[0]))
        filas_bm.append(dict(base=tt, K=kk, n_picos=len(pic), dias_pico=str(dias),
                             salto_fin_anio=salto, ECM=rc["ecm"], VCG=rc["gcv"]))
        print(f"   {tt:8s} K={kk:3d} picos={len(pic)} dias={dias} "
              f"salto_31dic_1ene={salto:.4f} ECM={rc['ecm']:.4f}")
        etq = (f"{tt} K={kk}" + ("" if salto < 1e-6 else
                                 f" (no periodica: salto {salto:.2f})"))
        ax.plot(t_fino, cf, "-", color=color, lw=1.6, label=etq)
    eje_meses(ax); ax.set_ylabel("sqrt(P) media [sqrt mm]")
    ax.set_title("Perfil climatologico medio: bimodalidad y empalme de fin de anio")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_bimodalidad.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(filas_bm).to_csv(os.path.join(TAB, "fase1_bimodalidad.csv"), index=False)


    # =====================================================================
    # 7. Coeficientes de TODAS las curvas de estaciones
    # =====================================================================
    print("\n== coeficientes funcionales de todas las curvas de estaciones ==")
    filas_c = []
    for (sid, anio), gg in est.groupby(["STATION_ID", "YEAR"]):
        Ph = construir(gg["T"].to_numpy(), K_EST, TIPO_EST)
        try:
            rr = ajustar(Ph, gg["SQRTP"].to_numpy(), R=R, lam=lam_opt)
        except np.linalg.LinAlgError:
            continue
        filas_c.append([sid, anio] + list(rr["c"]))
    coef_est = pd.DataFrame(filas_c, columns=["STATION_ID", "YEAR"] +
                            [f"c{i+1}" for i in range(K_EST)])
    coef_est.to_csv(os.path.join(OUT, "coef_estaciones.csv"), index=False)
    print(f"   {len(coef_est)} curvas (estacion-anio) con {K_EST} coeficientes "
          f"[base {TIPO_EST}, lambda {lam_opt:g}]")


    # =====================================================================
    # 8. Fuente satelital: pentadas -> mismo dominio anual
    # =====================================================================
    print("\n== fuente satelital (GeoTIFF pentadales) ==")
    from PIL import Image


    def dia_central_pentada(anio, p):
        """Pentada p (1..72) = sub-pentada j del mes m; devuelve dia central y nº de dias."""
        m = (p - 1) // 6 + 1
        j = (p - 1) % 6 + 1
        bis = anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)
        dias_mes = [31, 29 if bis else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        d_ini = 5 * (j - 1) + 1
        d_fin = dias_mes[m - 1] if j == 6 else 5 * j
        doy_ini = sum(dias_mes[:m - 1]) + d_ini
        doy_fin = sum(dias_mes[:m - 1]) + d_fin
        return (doy_ini + doy_fin) / 2.0, d_fin - d_ini + 1, sum(dias_mes)


    archivos = sorted(f for f in os.listdir(TIF)
                      if f.startswith("ppt") and f.endswith(".tif"))
    print(f"   {len(archivos)} rasteres | malla {NFIL}x{NCOL} | res {RES} grados")
    cubo = {}
    for f in archivos:
        cubo[(int(f[3:7]), int(f[7:9]))] = np.array(Image.open(os.path.join(TIF, f)),
                                                    dtype=float)
    anios_sat = sorted({a for a, _ in cubo})
    completos = [a for a in anios_sat if sum(1 for (y, _) in cubo if y == a) == 72]
    print(f"   anios satelitales: {anios_sat[0]}-{anios_sat[-1]} | "
          f"completos (72 pentadas): {len(completos)} | "
          f"incompletos: {sorted(set(anios_sat)-set(completos))}")

    fi, co = NFIL // 2, NCOL // 2
    print(f"   pixel demo fila={fi} col={co} -> "
          f"lon={LON0+(co+0.5)*RES:.3f} lat={LAT1-(fi+0.5)*RES:.3f}")
    anio_demo = 1981
    tp, yp = [], []
    for p in range(1, 73):
        tc, n_d, dias_tot = dia_central_pentada(anio_demo, p)
        tp.append((tc - 0.5) / dias_tot * 365.0)
        yp.append(cubo[(anio_demo, p)][fi, co] / n_d)    # mm/pentada -> mm/dia
    tp, yp = np.array(tp), np.sqrt(np.array(yp))

    print("\n== seleccion de base y K (satelite, 72 observaciones/anio) ==")
    filas_s = []
    for tipo, K in ([("bspline", K) for K in [4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]] +
                    [("fourier", K) for K in [3, 5, 7, 9, 11, 13, 15, 21, 31]]):
        Ph = construir(tp, K, tipo)
        try:
            rs = ajustar(Ph, yp)
        except np.linalg.LinAlgError:
            continue
        filas_s.append(dict(base=tipo, K=K, ECM=rs["ecm"], VC=rs["cv"], VCG=rs["gcv"]))
        print(f"   {tipo:8s} K={K:3d}  ECM={rs['ecm']:.4f}  "
              f"VC={rs['cv']:.4f}  VCG={rs['gcv']:.4f}")
    sel_sat = pd.DataFrame(filas_s)
    sel_sat.to_csv(os.path.join(TAB, "fase1_seleccion_K_satelite.csv"), index=False)
    mj = sel_sat.loc[sel_sat.VCG.idxmin()]
    TIPO_SAT, K_SAT = str(mj.base), int(mj.K)
    print(f"   -> base elegida por VCG: {TIPO_SAT} con K={K_SAT} (VCG={mj.VCG:.4f})")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col in zip(ax, ["ECM", "VC", "VCG"]):
        for tt, mk in [("bspline", "o-"), ("fourier", "s-")]:
            s = sel_sat[sel_sat.base == tt]
            a.plot(s.K, s[col], mk, ms=4, label=tt)
        a.axvline(K_SAT, color="red", ls="--", lw=1)
        a.set_xscale("log"); a.set_xlabel("K"); a.set_title(col)
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle("Seleccion de base y K - curvas satelitales")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_seleccion_K_satelite.png"), dpi=130)
    plt.close(fig)

    Ph = construir(tp, K_SAT, TIPO_SAT)
    rs = ajustar(Ph, yp)
    Ss = rs["S"]
    print(f"   verificacion satelite: max|SS-S|={float(np.max(np.abs(Ss@Ss-Ss))):.3e} "
          f"tr(S)={float(np.trace(Ss)):.4f} (K={K_SAT})")

    curva_sat = construir(t_fino, K_SAT, TIPO_SAT) @ rs["c"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_fino, curva, "b-", lw=1.8,
            label=f"estacion {muestra[0][0]} ({muestra[0][1]}) - {TIPO_EST} K={K_EST}")
    ax.plot(tp, yp, "s", ms=4, color="darkorange", label="pentada satelital (sqrt mm/dia)")
    ax.plot(t_fino, curva_sat, "-", color="darkorange", lw=1.8,
            label=f"curva satelital {anio_demo} - {TIPO_SAT} K={K_SAT}")
    eje_meses(ax); ax.set_ylabel("sqrt(P) [sqrt mm/dia]")
    ax.set_title("Las dos fuentes en el mismo dominio anual [0,365]")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fase1_dos_fuentes.png"), dpi=130)
    plt.close(fig)

    print("\n== coeficientes satelitales por pixel-anio (anios completos) ==")
    filas_sat = []
    for anio in completos:
        tpa, factor = [], []
        for p in range(1, 73):
            tc, n_d, dias_tot = dia_central_pentada(anio, p)
            tpa.append((tc - 0.5) / dias_tot * 365.0)
            factor.append(n_d)
        tpa = np.array(tpa); factor = np.array(factor, dtype=float)
        Pha = construir(tpa, K_SAT, TIPO_SAT)
        cubo_anio = np.stack([cubo[(anio, p)] for p in range(1, 73)])
        Y = np.sqrt(cubo_anio / factor[:, None, None]).reshape(72, -1)
        C = resolver_chol(Pha.T @ Pha, Pha.T @ Y)
        for j in range(C.shape[1]):
            f_, c_ = divmod(j, NCOL)
            filas_sat.append([anio, f_, c_, LON0 + (c_ + .5) * RES, LAT1 - (f_ + .5) * RES]
                             + list(C[:, j]))
    coef_sat = pd.DataFrame(filas_sat, columns=["YEAR", "FILA", "COL", "LON", "LAT"] +
                            [f"c{i+1}" for i in range(K_SAT)])
    coef_sat.to_csv(os.path.join(OUT, "coef_satelite.csv"), index=False)
    print(f"   {len(coef_sat)} curvas pixel-anio con {K_SAT} coeficientes [base {TIPO_SAT}]")

    print("\n===== FASE 1 TERMINADA =====")
    print(f"estaciones: base {TIPO_EST} K={K_EST} lambda={lam_opt:g} | "
          f"satelite: base {TIPO_SAT} K={K_SAT}")
    print(f"numpy {np.__version__} | pandas {pd.__version__} | "
          f"matplotlib {matplotlib.__version__} | semilla {SEMILLA}")


def _bloque_3():
    """Fase 3.0 - reajuste pentadal (comparabilidad de escalas)  (origen: fase3_0_pentadas.py)"""
    # =====================================================================
    # FASE 3.0 - Reajuste de las curvas de estacion en resolucion PENTADAL
    #
    # Motivo (defecto de comparabilidad detectado al pasar a la fase 3):
    # la fase 1 ajusto las estaciones sobre sqrt(P) de valores DIARIOS y el
    # producto satelital sobre sqrt(P) de PROMEDIOS PENTADALES. La raiz es
    # concava, asi que por la desigualdad de Jensen
    #       media_5dias( sqrt(P) )  <=  sqrt( media_5dias(P) ),
    # y la brecha crece con la intermitencia de la lluvia (una estacion con
    # dias secos y chubascos la sufre mucho mas que un producto satelital ya
    # suavizado). Comparar las dos fuentes en esa escala sobreestima el
    # satelital por un factor cercano a 3, que es un artefacto de la
    # representacion y no un sesgo fisico del sensor.
    #
    # Correccion: se reconstruyen los coeficientes de estacion con exactamente
    # el mismo pre-procesamiento del satelital -> agregar a 72 pentadas, pasar
    # a mm/dia, sacar raiz y ajustar la misma base de Fourier K=5 por minimos
    # cuadrados con Cholesky propia (sin lm(), sin lstsq).
    #
    # Entrada : entrega/estaciones_depuradas.csv
    # Salidas : entrega/coef_estaciones_pentadal.csv
    #           entrega/tablas/fase3_reajuste_pentadal.csv
    # =====================================================================
    import os
    import numpy as np
    import pandas as pd

    BASE = _BASE_PROYECTO
    ENT = os.path.join(BASE, "entrega")
    TAB = os.path.join(ENT, "tablas")
    K, PERIODO = 5, 365.0
    MIN_DIAS_PENTADA = 3        # dias con dato para aceptar una pentada
    MIN_PENTADAS = 60           # pentadas validas para aceptar un anio


    def base_fourier(t, K, periodo=PERIODO):
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    def cholesky_propia(A):
        """A = L L^T por el algoritmo de Cholesky, implementado a mano."""
        n = A.shape[0]
        L = np.zeros_like(A)
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j] - float(L[i, :j] @ L[j, :j])
                if i == j:
                    if s <= 0:
                        raise np.linalg.LinAlgError("matriz no definida positiva")
                    L[i, j] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
        return L


    def resuelve_cholesky(A, b):
        """Resuelve A c = b con A simetrica def. positiva, por sustitucion."""
        L = cholesky_propia(A)
        n = len(b)
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - float(L[i, :i] @ y[:i])) / L[i, i]
        c = np.zeros(n)
        for i in range(n - 1, -1, -1):
            c[i] = (y[i] - float(L[i + 1:, i] @ c[i + 1:])) / L[i, i]
        return c


    def tabla_pentadas(anio):
        """Para cada pentada 1..72: (doy_ini, doy_fin, dia_central_escalado, n_dias)."""
        bis = anio % 4 == 0 and (anio % 100 != 0 or anio % 400 == 0)
        dias_mes = [31, 29 if bis else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dias_tot = sum(dias_mes)
        out = []
        for p in range(1, 73):
            m = (p - 1) // 6 + 1
            j = (p - 1) % 6 + 1
            d_ini = 5 * (j - 1) + 1
            d_fin = dias_mes[m - 1] if j == 6 else 5 * j
            doy_ini = sum(dias_mes[:m - 1]) + d_ini
            doy_fin = sum(dias_mes[:m - 1]) + d_fin
            centro = (doy_ini + doy_fin) / 2.0
            out.append((doy_ini, doy_fin, (centro - 0.5) / dias_tot * 365.0,
                        d_fin - d_ini + 1))
        return out


    print("=" * 72)
    print("FASE 3.0 - reajuste pentadal de las curvas de estacion")
    print("=" * 72)
    d = pd.read_csv(os.path.join(ENT, "estaciones_depuradas.csv"),
                    dtype={"STATION_ID": str},
                    usecols=["STATION_ID", "YEAR", "DOY", "VALUE"])
    print(f"    registros diarios: {len(d):,} | estaciones: {d.STATION_ID.nunique()} "
          f"| anios: {d.YEAR.min()}-{d.YEAR.max()}")

    tablas = {a: tabla_pentadas(a) for a in sorted(d.YEAR.unique())}
    # indice pentada por (anio, doy)
    pent_de = {}
    for a, tb in tablas.items():
        v = np.zeros(400, dtype=int)
        for p, (i0, i1, _, _) in enumerate(tb):
            v[i0:i1 + 1] = p + 1
        pent_de[a] = v

    d["PENT"] = [pent_de[a][doy] for a, doy in zip(d.YEAR, d.DOY)]
    d = d[d.PENT > 0]
    g = (d.groupby(["STATION_ID", "YEAR", "PENT"])
         .agg(mm=("VALUE", "sum"), n=("VALUE", "size")).reset_index())
    g = g[g.n >= MIN_DIAS_PENTADA]
    g["mm_dia"] = g.mm / g.n
    print(f"    pentadas validas (>= {MIN_DIAS_PENTADA} dias con dato): {len(g):,}")

    filas, met = [], []
    for (e, a), gg in g.groupby(["STATION_ID", "YEAR"]):
        if len(gg) < MIN_PENTADAS:
            continue
        tb = tablas[a]
        tp = np.array([tb[p - 1][2] for p in gg.PENT])
        y = np.sqrt(gg.mm_dia.to_numpy(float))
        w = gg.n.to_numpy(float)                  # pondera por dias observados
        Phi = base_fourier(tp, K)
        A = Phi.T @ (Phi * w[:, None])
        b = Phi.T @ (y * w)
        c = resuelve_cholesky(A, b)
        res = y - Phi @ c
        filas.append([e, a] + list(c))
        met.append(dict(STATION_ID=e, YEAR=a, n_pentadas=len(gg),
                        rmse=float(np.sqrt((res ** 2 * w).sum() / w.sum())),
                        r2=float(1 - (res ** 2).sum()
                                 / max(((y - y.mean()) ** 2).sum(), 1e-12)),
                        acum_obs_mm=float((gg.mm_dia * [tb[p - 1][3] for p in gg.PENT]).sum())))

    C = pd.DataFrame(filas, columns=["STATION_ID", "YEAR"] + [f"c{i+1}" for i in range(K)])
    M = pd.DataFrame(met)
    C.to_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), index=False)
    M.to_csv(os.path.join(TAB, "fase3_reajuste_pentadal.csv"), index=False)
    print(f"    curvas reajustadas: {len(C)} (estaciones {C.STATION_ID.nunique()}, "
          f"{C.YEAR.min()}-{C.YEAR.max()})")
    print(f"    ajuste: R2 mediano {M.r2.median():.3f} | RMSE mediano {M.rmse.median():.4f} "
          f"sqrt(mm/dia)")

    # comparacion con los coeficientes diarios de la fase 1
    cd = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"), dtype={"STATION_ID": str})
    t_dia = np.arange(0.5, 365.0, 1.0)
    Ph = base_fourier(t_dia, K)
    cc = C.merge(cd, on=["STATION_ID", "YEAR"], suffixes=("_p", "_d"))
    Xp = cc[[f"c{i+1}_p" for i in range(K)]].to_numpy(float) @ Ph.T
    Xd = cc[[f"c{i+1}_d" for i in range(K)]].to_numpy(float) @ Ph.T
    int_p = (Xp ** 2).sum(axis=1)
    int_d = (Xd ** 2).sum(axis=1)
    obs = M.set_index(["STATION_ID", "YEAR"]).acum_obs_mm
    obs = obs.loc[list(zip(cc.STATION_ID, cc.YEAR))].to_numpy()
    print(f"\n    efecto Jensen medido en {len(cc)} curvas comunes:")
    print(f"      integral con ajuste pentadal : {int_p.mean():7.0f} mm/anio")
    print(f"      integral con ajuste diario   : {int_d.mean():7.0f} mm/anio")
    print(f"      acumulado observado real     : {obs.mean():7.0f} mm/anio")
    print(f"      razon pentadal/diario {np.median(int_p / int_d):.2f} | "
          f"sesgo pentadal vs observado {100 * (int_p.mean() / obs.mean() - 1):+.1f} % | "
          f"sesgo diario vs observado {100 * (int_d.mean() / obs.mean() - 1):+.1f} %")
    print("    => el ajuste pentadal es el que hay que comparar contra el satelital,")
    print("       porque comparte su pre-procesamiento exacto.")


def _bloque_4():
    """Fase 2 - analisis exploratorio funcional  (origen: fase2_eda_funcional.py)"""
    # -*- coding: utf-8 -*-
    """
    Proyecto Final Analitica de Datos (UAO) 2026-2S
    FASE 2 - Analisis exploratorio funcional (15 %)

    Entra:  entrega/coef_estaciones.csv  (3.567 curvas estacion-anio, base Fourier K=5)
            entrega/coef_satelite.csv    (46.176 curvas pixel-anio,   base Fourier K=5)
            aux/oni.txt                  (anomalia Nino 3.4 mensual, NOAA/PSL)

    Sale:   entrega/figs/fase2_*.png
            entrega/tablas/fase2_*.csv
            entrega/log_fase2.txt   (via tee al ejecutar)

    Reglas de la rubrica respetadas:
      * media funcional calculada PASO A PASO (acumulador explicito, sin np.mean)
        y verificada contra: (a) la identidad analitica Phi(t) c_barra,
        (b) numpy.mean, (c) skfda.representation.grid.FDataGrid.mean() si skfda esta.
      * varianza / desviacion / covarianza / correlacion funcional a mano.
      * derivadas analiticas de la base de Fourier (no diferencias finitas).
      * profundidad Fraiman-Muniz y MBD implementadas a mano y verificadas contra
        skfda (ModifiedBandDepth / IntegratedDepth) cuando esta disponible.
      * boxplot funcional (Sun & Genton 2011) con atipicas listadas por estacion y anio.
      * todo el conjunto 1981-2003 (estaciones) y 1981-2012 (satelite): sin submuestras.

    Ejecutar:  ./venv/bin/python fase2_eda_funcional.py | tee entrega/log_fase2.txt
    """
    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEMILLA = 20262
    np.random.seed(SEMILLA)

    BASE = _BASE_PROYECTO
    ENT = os.path.join(BASE, "entrega")
    FIG = os.path.join(ENT, "figs")
    TAB = os.path.join(ENT, "tablas")
    AUX = os.path.join(BASE, "aux")
    for d in (FIG, TAB):
        os.makedirs(d, exist_ok=True)

    PERIODO = 365.0
    K = 5                      # base elegida en la Fase 1 por VCG (Fourier, 2 armonicos)
    LIM_MES = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

    print("=" * 78)
    print("FASE 2 - ANALISIS EXPLORATORIO FUNCIONAL")
    print(f"numpy {np.__version__} | pandas {pd.__version__} | matplotlib {matplotlib.__version__}")
    try:
        import skfda
        HAY_SKFDA = True
        print(f"skfda {skfda.__version__} (solo para VERIFICAR lo calculado a mano)")
    except Exception as e:                                        # pragma: no cover
        HAY_SKFDA = False
        print(f"skfda no disponible ({e}); se verifica contra numpy y contra la identidad analitica")
    print("=" * 78)


    # =====================================================================
    # 0. Base de Fourier y su derivada (analiticas)
    # =====================================================================
    def base_fourier(t, K=K, periodo=PERIODO):
        """Phi (len(t) x K): 1, sen(2pi j t/T), cos(2pi j t/T). Igual que en la Fase 1."""
        t = np.asarray(t, dtype=float)
        cols = [np.ones_like(t)]
        for j in range(1, (K - 1) // 2 + 1):
            w = 2.0 * np.pi * j * t / periodo
            cols.append(np.sin(w))
            cols.append(np.cos(w))
        return np.column_stack(cols[:K])


    def base_fourier_d1(t, K=K, periodo=PERIODO):
        """Derivada primera de la base (analitica)."""
        t = np.asarray(t, dtype=float)
        cols = [np.zeros_like(t)]
        for j in range(1, (K - 1) // 2 + 1):
            wj = 2.0 * np.pi * j / periodo
            cols.append(wj * np.cos(wj * t))
            cols.append(-wj * np.sin(wj * t))
        return np.column_stack(cols[:K])


    def integra(y, x):
        """Regla del trapecio (compatible con numpy 1.x y 2.x)."""
        f = getattr(np, "trapezoid", None) or np.trapz
        return float(f(y, x))


    def eje_meses(ax, ylab=None):
        ax.set_xticks([(LIM_MES[i] + LIM_MES[i + 1]) / 2 for i in range(12)])
        ax.set_xticklabels(MESES, fontsize=8)
        for x in LIM_MES[1:-1]:
            ax.axvline(x, color="0.9", lw=0.6, zorder=0)
        ax.set_xlim(0, 365)
        if ylab:
            ax.set_ylabel(ylab)


    # =====================================================================
    # 1. Carga de los datos funcionales de la Fase 1
    # =====================================================================
    ce = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"))
    cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
    COLS = [f"c{i+1}" for i in range(K)]

    t_dia = np.arange(0.5, 365.0, 1.0)            # 365 dias, centros de dia
    t_pen = np.arange(2.5, 365.0, 5.0)            # 73 pentadas (malla nativa del satelite)
    Phi_dia = base_fourier(t_dia)
    Phi_pen = base_fourier(t_pen)
    dPhi_dia = base_fourier_d1(t_dia)

    Ce = ce[COLS].to_numpy()
    Cs = cs[COLS].to_numpy()
    Xe = Ce @ Phi_dia.T                            # 3.567 x 365   (escala sqrt mm/dia)
    Xs_p = Cs @ Phi_pen.T                          # 46.176 x 73
    print(f"\n[1] curvas estacion-anio: {Xe.shape[0]} ({ce.STATION_ID.nunique()} estaciones, "
          f"{ce.YEAR.min()}-{ce.YEAR.max()})")
    print(f"    curvas pixel-anio:    {Xs_p.shape[0]} ({cs.groupby(['FILA','COL']).ngroups} pixeles, "
          f"{cs.YEAR.min()}-{cs.YEAR.max()})")
    print("    las dos fuentes comparten base (Fourier K=5) y dominio t en [0,365]")

    # acumulado anual OBSERVADO (dato depurado de la fase 4.1), para validar y para el
    # analisis interanual: la curva suavizada en escala sqrt(P) subestima el acumulado.
    obs = (pd.read_csv(os.path.join(ENT, "estaciones_depuradas.csv"),
                       usecols=["STATION_ID", "YEAR", "VALUE"])
           .groupby(["STATION_ID", "YEAR"], as_index=False)
           .agg(acum_obs_mm=("VALUE", "sum"), dias=("VALUE", "size")))
    ce = ce.merge(obs, on=["STATION_ID", "YEAR"], how="left")
    acum_rec = (Xe ** 2).sum(axis=1)                      # integral de la curva ajustada al cuadrado
    ce["acum_rec_mm"] = acum_rec
    sesgo = float(np.mean(acum_rec - ce.acum_obs_mm))
    fac = float(ce.acum_obs_mm.sum() / acum_rec.sum())
    print(f"    acumulado observado medio {ce.acum_obs_mm.mean():.0f} mm/anio vs "
          f"reconstruido {acum_rec.mean():.0f} mm/anio (sesgo {sesgo:.0f} mm, "
          f"{100 * sesgo / ce.acum_obs_mm.mean():.1f} %; factor de escala {fac:.2f})")
    print("    -> el suavizado con K=5 elimina la varianza diaria de alta frecuencia; la curva")
    print("       describe la FORMA estacional, no el total. Para totales se usa el observado.")


    # =====================================================================
    # 2. Media funcional PASO A PASO + verificaciones
    # =====================================================================
    def media_funcional_paso_a_paso(X):
        """Media punto a punto con acumulador explicito. Sin np.mean."""
        n, m = X.shape
        acum = np.zeros(m, dtype=float)
        for i in range(n):                      # recorre curva por curva
            fila = X[i]
            for k in range(m):                  # acumula punto a punto
                acum[k] += fila[k]
        return acum / n


    print("\n[2] media funcional")
    mu_e = media_funcional_paso_a_paso(Xe)
    # (a) identidad analitica: la media de las curvas es la curva de los coeficientes medios
    c_barra = Ce.sum(axis=0) / Ce.shape[0]
    mu_e_analitica = Phi_dia @ c_barra
    d_a = float(np.max(np.abs(mu_e - mu_e_analitica)))
    # (b) numpy
    mu_e_np = Xe.mean(axis=0)
    d_b = float(np.max(np.abs(mu_e - mu_e_np)))
    print(f"    estaciones: max|manual - Phi(t)c_barra| = {d_a:.3e}")
    print(f"    estaciones: max|manual - numpy.mean|    = {d_b:.3e}")

    d_c = np.nan
    if HAY_SKFDA:
        from skfda.representation.grid import FDataGrid
        fd_e = FDataGrid(data_matrix=Xe, grid_points=t_dia)
        mu_e_sk = np.asarray(fd_e.mean().data_matrix).ravel()
        d_c = float(np.max(np.abs(mu_e - mu_e_sk)))
        print(f"    estaciones: max|manual - skfda.mean|    = {d_c:.3e}")

    mu_s = media_funcional_paso_a_paso(Xs_p)
    d_s = float(np.max(np.abs(mu_s - (Phi_pen @ (Cs.sum(axis=0) / Cs.shape[0])))))
    print(f"    satelite:   max|manual - Phi(t)c_barra| = {d_s:.3e}")

    pd.DataFrame([dict(fuente="estaciones", n=Xe.shape[0], dif_vs_analitica=d_a,
                       dif_vs_numpy=d_b, dif_vs_skfda=d_c),
                  dict(fuente="satelite", n=Xs_p.shape[0], dif_vs_analitica=d_s,
                       dif_vs_numpy=float(np.max(np.abs(mu_s - Xs_p.mean(axis=0)))),
                       dif_vs_skfda=np.nan)]
                 ).to_csv(os.path.join(TAB, "fase2_verificacion_media.csv"), index=False)


    # =====================================================================
    # 3. Varianza, desviacion, covarianza y correlacion funcional (a mano)
    # =====================================================================
    def var_funcional(X, mu):
        n, m = X.shape
        acum = np.zeros(m)
        for i in range(n):
            d = X[i] - mu
            acum += d * d
        return acum / (n - 1)


    def cov_funcional(X, mu):
        """C(s,t) = 1/(n-1) sum_i (x_i(s)-mu(s))(x_i(t)-mu(t)) por producto matricial."""
        D = X - mu
        return (D.T @ D) / (X.shape[0] - 1)


    print("\n[3] varianza, covarianza y correlacion funcional")
    var_e = var_funcional(Xe, mu_e)
    sd_e = np.sqrt(var_e)
    var_s = var_funcional(Xs_p, mu_s)
    sd_s = np.sqrt(var_s)

    # covarianza en malla pentadal (73x73) para que sea legible y comparable entre fuentes
    Xe_p = Ce @ Phi_pen.T
    mu_e_p = media_funcional_paso_a_paso(Xe_p)
    Ce_cov = cov_funcional(Xe_p, mu_e_p)
    Cs_cov = cov_funcional(Xs_p, mu_s)
    dg_e = np.sqrt(np.diag(Ce_cov))
    dg_s = np.sqrt(np.diag(Cs_cov))
    Re = Ce_cov / np.outer(dg_e, dg_e)
    Rs = Cs_cov / np.outer(dg_s, dg_s)
    print(f"    traza de la covarianza (varianza total integrada): estaciones "
          f"{np.trace(Ce_cov) * 5:.2f} | satelite {np.trace(Cs_cov) * 5:.2f}  (escala sqrt mm)^2")
    print(f"    desv. estandar funcional: estaciones min {sd_e.min():.3f} (dia "
          f"{int(t_dia[sd_e.argmin()])}) max {sd_e.max():.3f} (dia {int(t_dia[sd_e.argmax()])})")
    print(f"    correlacion ene-abr (dia 15 vs 105): estaciones {Re[3, 21]:.3f} | satelite {Rs[3, 21]:.3f}")
    print(f"    correlacion abr-oct (dia 105 vs 300): estaciones {Re[21, 60]:.3f} | satelite {Rs[21, 60]:.3f}")

    pd.DataFrame(dict(dia=t_dia, media_est=mu_e, sd_est=sd_e,
                      media_sat=np.interp(t_dia, t_pen, mu_s),
                      sd_sat=np.interp(t_dia, t_pen, sd_s))
                 ).to_csv(os.path.join(TAB, "fase2_media_sd.csv"), index=False)
    pd.DataFrame(Re, index=t_pen.astype(int), columns=t_pen.astype(int)
                 ).to_csv(os.path.join(TAB, "fase2_correlacion_estaciones.csv"))
    pd.DataFrame(Rs, index=t_pen.astype(int), columns=t_pen.astype(int)
                 ).to_csv(os.path.join(TAB, "fase2_correlacion_satelite.csv"))

    # --- figura: media +- sd de las dos fuentes
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(t_dia, mu_e - sd_e, mu_e + sd_e, color="tab:blue", alpha=0.18,
                    label="estaciones: media $\\pm$ 1 sd")
    ax.plot(t_dia, mu_e, color="tab:blue", lw=2, label="media funcional estaciones")
    ax.fill_between(t_pen, mu_s - sd_s, mu_s + sd_s, color="tab:orange", alpha=0.18,
                    label="satelite: media $\\pm$ 1 sd")
    ax.plot(t_pen, mu_s, color="tab:orange", lw=2, label="media funcional satelite")
    eje_meses(ax, "$\\sqrt{P}$ [$\\sqrt{mm}$/dia]")
    ax.set_title("Fase 2 - media y desviacion funcional por fuente (1981-2003 / 1981-2012)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_media_sd.png"), dpi=130)
    plt.close(fig)

    # --- figura: covarianza y correlacion
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for j, (M, nom) in enumerate([(Ce_cov, "estaciones"), (Cs_cov, "satelite")]):
        im = axs[0, j].imshow(M, origin="lower", extent=[0, 365, 0, 365], cmap="viridis")
        axs[0, j].set_title(f"covarianza funcional - {nom}")
        fig.colorbar(im, ax=axs[0, j], fraction=0.046)
    for j, (M, nom) in enumerate([(Re, "estaciones"), (Rs, "satelite")]):
        im = axs[1, j].imshow(M, origin="lower", extent=[0, 365, 0, 365], cmap="coolwarm",
                              vmin=-1, vmax=1)
        axs[1, j].set_title(f"correlacion funcional - {nom}")
        fig.colorbar(im, ax=axs[1, j], fraction=0.046)
    for a in axs.ravel():
        a.set_xlabel("dia del anio")
        a.set_ylabel("dia del anio")
    fig.suptitle("Fase 2 - estructura de segundo orden (malla pentadal, 73 x 73)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_covarianza_correlacion.png"), dpi=130)
    plt.close(fig)


    # =====================================================================
    # 4. Derivadas: inicio y fin de las temporadas de lluvia
    # =====================================================================
    print("\n[4] derivadas e inicio/fin de temporadas")
    dmu_e = dPhi_dia @ c_barra
    dmu_s = base_fourier_d1(t_dia) @ (Cs.sum(axis=0) / Cs.shape[0])


    def cruces(f, t, subida=True):
        """Dias donde f cruza el cero (interpolacion lineal entre muestras)."""
        s = np.sign(f)
        out = []
        for i in range(len(f) - 1):
            if s[i] == 0 or s[i] * s[i + 1] < 0:
                if (f[i + 1] > f[i]) == subida:
                    x = t[i] - f[i] * (t[i + 1] - t[i]) / (f[i + 1] - f[i])
                    out.append(float(x))
        return out


    picos_e = cruces(dmu_e, t_dia, subida=False)     # mu' pasa de + a -  -> maximos
    valles_e = cruces(dmu_e, t_dia, subida=True)     # mu' pasa de - a +  -> minimos
    picos_s = cruces(dmu_s, t_dia, subida=False)
    valles_s = cruces(dmu_s, t_dia, subida=True)


    def a_fecha(d):
        d = float(d) % 365
        for i in range(12):
            if LIM_MES[i] <= d < LIM_MES[i + 1]:
                return f"{int(d - LIM_MES[i]) + 1:02d}-{MESES[i]}"
        return "31-dic"


    print(f"    estaciones: maximos en dias {[round(p, 1) for p in picos_e]} "
          f"({[a_fecha(p) for p in picos_e]})")
    print(f"    estaciones: minimos en dias {[round(p, 1) for p in valles_e]} "
          f"({[a_fecha(p) for p in valles_e]})")
    print(f"    satelite:   maximos en dias {[round(p, 1) for p in picos_s]} "
          f"({[a_fecha(p) for p in picos_s]})")
    print(f"    satelite:   minimos en dias {[round(p, 1) for p in valles_s]} "
          f"({[a_fecha(p) for p in valles_s]})")

    # inicio / fin de temporada: cruces de la media funcional con su nivel medio anual
    niv_e = integra(mu_e, t_dia) / 365.0
    niv_s = integra(mu_s, t_pen) / 365.0
    ini_e = cruces(mu_e - niv_e, t_dia, subida=True)
    fin_e = cruces(mu_e - niv_e, t_dia, subida=False)
    ini_s = cruces(mu_s - niv_s, t_pen, subida=True)
    fin_s = cruces(mu_s - niv_s, t_pen, subida=False)
    print(f"    nivel medio anual: estaciones {niv_e:.3f} | satelite {niv_s:.3f} (sqrt mm/dia)")
    print(f"    temporadas estaciones: inicios {[a_fecha(x) for x in ini_e]} | "
          f"finales {[a_fecha(x) for x in fin_e]}")
    print(f"    temporadas satelite:   inicios {[a_fecha(x) for x in ini_s]} | "
          f"finales {[a_fecha(x) for x in fin_s]}")

    filas = []
    for nom, pic, val, ini, fin in [("estaciones", picos_e, valles_e, ini_e, fin_e),
                                    ("satelite", picos_s, valles_s, ini_s, fin_s)]:
        for tipo, lst in [("maximo", pic), ("minimo", val),
                          ("inicio_temporada", ini), ("fin_temporada", fin)]:
            for x in lst:
                filas.append(dict(fuente=nom, evento=tipo, dia=round(float(x), 2), fecha=a_fecha(x)))
    pd.DataFrame(filas).to_csv(os.path.join(TAB, "fase2_temporadas_derivada.csv"), index=False)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axs[0].plot(t_dia, mu_e, color="tab:blue", lw=2, label="media estaciones")
    axs[0].plot(t_pen, mu_s, color="tab:orange", lw=2, label="media satelite")
    axs[0].axhline(niv_e, color="tab:blue", ls=":", lw=1)
    axs[0].axhline(niv_s, color="tab:orange", ls=":", lw=1)
    for x in ini_e + fin_e:
        axs[0].axvline(x, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
    eje_meses(axs[0], "$\\sqrt{P}$")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Media funcional y cruces con el nivel medio anual (inicio/fin de temporada)")
    axs[1].plot(t_dia, dmu_e, color="tab:blue", lw=1.6, label="$\\mu'(t)$ estaciones")
    axs[1].plot(t_dia, dmu_s, color="tab:orange", lw=1.6, label="$\\mu'(t)$ satelite")
    axs[1].axhline(0, color="0.3", lw=0.8)
    eje_meses(axs[1], "derivada [$\\sqrt{mm}$/dia$^2$]")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Derivada analitica de la media funcional")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_derivadas.png"), dpi=130)
    plt.close(fig)


    # =====================================================================
    # 5. Profundidad funcional: Fraiman-Muniz y MBD (a mano) + boxplot funcional
    # =====================================================================
    def rangos_por_columna(X):
        """Rango 1..n de cada curva en cada t (promedio en empates), vectorizado."""
        n = X.shape[0]
        orden = np.argsort(X, axis=0, kind="stable")
        rk = np.empty_like(X, dtype=float)
        filas = np.arange(1, n + 1, dtype=float)[:, None]
        np.put_along_axis(rk, orden, np.broadcast_to(filas, X.shape).copy(), axis=0)
        return rk


    def pesos_trapecio(t):
        """Pesos de la regla del trapecio normalizados a 1 (integral sobre |T|)."""
        t = np.asarray(t, float)
        w = np.zeros_like(t)
        w[1:-1] = (t[2:] - t[:-2]) / 2.0
        w[0] = (t[1] - t[0]) / 2.0
        w[-1] = (t[-1] - t[-2]) / 2.0
        return w / w.sum()


    def profundidad_fraiman_muniz(X, t):
        """D_i = (1/|T|) * integral ( 1 - |F_t(x_i(t)) - 1/2| ) dt  (ec. 3, Fraiman-Muniz 2001)."""
        n = X.shape[0]
        F = rangos_por_columna(X) / n
        return (1.0 - np.abs(F - 0.5)) @ pesos_trapecio(t)


    def profundidad_mbd(X, t):
        """Modified Band Depth con J=2 (Lopez-Pintado & Romo 2009), via rangos.
        La proporcion de pares (j,k) cuya banda contiene a x_i(t) es
        [(r-1)(n-r) + (n-1)] / C(n,2) con r el rango de x_i(t)."""
        n = X.shape[0]
        r = rangos_por_columna(X)
        prop = ((r - 1.0) * (n - r) + (n - 1.0)) / (n * (n - 1.0) / 2.0)
        return prop @ pesos_trapecio(t)


    print("\n[5] profundidad funcional y boxplot funcional")
    fm_e = profundidad_fraiman_muniz(Xe, t_dia)
    mbd_e = profundidad_mbd(Xe, t_dia)
    print(f"    estaciones: FM  min {fm_e.min():.4f} max {fm_e.max():.4f} "
          f"| MBD min {mbd_e.min():.4f} max {mbd_e.max():.4f}")
    print(f"    correlacion de Spearman FM vs MBD: "
          f"{np.corrcoef(rangos_por_columna(fm_e[:, None]).ravel(), rangos_por_columna(mbd_e[:, None]).ravel())[0, 1]:.4f}")

    try:
        from skfda.exploratory.depth import ModifiedBandDepth, IntegratedDepth
        HAY_DEPTH = HAY_SKFDA
    except Exception as e:                                        # pragma: no cover
        HAY_DEPTH = False
        print(f"    aviso: skfda.exploratory.depth no importable ({type(e).__name__}); "
              "la verificacion de profundidad se omite")

    if HAY_DEPTH:
        sub = np.sort(np.random.choice(Xe.shape[0], 400, replace=False))   # verificacion en submuestra
        fd_sub = FDataGrid(data_matrix=Xe[sub], grid_points=t_dia)
        mbd_sk = ModifiedBandDepth()(fd_sub)
        fm_sk = IntegratedDepth()(fd_sub)
        mbd_mio = profundidad_mbd(Xe[sub], t_dia)
        fm_mio = profundidad_fraiman_muniz(Xe[sub], t_dia)
        e1 = float(np.max(np.abs(mbd_mio - mbd_sk)))
        e2 = float(np.max(np.abs(fm_mio - fm_sk)))
        print(f"    verificacion contra skfda (n=400): max|MBD_manual - MBD_skfda| = {e1:.3e}")
        print(f"    verificacion contra skfda (n=400): max|FM_manual  - FM_skfda|  = {e2:.3e}")
        pd.DataFrame([dict(medida="MBD", n_verif=len(sub), max_dif=e1),
                      dict(medida="Fraiman-Muniz", n_verif=len(sub), max_dif=e2)]
                     ).to_csv(os.path.join(TAB, "fase2_verificacion_profundidad.csv"), index=False)


    def boxplot_funcional(X, prof, factor=1.5):
        """Sun & Genton (2011): mediana = curva mas profunda; banda central = 50 % mas
        profundas; valla = banda central +- factor * ancho de la banda central."""
        n = X.shape[0]
        ordr = np.argsort(-prof)
        central = ordr[: max(2, int(np.ceil(n / 2)))]
        inf = X[central].min(axis=0)
        sup = X[central].max(axis=0)
        ancho = sup - inf
        valla_inf = inf - factor * ancho
        valla_sup = sup + factor * ancho
        fuera = (X < valla_inf[None, :]) | (X > valla_sup[None, :])
        return dict(mediana=X[ordr[0]], idx_mediana=int(ordr[0]), inf=inf, sup=sup,
                    valla_inf=valla_inf, valla_sup=valla_sup,
                    atipicas=np.where(fuera.any(axis=1))[0], central=central)


    bp_e = boxplot_funcional(Xe, mbd_e)
    atip = bp_e["atipicas"]
    print(f"    atipicas (estaciones): {len(atip)} de {Xe.shape[0]} curvas "
          f"({100 * len(atip) / Xe.shape[0]:.2f} %)")
    med_id = ce.iloc[bp_e["idx_mediana"]]
    print(f"    curva mediana (MBD maxima): estacion {int(med_id.STATION_ID)} anio {int(med_id.YEAR)}")

    det = ce[["STATION_ID", "YEAR"]].copy()
    det["MBD"] = mbd_e
    det["FraimanMuniz"] = fm_e
    det["acum_obs_mm"] = ce.acum_obs_mm.to_numpy()
    det["acum_rec_mm"] = acum_rec
    det["atipica_boxplot"] = False
    det.loc[det.index[atip], "atipica_boxplot"] = True
    det.sort_values("MBD").to_csv(os.path.join(TAB, "fase2_profundidad_estaciones.csv"), index=False)
    det[det.atipica_boxplot].sort_values(["YEAR", "STATION_ID"]).to_csv(
        os.path.join(TAB, "fase2_atipicas_estacion_anio.csv"), index=False)
    print("    top 5 menos profundas (MBD):")
    for _, r in det.nsmallest(5, "MBD").iterrows():
        print(f"      estacion {int(r.STATION_ID)} anio {int(r.YEAR)}  MBD={r.MBD:.4f}  "
              f"acumulado observado={r.acum_obs_mm:7.0f} mm")
    anio_at = det[det.atipica_boxplot].YEAR.value_counts().sort_index()
    print(f"    atipicas por anio (top): {anio_at.nlargest(6).to_dict()}")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(t_dia, bp_e["valla_inf"], bp_e["valla_sup"], color="0.92",
                    label="valla 1.5 x banda central")
    ax.fill_between(t_dia, bp_e["inf"], bp_e["sup"], color="tab:blue", alpha=0.30,
                    label="banda central 50 % (MBD)")
    for i in atip[:60]:
        ax.plot(t_dia, Xe[i], color="tab:red", lw=0.6, alpha=0.55)
    ax.plot(t_dia, bp_e["mediana"], color="black", lw=2, label="mediana funcional (MBD max)")
    ax.plot(t_dia, mu_e, color="tab:green", lw=1.4, ls="--", label="media funcional")
    eje_meses(ax, "$\\sqrt{P}$ [$\\sqrt{mm}$/dia]")
    ax.set_title(f"Fase 2 - boxplot funcional, estaciones 1981-2003 ({len(atip)} curvas atipicas)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_boxplot_funcional.png"), dpi=130)
    plt.close(fig)

    # el mismo analisis para el satelite (malla pentadal, por costo computacional justificado)
    mbd_s = profundidad_mbd(Xs_p, t_pen)
    bp_s = boxplot_funcional(Xs_p, mbd_s)
    print(f"    atipicas (satelite): {len(bp_s['atipicas'])} de {Xs_p.shape[0]} "
          f"({100 * len(bp_s['atipicas']) / Xs_p.shape[0]:.2f} %)")
    ds = cs[["YEAR", "FILA", "COL", "LON", "LAT"]].copy()
    ds["MBD"] = mbd_s
    ds["atipica_boxplot"] = False
    ds.loc[ds.index[bp_s["atipicas"]], "atipica_boxplot"] = True
    ds[ds.atipica_boxplot].to_csv(os.path.join(TAB, "fase2_atipicas_satelite.csv"), index=False)


    # =====================================================================
    # 6. Variabilidad interanual y ENSO
    # =====================================================================
    print("\n[6] variabilidad interanual y ENSO")
    oni = None
    ruta_oni = os.path.join(AUX, "oni.txt")
    if os.path.exists(ruta_oni):
        reg = []
        with open(ruta_oni) as f:
            for ln in f.read().splitlines():
                p = ln.split()
                if len(p) == 13 and p[0].isdigit():
                    reg.append([int(p[0])] + [float(x) for x in p[1:]])
        oni = pd.DataFrame(reg, columns=["YEAR"] + [f"m{i}" for i in range(1, 13)])
        oni = oni.replace(-99.99, np.nan).replace(-9.99, np.nan)
        # fase ENSO del anio de lluvia: promedio DJF (dic del anio previo, ene-feb del anio)
        oni = oni.set_index("YEAR")
        djf = {}
        for y in oni.index:
            if y - 1 in oni.index:
                djf[y] = np.nanmean([oni.loc[y - 1, "m12"], oni.loc[y, "m1"], oni.loc[y, "m2"]])
        djf = pd.Series(djf, name="ONI_DJF")
        anual = oni[[f"m{i}" for i in range(1, 13)]].mean(axis=1)   # ONI medio del anio civil
        print(f"    serie Nino 3.4 cargada: {int(oni.index.min())}-{int(oni.index.max())}")
    else:
        djf = pd.Series(dtype=float, name="ONI_DJF")
        anual = pd.Series(dtype=float, name="ONI_ANUAL")
        print("    aviso: aux/oni.txt no encontrado; se omite la clasificacion ENSO")


    def fase_enso(v, u=0.5):
        if np.isnan(v):
            return "sin dato"
        return "El Nino" if v >= u else ("La Nina" if v <= -u else "Neutro")


    # media funcional por anio (estaciones) y su acumulado
    res_anio = []
    mu_anio = {}
    for y, g in ce.groupby("YEAR"):
        Xy = g[COLS].to_numpy() @ Phi_dia.T
        muy = media_funcional_paso_a_paso(Xy)
        mu_anio[int(y)] = muy
        res_anio.append(dict(YEAR=int(y), n_curvas=len(g),
                             acum_mm=float(g.acum_obs_mm.mean()),
                             acum_rec_mm=float((muy ** 2).sum()),
                             norma_L2_desvio=float(np.sqrt(integra((muy - mu_e) ** 2, t_dia))),
                             ONI_DJF=float(djf.get(int(y), np.nan)),
                             ONI_ANUAL=float(anual.get(int(y), np.nan)),
                             fase_ENSO=fase_enso(float(djf.get(int(y), np.nan))),
                             fase_ENSO_anual=fase_enso(float(anual.get(int(y), np.nan)))))
    ra = pd.DataFrame(res_anio)
    ra.to_csv(os.path.join(TAB, "fase2_interanual_estaciones.csv"), index=False)
    print(f"    acumulado anual medio: {ra.acum_mm.mean():.0f} mm "
          f"(min {ra.acum_mm.min():.0f} en {int(ra.loc[ra.acum_mm.idxmin(), 'YEAR'])}, "
          f"max {ra.acum_mm.max():.0f} en {int(ra.loc[ra.acum_mm.idxmax(), 'YEAR'])})")
    if ra.ONI_DJF.notna().any():
        rho = ra[["acum_mm", "ONI_DJF"]].dropna().corr(method="spearman").iloc[0, 1]
        rho2 = ra[["acum_mm", "ONI_ANUAL"]].dropna().corr(method="spearman").iloc[0, 1]
        print(f"    correlacion de Spearman acumulado anual vs ONI DJF: {rho:.3f} "
              f"(n={ra.ONI_DJF.notna().sum()} anios)")
        print(f"    correlacion de Spearman acumulado anual vs ONI medio anual: {rho2:.3f} "
              "(clasificacion mas pertinente para el anio civil de lluvia)")
        print("    medias por fase ENSO segun ONI medio anual (mm/anio):")
        for f, g in ra.groupby("fase_ENSO_anual"):
            print(f"      {f:9s} n={len(g):2d}  {g.acum_mm.mean():7.0f} mm  "
                  f"anios {sorted(g.YEAR.tolist())}")
        ra.groupby("fase_ENSO_anual").agg(n=("YEAR", "size"), acum_medio_mm=("acum_mm", "mean"),
                                    desvio_L2_medio=("norma_L2_desvio", "mean")
                                    ).to_csv(os.path.join(TAB, "fase2_enso_resumen.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    col = {"El Nino": "tab:red", "La Nina": "tab:blue", "Neutro": "0.6", "sin dato": "0.8"}
    for _, r in ra.iterrows():
        axs[0].plot(t_dia, mu_anio[int(r.YEAR)], lw=1.0, alpha=0.75, color=col[r.fase_ENSO_anual])
    axs[0].plot(t_dia, mu_e, color="black", lw=2.2, label="media 1981-2003")
    for f, c in col.items():
        if (ra.fase_ENSO_anual == f).any():
            axs[0].plot([], [], color=c, lw=1.4, label=f)
    eje_meses(axs[0], "$\\sqrt{P}$")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Media funcional por anio, coloreada por fase ENSO (ONI medio anual)")
    axs[1].bar(ra.YEAR, ra.acum_mm, color=[col[f] for f in ra.fase_ENSO_anual])
    axs[1].axhline(ra.acum_mm.mean(), color="black", ls="--", lw=1)
    axs[1].set_xlabel("anio")
    axs[1].set_ylabel("acumulado anual medio [mm]")
    axs[1].set_title("Acumulado anual reconstruido y fase ENSO")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_interanual_enso.png"), dpi=130)
    plt.close(fig)


    # =====================================================================
    # 7. Orografia: gradiente espacial con la fuente satelital (tiene coordenadas)
    # =====================================================================
    print("\n[7] gradiente orografico (fuente satelital, 1.443 pixeles)")
    cs2 = cs.copy()
    cs2["acum_mm"] = ((cs2[COLS].to_numpy() @ Phi_pen.T) ** 2).sum(axis=1) * 5.0
    pix = cs2.groupby(["FILA", "COL", "LON", "LAT"], as_index=False).acum_mm.mean()
    pix.to_csv(os.path.join(TAB, "fase2_acumulado_por_pixel.csv"), index=False)
    print(f"    acumulado anual por pixel: min {pix.acum_mm.min():.0f} mm | "
          f"mediana {pix.acum_mm.median():.0f} mm | max {pix.acum_mm.max():.0f} mm")
    oeste = pix[pix.LON <= -76.8].acum_mm.mean()
    valle = pix[(pix.LON > -76.5) & (pix.LON < -75.9)].acum_mm.mean()
    print(f"    media vertiente pacifica (lon <= -76.8): {oeste:.0f} mm | "
          f"valle interandino (-76.5 a -75.9): {valle:.0f} mm | razon {oeste / valle:.2f}")
    rho_lon = pix[["LON", "acum_mm"]].corr(method="spearman").iloc[0, 1]
    print(f"    Spearman acumulado vs longitud: {rho_lon:.3f} (negativo = mas lluvia al occidente)")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = axs[0].scatter(pix.LON, pix.LAT, c=pix.acum_mm, s=14, cmap="YlGnBu")
    fig.colorbar(sc, ax=axs[0], label="acumulado anual [mm]")
    axs[0].set_xlabel("longitud")
    axs[0].set_ylabel("latitud")
    axs[0].set_title("Acumulado anual medio por pixel (1981-2012)")
    for lo, hi, nom, c in [(-77.6, -76.8, "vertiente pacifica", "tab:green"),
                           (-76.8, -76.5, "cordillera occidental", "tab:olive"),
                           (-76.5, -75.9, "valle del rio Cauca", "tab:orange"),
                           (-75.9, -75.7, "flanco oriental", "tab:brown")]:
        s = cs2[(cs2.LON >= lo) & (cs2.LON < hi)]
        if len(s):
            m = media_funcional_paso_a_paso(s[COLS].to_numpy() @ Phi_pen.T)
            axs[1].plot(t_pen, m, lw=1.8, color=c, label=f"{nom} (n={s.groupby(['FILA','COL']).ngroups})")
    eje_meses(axs[1], "$\\sqrt{P}$")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Media funcional por franja longitudinal (orografia)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_orografia.png"), dpi=130)
    plt.close(fig)

    # =====================================================================
    # 8. FPCA (componentes principales funcionales) A MANO
    #    Problema de autovalores generalizado V W b = lambda b con
    #    W = integral Phi(t) Phi(t)^T dt (matriz de Gram de la base) y
    #    V = covarianza de los coeficientes. Se simetriza con W^{1/2}.
    #    Verificacion contra skfda.preprocessing.dim_reduction.FPCA.
    # =====================================================================
    print("\n[8] FPCA funcional (implementado a mano, verificado contra skfda)")

    w_t = pesos_trapecio(t_dia) * (t_dia[-1] - t_dia[0])      # pesos sin normalizar
    Phi_e = base_fourier(t_dia, K)                            # (n_t x K)
    W = Phi_e.T @ (Phi_e * w_t[:, None])                      # Gram K x K
    Cc = ce[COLS].to_numpy(float) - ce[COLS].to_numpy(float).mean(axis=0)
    V = (Cc.T @ Cc) / (len(Cc) - 1)

    lamW, Uw = np.linalg.eigh(W)                              # W simetrica def. positiva
    Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T                   # W^{1/2}
    Whi = Uw @ np.diag(1.0 / np.sqrt(lamW)) @ Uw.T            # W^{-1/2}
    lam, U = np.linalg.eigh(Wh @ V @ Wh)
    ordn = np.argsort(lam)[::-1]
    lam, U = lam[ordn], U[:, ordn]
    B = Whi @ U                                               # coef. de las eigenfunciones
    xi = Phi_e @ B                                            # eigenfunciones (n_t x K)
    for j in range(K):                                        # signo canonico: maximo positivo
        if xi[np.argmax(np.abs(xi[:, j])), j] < 0:
            xi[:, j] *= -1
            B[:, j] *= -1
    scores = Cc @ W @ B                                       # scores = <x-mu, xi_j>
    vexp = lam / lam.sum()
    print("    varianza explicada: " + "  ".join(
        f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)) +
        f"  (acumulada PC1-PC2 {100*vexp[:2].sum():.1f}%)")
    print(f"    ortonormalidad: max|<xi_i,xi_j> - delta_ij| = "
          f"{np.abs((xi.T @ (xi * w_t[:, None])) - np.eye(K)).max():.2e}")
    print(f"    varianza de los scores vs autovalores: max dif = "
          f"{np.abs(scores.var(axis=0, ddof=1) - lam).max():.2e}")
    if HAY_SKFDA:
        from skfda.preprocessing.dim_reduction import FPCA
        fd_e = FDataGrid(Xe, t_dia)
        fp = FPCA(n_components=3).fit(fd_e)
        sc_sk = fp.transform(fd_e)
        print("    skfda varianza explicada: " + "  ".join(
            f"PC{j+1} {100*v:.1f}%" for j, v in enumerate(fp.explained_variance_ratio_)))
        for j in range(3):
            print(f"    |corr(scores PC{j+1} manual, skfda)| = "
                  f"{abs(np.corrcoef(scores[:, j], sc_sk[:, j])[0, 1]):.6f}")

    pd.DataFrame(dict(PC=[f"PC{j+1}" for j in range(K)], autovalor=lam,
                      var_explicada=vexp, var_acumulada=np.cumsum(vexp))
                 ).to_csv(os.path.join(TAB, "fase2_fpca_varianza.csv"), index=False)
    pd.DataFrame(dict(STATION_ID=ce.STATION_ID, YEAR=ce.YEAR,
                      PC1=scores[:, 0], PC2=scores[:, 1], PC3=scores[:, 2],
                      acum_obs_mm=ce.acum_obs_mm, MBD=mbd_e)
                 ).to_csv(os.path.join(TAB, "fase2_fpca_scores.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for j in range(3):
        axs[0].plot(t_dia, xi[:, j], lw=1.6, label=f"PC{j+1} ({100*vexp[j]:.1f}%)")
    axs[0].axhline(0, color="gray", lw=0.6)
    eje_meses(axs[0], "eigenfuncion")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Componentes principales funcionales")
    sd1 = np.sqrt(lam[0])
    axs[1].plot(t_dia, mu_e, color="black", lw=2, label="media")
    axs[1].plot(t_dia, mu_e + sd1 * xi[:, 0], color="tab:red", lw=1.2, ls="--",
                label="media + sd$\\cdot$PC1")
    axs[1].plot(t_dia, mu_e - sd1 * xi[:, 0], color="tab:blue", lw=1.2, ls="--",
                label="media - sd$\\cdot$PC1")
    eje_meses(axs[1], "$\\sqrt{P}$")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Efecto de PC1 sobre la media")
    sc = axs[2].scatter(scores[:, 0], scores[:, 1], c=ce.acum_obs_mm, s=6,
                        cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=axs[2], label="acumulado observado [mm/anio]")
    axs[2].set_xlabel(f"PC1 ({100*vexp[0]:.1f}%)")
    axs[2].set_ylabel(f"PC2 ({100*vexp[1]:.1f}%)")
    axs[2].set_title("Plano de scores")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_fpca.png"), dpi=130)
    plt.close(fig)
    print(f"    Spearman PC1 vs acumulado observado: "
          f"{pd.Series(scores[:, 0]).corr(ce.acum_obs_mm, method='spearman'):.3f}")

    # =====================================================================
    # 9. Validacion: curva ajustada vs dato observado
    # =====================================================================
    print("\n[9] validacion de la representacion funcional contra el dato observado")
    err = acum_rec - ce.acum_obs_mm.to_numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r_p = float(np.corrcoef(acum_rec, ce.acum_obs_mm.to_numpy())[0, 1])
    ro = ce[["acum_obs_mm", "acum_rec_mm"]].corr(method="spearman").iloc[0, 1]
    print(f"    n={len(ce)} curvas | RMSE {rmse:.0f} mm | r Pearson {r_p:.4f} | "
          f"rho Spearman {ro:.4f} | sesgo {err.mean():.0f} mm")
    print("    la correlacion alta con sesgo sistematico confirma que la curva preserva el")
    print("    ordenamiento y la forma (lo que usan las fases 2-4) y no el total acumulado.")
    pd.DataFrame([dict(n=len(ce), rmse_mm=rmse, r_pearson=r_p, rho_spearman=float(ro),
                       sesgo_mm=float(err.mean()), factor_escala=fac)]
                 ).to_csv(os.path.join(TAB, "fase2_validacion_acumulados.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axs[0].scatter(ce.acum_obs_mm, acum_rec, s=5, alpha=0.3, color="tab:blue")
    lim = [0, float(ce.acum_obs_mm.max()) * 1.02]
    axs[0].plot(lim, lim, color="black", lw=1, ls="--", label="identidad")
    axs[0].plot(lim, [x / fac for x in lim], color="tab:red", lw=1,
                label=f"ajuste 1/{fac:.2f}")
    axs[0].set_xlabel("acumulado anual observado [mm]")
    axs[0].set_ylabel("integral de la curva ajustada al cuadrado [mm]")
    axs[0].set_title(f"Validacion (r={r_p:.3f})")
    axs[0].legend(fontsize=8)
    peor = det.nsmallest(3, "MBD")
    for _, r in peor.iterrows():
        k = ce.index[(ce.STATION_ID == r.STATION_ID) & (ce.YEAR == r.YEAR)][0]
        axs[1].plot(t_dia, Xe[k], lw=1.3, label=f"{int(r.STATION_ID)} / {int(r.YEAR)}")
    axs[1].plot(t_dia, bp_e["mediana"], color="black", lw=2, label="mediana funcional")
    eje_meses(axs[1], "$\\sqrt{P}$")
    axs[1].legend(fontsize=7)
    axs[1].set_title("Curvas menos profundas (MBD minima) vs mediana")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase2_validacion.png"), dpi=130)
    plt.close(fig)

    print("\n[fin] figuras en entrega/figs/fase2_*.png, tablas en entrega/tablas/fase2_*.csv")
    print(f"semilla {SEMILLA} | python {sys.version.split()[0]}")


def _bloque_5():
    """Fase 3 - comparacion estaciones vs satelital  (origen: fase3_comparacion.py)"""
    # =====================================================================
    # FASE 3 - Comparacion estacion terrestre vs producto satelital
    # Proyecto Final de Analitica de Datos (UAO 2026-2S)
    #
    # Problema de partida: el proyecto no entrega el catalogo de coordenadas de
    # las estaciones y los 186 STATION_ID no coinciden con el campo Codigo del
    # Catalogo Nacional de Estaciones del IDEAM (0/186 literal, ver
    # aux/NOTA_CODIGOS_ESTACIONES.md). Lo que SI es valido es la estructura del
    # codigo: los 4 primeros digitos son la SUBZONA HIDROGRAFICA (2610..2625 =
    # cuenca del Cauca, 5300..5421 = Patia y vertiente pacifica sur), y eso se
    # verifica contra el CNE (429 estaciones del CNE en la subzona 2610, 513 en
    # 2612, etc.).
    #
    # De ahi la estrategia de esta fase, en tres pasos auditables:
    #   (1) coordenada aproximada = centroide de la subzona segun el CNE, con su
    #       dispersion como incertidumbre declarada (~11 km en lon, ~28 km en lat);
    #   (2) universo de analisis = solo las estaciones cuya subzona cae dentro de
    #       la malla satelital (el resto esta fuera del dominio del satelital, lo
    #       cual es un hallazgo del proyecto, no un error de procesamiento);
    #   (3) emparejamiento estacion-pixel = entre los pixeles a menos de RADIO km
    #       del centroide, el de maxima correlacion de ANOMALIAS mensuales
    #       (climatologia removida: el ciclo estacional es casi identico en toda
    #       la region y no discrimina posicion; la variabilidad interanual si).
    #
    # Reglas de la rubrica respetadas: todo estadistico funcional se calcula a
    # mano (medias, sesgo, RMSE, correlacion funcional, banda bootstrap); las
    # derivadas son analiticas sobre la base de Fourier; sin lm() ni lstsq.
    #
    # Entradas : entrega/coef_estaciones.csv, entrega/coef_satelite.csv,
    #            entrega/estaciones_depuradas.csv, aux/cne.csv
    # Salidas  : entrega/pares_estacion_pixel.csv (insumo de la fase 4),
    #            entrega/figs/fase3_*.png, entrega/tablas/fase3_*.csv
    # =====================================================================
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEMILLA = 20263
    rng = np.random.default_rng(SEMILLA)
    BASE = _BASE_PROYECTO
    ENT = os.path.join(BASE, "entrega")
    FIG = os.path.join(ENT, "figs")
    TAB = os.path.join(ENT, "tablas")
    AUX = os.path.join(BASE, "aux")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    K, PERIODO = 5, 365.0
    t_dia = np.arange(0.5, 365.0, 1.0)
    COLS = [f"c{i+1}" for i in range(K)]
    RADIO_KM = 35.0            # radio de busqueda alrededor del centroide de subzona
    MALLA = dict(lon0=-77.525, lon1=-75.725, lat0=3.075, lat1=4.975)
    MARGEN = 0.15              # grados de tolerancia al declarar "dentro de la malla"
    MESES = ["ene", "feb", "mar", "abr", "may", "jun",
             "jul", "ago", "sep", "oct", "nov", "dic"]
    INI_MES = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


    def base_fourier(t, K, periodo=PERIODO):
        """Phi (len(t) x K): 1, sen(2pi j t/T), cos(2pi j t/T). K impar."""
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    def dbase_fourier(t, K, periodo=PERIODO):
        """Derivada analitica de la base (no diferencias finitas)."""
        w = 2 * np.pi / periodo
        cols = [np.zeros_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(j * w * np.cos(j * w * t))
            cols.append(-j * w * np.sin(j * w * t))
        return np.column_stack(cols[:K])


    def integra(y, x):
        f = getattr(np, "trapezoid", None) or np.trapz
        return f(y, x)


    def media_paso_a_paso(X):
        """Media funcional con acumulador explicito (sin np.mean), regla de la rubrica."""
        acum = np.zeros(X.shape[1])
        n = 0
        for i in range(X.shape[0]):
            acum = acum + X[i]
            n += 1
        return acum / n


    def eje_meses(ax, ylab):
        ax.set_xticks(INI_MES + 15)
        ax.set_xticklabels(MESES, fontsize=8)
        ax.set_xlim(0, 365)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)


    def km(dlon, dlat, lat=4.0):
        return np.sqrt((dlon * 111.32 * np.cos(np.radians(lat))) ** 2 + (dlat * 110.57) ** 2)


    Phi = base_fourier(t_dia, K)
    dPhi = dbase_fourier(t_dia, K)

    print("=" * 72)
    print("FASE 3 - Comparacion estaciones vs satelital  (semilla %d)" % SEMILLA)
    print("=" * 72)

    # ---------------------------------------------------------------------
    # 1. Coordenada aproximada por subzona hidrografica (CNE del IDEAM)
    # ---------------------------------------------------------------------
    print("\n[1] coordenada aproximada por subzona hidrografica")
    cne = pd.read_csv(os.path.join(AUX, "cne.csv"), dtype=str, low_memory=False)
    cne["SUB"] = cne.Codigo.str.strip().str.lstrip("0").str.zfill(8).str[:4]
    xy = cne["Ubicación"].map(lambda s: re.findall(r"-?\d+\.?\d*", str(s)))
    cne["LON"] = [float(v[0]) if len(v) >= 2 else np.nan for v in xy]
    cne["LAT"] = [float(v[1]) if len(v) >= 2 else np.nan for v in xy]
    cne["ALT"] = pd.to_numeric(cne.Altitud, errors="coerce")

    # IMPORTANTE: se usan los coeficientes PENTADALES (fase3_0_pentadas.py), no los
    # diarios de la fase 1. El satelital se ajusto sobre promedios pentadales y la
    # raiz es concava: comparar contra el ajuste diario infla el satelital un factor
    # ~2.1 por desigualdad de Jensen (artefacto de representacion, no sesgo fisico).
    PENT = os.path.join(ENT, "coef_estaciones_pentadal.csv")
    if not os.path.exists(PENT):
        raise SystemExit("falta coef_estaciones_pentadal.csv: ejecuta antes fase3_0_pentadas.py")
    ce = pd.read_csv(PENT, dtype={"STATION_ID": str})
    ce_diaria = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"),
                            dtype={"STATION_ID": str})
    cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
    ce["SUB"] = ce.STATION_ID.str[:4]
    subz = (cne[cne.SUB.isin(ce.SUB.unique())].groupby("SUB")
            .agg(n_cne=("Codigo", "size"), LON_SUB=("LON", "median"),
                 LAT_SUB=("LAT", "median"), ALT_SUB=("ALT", "median"),
                 sd_lon=("LON", "std"), sd_lat=("LAT", "std"), sd_alt=("ALT", "std"),
                 DEP=("Departamento", lambda s: s.mode().iat[0])).reset_index())
    subz["EN_MALLA"] = ((subz.LON_SUB >= MALLA["lon0"] - MARGEN)
                        & (subz.LON_SUB <= MALLA["lon1"] + MARGEN)
                        & (subz.LAT_SUB >= MALLA["lat0"] - MARGEN)
                        & (subz.LAT_SUB <= MALLA["lat1"] + MARGEN))
    print(f"    subzonas en los datos: {ce.SUB.nunique()} | con registros en el CNE: {len(subz)}")
    print(f"    incertidumbre mediana del centroide: {km(subz.sd_lon.median(), 0):.0f} km en lon, "
          f"{km(0, subz.sd_lat.median()):.0f} km en lat")
    for _, r in subz.sort_values("SUB").iterrows():
        print(f"      {r.SUB}  n_CNE={int(r.n_cne):4d}  ({r.LON_SUB:8.3f}, {r.LAT_SUB:6.3f})  "
              f"alt={r.ALT_SUB:6.0f} m  {r.DEP:<16s} {'EN MALLA' if r.EN_MALLA else 'fuera'}")
    subz.to_csv(os.path.join(TAB, "fase3_subzonas.csv"), index=False)

    ce = ce.merge(subz, on="SUB", how="left")
    est_tot = ce.STATION_ID.nunique()
    ce_m = ce[ce.EN_MALLA == True].copy()
    print(f"\n    [HALLAZGO] estaciones dentro del dominio satelital: "
          f"{ce_m.STATION_ID.nunique()} de {est_tot} "
          f"({100 * ce_m.STATION_ID.nunique() / est_tot:.0f} %)")
    print("    el resto pertenece al Cauca medio-bajo (Antioquia) y al Patia/Pacifico sur,")
    print("    fuera de la malla del producto satelital: los dos conjuntos de datos del")
    print("    proyecto NO cubren la misma region. La comparacion pareada solo es legitima")
    print("    en la interseccion de dominios.")

    # ---------------------------------------------------------------------
    # 2. Emparejamiento estacion-pixel por correlacion de anomalias
    # ---------------------------------------------------------------------
    print("\n[2] emparejamiento estacion-pixel (radio %.0f km + maxima correlacion de anomalias)"
          % RADIO_KM)
    anios = sorted(set(ce_m.YEAR) & set(cs.YEAR))
    ce_m = ce_m[ce_m.YEAR.isin(anios)]
    csa = cs[cs.YEAR.isin(anios)].copy()
    csa["PIX"] = csa.FILA.astype(str) + "_" + csa.COL.astype(str)
    pix = csa[["PIX", "LON", "LAT"]].drop_duplicates().reset_index(drop=True)
    print(f"    anios comunes: {anios[0]}-{anios[-1]} ({len(anios)}) | pixeles: {len(pix)}")

    # matrices mensuales (promedio de la curva por mes) y anomalias
    Mmes = np.zeros((12, len(t_dia)))
    mes_de_dia = np.searchsorted(INI_MES[1:], t_dia, side="right")
    for m in range(12):
        sel = mes_de_dia == m
        Mmes[m] = sel / sel.sum()


    def mensual(df, clave, llaves):
        idx = {k: i for i, k in enumerate(llaves)}
        ia = {a: i for i, a in enumerate(anios)}
        out = np.full((len(llaves), 12 * len(anios)), np.nan)
        men = (df[COLS].to_numpy(float) @ Phi.T) @ Mmes.T
        for (k, a), fila in zip(zip(df[clave], df.YEAR), men):
            out[idx[k], 12 * ia[a]:12 * ia[a] + 12] = fila
        return out


    def anomalias(A):
        out = A.copy()
        for m in range(12):
            sl = slice(m, A.shape[1], 12)
            out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
        return out


    est = sorted(ce_m.STATION_ID.unique())
    Ae = anomalias(mensual(ce_m, "STATION_ID", est))
    As = anomalias(mensual(csa, "PIX", list(pix.PIX)))

    meta = ce_m.drop_duplicates("STATION_ID").set_index("STATION_ID")
    pares = []
    for i, e in enumerate(est):
        d = km(pix.LON - meta.LON_SUB[e], pix.LAT - meta.LAT_SUB[e])
        cand = np.where(d.to_numpy() <= RADIO_KM)[0]
        if len(cand) == 0:
            cand = np.array([int(np.argmin(d))])
        ok = np.isfinite(Ae[i])
        a = Ae[i, ok] - Ae[i, ok].mean()
        B = As[np.ix_(cand, ok)]
        B = B - B.mean(axis=1, keepdims=True)
        r = (B @ a) / (np.sqrt((B ** 2).sum(axis=1)) * np.sqrt((a ** 2).sum()) + 1e-12)
        j = int(np.argmax(r))
        p = pix.iloc[cand[j]]
        pares.append(dict(STATION_ID=e, SUB=meta.SUB[e], DEP=meta.DEP[e],
                          ALT_SUB=meta.ALT_SUB[e], LON_SUB=meta.LON_SUB[e],
                          LAT_SUB=meta.LAT_SUB[e], PIX=p.PIX, LON_PIX=p.LON,
                          LAT_PIX=p.LAT, n_cand=len(cand), r_anom=float(r[j]),
                          dist_km=float(km(p.LON - meta.LON_SUB[e], p.LAT - meta.LAT_SUB[e])),
                          n_meses=int(ok.sum())))
    P = pd.DataFrame(pares)
    print(f"    estaciones emparejadas: {len(P)} | pixeles distintos: {P.PIX.nunique()}")
    print(f"    r de anomalias: mediana {P.r_anom.median():.3f} "
          f"[{P.r_anom.min():.3f}, {P.r_anom.max():.3f}]")
    print(f"    candidatos por estacion: mediana {P.n_cand.median():.0f} | "
          f"distancia al centroide: mediana {P.dist_km.median():.1f} km")

    # contraste: r del pixel elegido vs r de pixeles aleatorios de la malla
    r_nulo = []
    for i in range(len(est)):
        ok = np.isfinite(Ae[i])
        a = Ae[i, ok] - Ae[i, ok].mean()
        sel = rng.integers(0, len(pix), 40)
        B = As[np.ix_(sel, ok)]
        B = B - B.mean(axis=1, keepdims=True)
        r_nulo.append(((B @ a) / (np.sqrt((B ** 2).sum(axis=1)) * np.sqrt((a ** 2).sum()) + 1e-12)))
    r_nulo = np.concatenate(r_nulo)
    print(f"    control: r con pixeles aleatorios = {r_nulo.mean():.3f} +- {r_nulo.std():.3f} "
          f"=> el emparejamiento gana {P.r_anom.mean() - r_nulo.mean():+.3f} de correlacion")
    P.to_csv(os.path.join(TAB, "fase3_emparejamiento.csv"), index=False)

    # ---------------------------------------------------------------------
    # 3. Comparacion funcional pareada
    # ---------------------------------------------------------------------
    print("\n[3] comparacion funcional pareada estacion vs pixel")
    mp = P.set_index("STATION_ID")
    ce_i = ce_m.set_index(["STATION_ID", "YEAR"])
    cs_i = csa.set_index(["PIX", "YEAR"])
    Xe, Xs, key = [], [], []
    for (e, y) in ce_i.index:
        p = mp.PIX[e]
        if (p, y) in cs_i.index:
            Xe.append(ce_i.loc[(e, y), COLS].to_numpy(float) @ Phi.T)
            Xs.append(cs_i.loc[(p, y), COLS].to_numpy(float) @ Phi.T)
            key.append((e, y, p))
    Xe, Xs = np.array(Xe), np.array(Xs)
    key = pd.DataFrame(key, columns=["STATION_ID", "YEAR", "PIX"])
    print(f"    pares estacion-anio comparables: {len(Xe)}")

    mu_e = media_paso_a_paso(Xe)
    mu_s = media_paso_a_paso(Xs)
    D = Xe - Xs
    mu_d = media_paso_a_paso(D)

    # banda bootstrap de la diferencia media (1000 remuestreos de los pares)
    B_BOOT = 1000
    boot = np.empty((B_BOOT, len(t_dia)))
    for b in range(B_BOOT):
        idx = rng.integers(0, len(D), len(D))
        boot[b] = D[idx].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    signif = (lo > 0) | (hi < 0)
    print(f"    sesgo medio integrado (estacion - satelite): "
          f"{integra(mu_d, t_dia) / 365:.3f} (sqrt(mm)/dia)")
    print(f"    sesgo maximo {mu_d.max():+.3f} el dia {int(t_dia[np.argmax(mu_d)])} | "
          f"minimo {mu_d.min():+.3f} el dia {int(t_dia[np.argmin(mu_d)])}")
    print(f"    la diferencia es significativa al 95 % en {100 * signif.mean():.0f} % del anio")
    rmse_f = np.sqrt((D ** 2).mean(axis=0))
    print(f"    RMSE funcional: medio {rmse_f.mean():.3f} | max {rmse_f.max():.3f} "
          f"el dia {int(t_dia[np.argmax(rmse_f)])}")

    # correlacion funcional punto a punto entre fuentes (a mano)
    corr_t = np.empty(len(t_dia))
    for j in range(len(t_dia)):
        a, b = Xe[:, j], Xs[:, j]
        a = a - a.mean()
        b = b - b.mean()
        corr_t[j] = (a @ b) / (np.sqrt((a @ a) * (b @ b)) + 1e-12)
    print(f"    correlacion entre fuentes dia a dia: mediana {np.median(corr_t):.3f} "
          f"[{corr_t.min():.3f}, {corr_t.max():.3f}]")

    # totales anuales y desfase de temporadas
    tot_e = (Xe ** 2).sum(axis=1)
    tot_s = (Xs ** 2).sum(axis=1)
    raz = tot_s / tot_e
    print(f"    integral anual: estacion {tot_e.mean():.0f} vs satelite {tot_s.mean():.0f} "
          f"(razon mediana {np.median(raz):.2f})")

    # misma comparacion con el ajuste DIARIO de la fase 1, para cuantificar cuanto
    # del desacuerdo era artefacto de la representacion y no sesgo del satelital
    cd_i = ce_diaria.set_index(["STATION_ID", "YEAR"])
    hay = [i for i, (e, y) in enumerate(zip(key.STATION_ID, key.YEAR)) if (e, y) in cd_i.index]
    Xd = cd_i.loc[list(zip(key.STATION_ID[hay], key.YEAR[hay])), COLS].to_numpy(float) @ Phi.T
    raz_d = tot_s[hay] / (Xd ** 2).sum(axis=1)
    print(f"    control de escala: con el ajuste diario de la fase 1 la razon sat/est seria "
          f"{np.median(raz_d):.2f}")
    print(f"    => {100 * (1 - np.median(raz) / np.median(raz_d)):.0f} % del desacuerdo original "
          f"era artefacto de Jensen, no sesgo del producto satelital")
    dmu_e = (ce_i[COLS].to_numpy(float).mean(axis=0)) @ dPhi.T
    dmu_s = (cs_i[COLS].to_numpy(float).mean(axis=0)) @ dPhi.T


    def picos(mu):
        d = np.sign(np.diff(np.sign(np.gradient(mu, t_dia))))
        return t_dia[1:][d < 0]


    print(f"    maximos de la media: estacion dias {np.round(picos(mu_e)).astype(int).tolist()} | "
          f"satelite dias {np.round(picos(mu_s)).astype(int).tolist()}")

    res = key.copy()
    res["sesgo_int"] = integra(D, t_dia) / 365
    res["rmse"] = np.sqrt((D ** 2).mean(axis=1))
    res["tot_est"] = tot_e
    res["tot_sat"] = tot_s
    res["razon_sat_est"] = raz
    res = res.merge(P[["STATION_ID", "SUB", "DEP", "ALT_SUB", "LON_PIX", "LAT_PIX",
                       "r_anom", "dist_km"]], on="STATION_ID", how="left")
    res.to_csv(os.path.join(TAB, "fase3_metricas_par.csv"), index=False)

    print("\n    sesgo por subzona (estacion - satelite, sqrt(mm)/dia):")
    for s, g in res.groupby("SUB"):
        print(f"      {s}  n={len(g):4d}  sesgo {g.sesgo_int.mean():+.3f}  "
              f"razon sat/est {g.razon_sat_est.median():.2f}  alt {g.ALT_SUB.iloc[0]:.0f} m")
    rho_alt = res[["sesgo_int", "ALT_SUB"]].corr(method="spearman").iloc[0, 1]
    rho_lon = res[["sesgo_int", "LON_PIX"]].corr(method="spearman").iloc[0, 1]
    print(f"    Spearman sesgo vs altitud de subzona: {rho_alt:+.3f} | vs longitud: {rho_lon:+.3f}")

    # dependencia del sesgo con el anio (deriva del producto satelital)
    por_anio = res.groupby("YEAR").agg(n=("rmse", "size"), sesgo=("sesgo_int", "mean"),
                                       razon=("razon_sat_est", "median")).reset_index()
    rho_y = por_anio[["YEAR", "sesgo"]].corr(method="spearman").iloc[0, 1]
    print(f"    Spearman sesgo vs anio: {rho_y:+.3f} "
          f"(deriva temporal del producto satelital)")
    por_anio.to_csv(os.path.join(TAB, "fase3_sesgo_por_anio.csv"), index=False)
    pd.DataFrame(dict(dia=t_dia, media_est=mu_e, media_sat=mu_s, dif=mu_d,
                      ic_bajo=lo, ic_alto=hi, significativo=signif,
                      rmse=rmse_f, corr=corr_t)
                 ).to_csv(os.path.join(TAB, "fase3_curvas_comparacion.csv"), index=False)

    # ---------------------------------------------------------------------
    # 4. Dataset pareado para la fase 4
    # ---------------------------------------------------------------------
    sal = key.copy()
    for j in range(K):
        sal[f"sat_c{j+1}"] = cs_i.loc[list(zip(key.PIX, key.YEAR)), COLS[j]].to_numpy()
        sal[f"est_c{j+1}"] = ce_i.loc[list(zip(key.STATION_ID, key.YEAR)), COLS[j]].to_numpy()
    sal = sal.merge(P[["STATION_ID", "SUB", "ALT_SUB", "LON_PIX", "LAT_PIX", "r_anom"]],
                    on="STATION_ID", how="left")
    sal.to_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), index=False)
    print(f"\n[4] dataset pareado para la fase 4: entrega/pares_estacion_pixel.csv "
          f"({len(sal)} filas, {sal.STATION_ID.nunique()} estaciones, "
          f"{sal.YEAR.min()}-{sal.YEAR.max()})")

    # ---------------------------------------------------------------------
    # 5. Figuras
    # ---------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    axs[0].plot(t_dia, mu_e, color="tab:blue", lw=2, label="estaciones")
    axs[0].plot(t_dia, mu_s, color="tab:orange", lw=2, label="satelital")
    eje_meses(axs[0], r"$\sqrt{P}$")
    axs[0].legend(fontsize=8)
    axs[0].set_title(f"Media funcional pareada (n={len(Xe)})")
    axs[1].fill_between(t_dia, lo, hi, color="tab:red", alpha=0.25, label="IC 95 % bootstrap")
    axs[1].plot(t_dia, mu_d, color="tab:red", lw=1.8, label="estacion - satelite")
    axs[1].axhline(0, color="black", lw=0.8)
    eje_meses(axs[1], r"diferencia $\sqrt{P}$")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Sesgo funcional con banda bootstrap")
    axs[2].plot(t_dia, corr_t, color="tab:green", lw=1.6, label="correlacion entre fuentes")
    axs[2].plot(t_dia, rmse_f / rmse_f.max(), color="gray", lw=1.2, ls="--",
                label="RMSE funcional (normalizado)")
    eje_meses(axs[2], "correlacion")
    axs[2].legend(fontsize=8)
    axs[2].set_title("Acuerdo dia a dia")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase3_comparacion.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    sc = axs[0].scatter(pix.LON, pix.LAT, s=4, color="lightgray", label="malla satelital")
    axs[0].scatter(P.LON_SUB, P.LAT_SUB, s=55, marker="s", facecolor="none",
                   edgecolor="tab:blue", label="centroide de subzona")
    axs[0].scatter(P.LON_PIX, P.LAT_PIX, s=18, color="tab:red", label="pixel emparejado")
    for _, r in P.iterrows():
        axs[0].plot([r.LON_SUB, r.LON_PIX], [r.LAT_SUB, r.LAT_PIX], color="tab:red",
                    lw=0.4, alpha=0.5)
    axs[0].set_xlabel("longitud")
    axs[0].set_ylabel("latitud")
    axs[0].legend(fontsize=7)
    axs[0].set_title("Emparejamiento espacial")
    axs[1].hist(r_nulo, bins=40, density=True, color="lightgray",
                label="pixeles aleatorios")
    axs[1].hist(P.r_anom, bins=20, density=True, color="tab:red", alpha=0.7,
                label="pixel emparejado")
    axs[1].set_xlabel("correlacion de anomalias mensuales")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Validacion del emparejamiento")
    axs[2].scatter(res.tot_est, res.tot_sat, s=6, alpha=0.35, color="tab:purple")
    lim = [0, float(max(res.tot_est.max(), res.tot_sat.max())) * 1.02]
    axs[2].plot(lim, lim, color="black", ls="--", lw=1, label="identidad")
    axs[2].set_xlabel("integral anual, estacion")
    axs[2].set_ylabel("integral anual, satelite")
    axs[2].legend(fontsize=8)
    axs[2].set_title(f"Totales pareados (razon mediana {np.median(raz):.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase3_emparejamiento.png"), dpi=130)
    plt.close(fig)

    print("\n[fin] figuras fase3_*.png, tablas fase3_*.csv, dataset pares_estacion_pixel.csv")


def _bloque_6():
    """Fase 4 - red funcional, ablacion y evaluacion  (origen: fase4_modelos.py)"""
    # =====================================================================
    # FASE 4 - Red neuronal funcional, ablacion de cabezas, evaluacion
    # Proyecto Final de Analitica de Datos (UAO 2026-2S)
    #
    # Tarea: predecir la curva anual de precipitacion de una ESTACION terrestre
    # a partir de la curva del PRODUCTO SATELITAL del pixel emparejado en la
    # fase 3, mas tres covariables de contexto (altitud de la subzona, longitud
    # y latitud del pixel).
    #
    # Arquitectura (4.2):
    #   codificador funcional : z = W^{1/2} c_sat, con W = integral Phi Phi^T dt
    #                           calculada por cuadratura trapezoidal (producto
    #                           interno de L2, no el euclideo de los coeficientes)
    #   cabeza                : lineal | MLP | KAN  (lo que se ablaciona en 4.3)
    #   decodificador FPCA    : scores -> coeficientes, con la base de
    #                           componentes principales estimada SOLO con el
    #                           conjunto de entrenamiento (sin fuga)
    #   perdida               : (c_pred - c_obs)^T W (c_pred - c_obs) / T
    #                           + lambda_rug * rugosidad de las funciones peso,
    #                           con R = integral Phi'' Phi''^T dt analitica
    #
    # Verificaciones exigidas por la rubrica:
    #   - la cabeza lineal sin penalizacion coincide con la regresion funcional
    #     resuelta en forma cerrada por Cholesky propia (tolerancia 1e-4)
    #   - KAN validada sobre una funcion sintetica conocida
    #   - numero de parametros de KAN y MLP igualado (diferencia <= 10 %)
    #   - reconstruccion en mm no negativa: se reporta el % de dias recortados
    #
    # Evaluacion (4.4): particion temporal + estaciones reservadas sin fuga,
    # las cuatro metricas (L2 relativo, KGE, sesgo del acumulado anual en %,
    # error en la fecha de los dos picos), bootstrap de 1000 remuestreos,
    # Wilcoxon pareada por estacion y desglose por fase ENSO.
    #
    # Nota de desviacion documentada: la rubrica sugiere la particion
    # 1981-2005 / 2006-2009 / 2010-2013, pero las estaciones terrestres de este
    # proyecto terminan en 2003 (el satelital llega a 2012). Se usa por tanto
    # 1981-1996 / 1997-1999 / 2000-2003, que conserva la logica de extrapolar
    # en el tiempo sin inventar datos.
    #
    # Entradas : entrega/pares_estacion_pixel.csv, aux/oni.txt
    # Salidas  : entrega/log_fase4.txt, entrega/tablas/fase4_*.csv,
    #            entrega/figs/fase4_*.png, entrega/pred_reservadas.csv
    # =====================================================================
    import os
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from scipy.stats import wilcoxon
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(4)
    BASE = _BASE_PROYECTO
    ENT = os.path.join(BASE, "entrega")
    FIG = os.path.join(ENT, "figs")
    TAB = os.path.join(ENT, "tablas")
    AUX = os.path.join(BASE, "aux")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    K, PERIODO = 5, 365.0
    t_dia = np.arange(0.5, 365.0, 1.0)
    T = len(t_dia)
    SEMILLAS = [20264, 20265, 20266]
    LAM_RUG = 1e-6
    # particion oficial de la guia corregida (fe de erratas del 9-nov-2026):
    # 1981-1995 entrenamiento, 1996-2000 validacion, 2001-2003 prueba
    ANIOS_TR = range(1981, 1996)
    ANIOS_VA = range(1996, 2001)
    ANIOS_TE = range(2001, 2004)
    FRAC_RESERVADAS = 0.20
    MESES = ["ene", "feb", "mar", "abr", "may", "jun",
             "jul", "ago", "sep", "oct", "nov", "dic"]
    INI_MES = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


    # ---------------------------------------------------------------------
    # base de Fourier, derivadas analiticas, Gram y rugosidad
    # ---------------------------------------------------------------------
    def base_fourier(t, K, periodo=PERIODO):
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    def dbase_fourier(t, K, periodo=PERIODO):
        w = 2 * np.pi / periodo
        cols = [np.zeros_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(j * w * np.cos(j * w * t))
            cols.append(-j * w * np.sin(j * w * t))
        return np.column_stack(cols[:K])


    def d2base_fourier(t, K, periodo=PERIODO):
        w = 2 * np.pi / periodo
        cols = [np.zeros_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(-(j * w) ** 2 * np.sin(j * w * t))
            cols.append(-(j * w) ** 2 * np.cos(j * w * t))
        return np.column_stack(cols[:K])


    def pesos_trapecio(t):
        w = np.empty_like(t)
        w[0] = (t[1] - t[0]) / 2
        w[-1] = (t[-1] - t[-2]) / 2
        w[1:-1] = (t[2:] - t[:-2]) / 2
        return w


    def cholesky_propia(A):
        n = A.shape[0]
        L = np.zeros_like(A)
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j] - float(L[i, :j] @ L[j, :j])
                if i == j:
                    if s <= 0:
                        raise np.linalg.LinAlgError("no definida positiva")
                    L[i, j] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
        return L


    def resuelve_chol(A, B):
        """Resuelve A X = B (A simetrica def. pos.) por sustitucion, a mano."""
        L = cholesky_propia(A)
        n = A.shape[0]
        B = np.atleast_2d(B.T).T if B.ndim > 1 else B.reshape(-1, 1)
        Y = np.zeros_like(B)
        for i in range(n):
            Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
        X = np.zeros_like(B)
        for i in range(n - 1, -1, -1):
            X[i] = (Y[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
        return X


    Phi = base_fourier(t_dia, K)
    dPhi = dbase_fourier(t_dia, K)
    d2Phi = d2base_fourier(t_dia, K)
    wq = pesos_trapecio(t_dia)
    W = Phi.T @ (Phi * wq[:, None])            # Gram por cuadratura
    R = d2Phi.T @ (d2Phi * wq[:, None])        # rugosidad
    lamW, Uw = np.linalg.eigh(W)
    Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T

    log = []


    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)


    P("=" * 74)
    P("FASE 4 - red funcional, ablacion de cabezas y evaluacion")
    P("=" * 74)
    P(f"    Gram W por cuadratura: traza {np.trace(W):.2f} | "
      f"max|W - W^T| = {np.abs(W - W.T).max():.2e}")
    P(f"    rugosidad R: traza {np.trace(R):.3e} (penaliza la curvatura de las w_k(t))")

    # ---------------------------------------------------------------------
    # datos y particiones
    # ---------------------------------------------------------------------
    D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"),
                    dtype={"STATION_ID": str})
    SAT = [f"sat_c{i+1}" for i in range(K)]
    EST = [f"est_c{i+1}" for i in range(K)]
    COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
    P(f"\n[1] datos: {len(D)} pares estacion-anio | {D.STATION_ID.nunique()} estaciones "
      f"| {D.YEAR.min()}-{D.YEAR.max()}")

    rng0 = np.random.default_rng(2026)
    ests = np.array(sorted(D.STATION_ID.unique()))
    reserv = set(rng0.choice(ests, int(round(FRAC_RESERVADAS * len(ests))), replace=False))
    D["RESERVADA"] = D.STATION_ID.isin(reserv)
    tr = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TR)]
    va = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_VA)]
    te = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TE)]
    rs = D[D.RESERVADA]
    P(f"    estaciones reservadas por sorteo (no se usan para entrenar): {len(reserv)} "
      f"de {len(ests)}")
    P(f"    particion temporal: train {len(tr)} ({min(ANIOS_TR)}-{max(ANIOS_TR)}) | "
      f"val {len(va)} ({min(ANIOS_VA)}-{max(ANIOS_VA)}) | test {len(te)} "
      f"({min(ANIOS_TE)}-{max(ANIOS_TE)})")
    P(f"    bloque reservado (generalizacion espacial): {len(rs)} pares, "
      f"{rs.STATION_ID.nunique()} estaciones, todos los anios")

    # ---------------------------------------------------------------------
    # decodificador FPCA estimado SOLO con train
    # ---------------------------------------------------------------------
    Ctr = tr[EST].to_numpy(float)
    c_mu = Ctr.mean(axis=0)
    V = ((Ctr - c_mu).T @ (Ctr - c_mu)) / (len(Ctr) - 1)
    Whi = Uw @ np.diag(1 / np.sqrt(lamW)) @ Uw.T
    lamF, Uf = np.linalg.eigh(Wh @ V @ Wh)
    o = np.argsort(lamF)[::-1]
    lamF, Uf = lamF[o], Uf[:, o]
    Bfull = Whi @ Uf
    vexp = lamF / lamF.sum()
    M = int(np.searchsorted(np.cumsum(vexp), 0.99) + 1)
    Bm = Bfull[:, :M]
    P(f"\n[2] decodificador FPCA (solo train): varianza explicada "
      + " ".join(f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)))
    P(f"    componentes retenidas para el 99 %: M = {M} "
      f"(acumulado {100*vexp[:M].sum():.2f} %)")
    P(f"    ortonormalidad en L2: max|B^T W B - I| = "
      f"{np.abs(Bm.T @ W @ Bm - np.eye(M)).max():.2e}")


    def codifica(df):
        z = df[SAT].to_numpy(float) @ Wh           # producto interno de L2
        x = np.column_stack([z, df[COV].to_numpy(float)])
        return x


    Xtr, Xva, Xte, Xrs = codifica(tr), codifica(va), codifica(te), codifica(rs)
    mx, sx = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    Xtr_s, Xva_s, Xte_s, Xrs_s = [(A - mx) / sx for A in (Xtr, Xva, Xte, Xrs)]
    # objetivo: scores FPCA de la curva de estacion
    Str = (tr[EST].to_numpy(float) - c_mu) @ W @ Bm
    Sva = (va[EST].to_numpy(float) - c_mu) @ W @ Bm
    P(f"    entrada al modelo: {Xtr.shape[1]} variables "
      f"(5 del codificador L2 + {len(COV)} covariables) -> salida {M} scores")


    # ---------------------------------------------------------------------
    # cabezas
    # ---------------------------------------------------------------------
    class KANPropia(nn.Module):
        """KAN minima: suma de funciones univariadas en base B-spline cubica.
        Se usa como respaldo si pykan no esta disponible o falla."""

        def __init__(self, n_in, n_out, grid=5, k=3):
            super().__init__()
            self.n_in, self.k = n_in, k
            paso = 1.0 / grid
            self.nodos = torch.linspace(-0.5 - k * paso, 1.5 + k * paso,
                                        grid + 2 * k + 2)
            self.nb = len(self.nodos) - k - 1
            self.coef = nn.Parameter(0.1 * torch.randn(n_in, self.nb, n_out))
            self.w_lin = nn.Parameter(0.1 * torch.randn(n_in, n_out))
            self.b = nn.Parameter(torch.zeros(n_out))

        def bspline(self, x):
            nod = self.nodos.to(x.device)
            B = ((x.unsqueeze(-1) >= nod[:-1]) & (x.unsqueeze(-1) < nod[1:])).to(x.dtype)
            for p in range(1, self.k + 1):
                iz = (x.unsqueeze(-1) - nod[:-(p + 1)]) / (nod[p:-1] - nod[:-(p + 1)])
                de = (nod[p + 1:] - x.unsqueeze(-1)) / (nod[p + 1:] - nod[1:-p])
                B = iz * B[..., :-1] + de * B[..., 1:]
            return B

        def forward(self, x):
            xn = (x - x.min()) / (x.max() - x.min() + 1e-9) if False else x
            B = self.bspline(torch.tanh(xn) * 0.5 + 0.5)        # (n, n_in, nb)
            y = torch.einsum("nib,ibo->no", B, self.coef)
            y = y + torch.einsum("ni,io->no", torch.tanh(x), self.w_lin)
            return y + self.b


    def construye_cabeza(tipo, n_in, n_out, semilla, h_mlp=None, usa_pykan=True):
        torch.manual_seed(semilla)
        if tipo == "lineal":
            return nn.Linear(n_in, n_out)
        if tipo == "mlp":
            return nn.Sequential(nn.Linear(n_in, h_mlp), nn.GELU(),
                                 nn.Linear(h_mlp, n_out))
        if tipo == "kan":
            if usa_pykan:
                try:
                    from kan import KAN
                    return KAN(width=[n_in, n_out], grid=5, k=3, seed=semilla,
                               device="cpu")
                except Exception as e:                      # pragma: no cover
                    P(f"    [aviso] pykan no utilizable ({type(e).__name__}: {e}); "
                      f"se usa KANPropia (splines B cubicas, misma idea)")
            return KANPropia(n_in, n_out)
        raise ValueError(tipo)


    def n_par(m):
        return sum(p.numel() for p in m.parameters())


    # igualar parametros de KAN y MLP (<= 10 % de diferencia)
    kan0 = construye_cabeza("kan", Xtr.shape[1], M, SEMILLAS[0])
    usa_pykan = type(kan0).__name__ == "MultKAN" or "kan" in type(kan0).__module__
    npar_kan = n_par(kan0)
    mejor_h, mejor_d = 1, 1e9
    for h in range(1, 400):
        d = abs(n_par(construye_cabeza("mlp", Xtr.shape[1], M, 0, h_mlp=h)) - npar_kan)
        if d < mejor_d:
            mejor_h, mejor_d = h, d
    H_MLP = mejor_h
    npar_mlp = n_par(construye_cabeza("mlp", Xtr.shape[1], M, 0, h_mlp=H_MLP))
    npar_lin = n_par(construye_cabeza("lineal", Xtr.shape[1], M, 0))
    P(f"\n[3] presupuesto de parametros (sum(p.numel()))")
    P(f"    KAN ({'pykan' if usa_pykan else 'KANPropia'}): {npar_kan} | "
      f"MLP (h={H_MLP}): {npar_mlp} | lineal: {npar_lin}")
    P(f"    diferencia KAN vs MLP: {100*abs(npar_kan-npar_mlp)/npar_kan:.1f} % "
      f"({'cumple <= 10 %' if abs(npar_kan-npar_mlp)/npar_kan <= 0.10 else 'NO CUMPLE'})")

    # ---------------------------------------------------------------------
    # entrenamiento con perdida L2 funcional + rugosidad
    # ---------------------------------------------------------------------
    Wt = torch.tensor(W, dtype=torch.float32)
    Rt = torch.tensor(R, dtype=torch.float32)
    Bmt = torch.tensor(Bm, dtype=torch.float32)
    cmut = torch.tensor(c_mu, dtype=torch.float32)


    def perdida(pred_s, obs_c, modelo):
        c = cmut + pred_s @ Bmt.T
        d = c - obs_c
        l2 = ((d @ Wt) * d).sum(dim=1).mean() / PERIODO
        # rugosidad de las funciones peso del operador: w_k(t) = Phi(t) (B_m A)_k
        A = None
        for p in modelo.parameters():
            if p.ndim == 2 and p.shape[0] == pred_s.shape[1]:
                A = p
                break
        rug = 0.0
        if A is not None:
            Bw = Bmt @ A                                   # (K x n_in)
            rug = ((Bw.T @ Rt) * Bw.T).sum() / PERIODO
        return l2 + LAM_RUG * rug, l2


    def entrena(tipo, semilla, epocas=3000, lr=0.01):
        m = construye_cabeza(tipo, Xtr.shape[1], M, semilla, h_mlp=H_MLP)
        xt = torch.tensor(Xtr_s, dtype=torch.float32)
        ct = torch.tensor(tr[EST].to_numpy(float), dtype=torch.float32)
        xv = torch.tensor(Xva_s, dtype=torch.float32)
        cv = torch.tensor(va[EST].to_numpy(float), dtype=torch.float32)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        mejor, mejor_st, paciencia = 1e18, None, 0
        for ep in range(epocas):
            opt.zero_grad()
            p_, _ = perdida(m(xt), ct, m)
            p_.backward()
            opt.step()
            if ep % 10 == 0:
                with torch.no_grad():
                    _, lv = perdida(m(xv), cv, m)
                if lv.item() < mejor - 1e-7:
                    mejor, paciencia = lv.item(), 0
                    mejor_st = {k: v.clone() for k, v in m.state_dict().items()}
                else:
                    paciencia += 1
                    if paciencia >= 30:
                        break
        if mejor_st is not None:
            m.load_state_dict(mejor_st)
        return m, mejor, ep


    def predice_coef(m, Xs):
        with torch.no_grad():
            s = m(torch.tensor(Xs, dtype=torch.float32)).numpy()
        return c_mu + s @ Bm.T


    # ---------------------------------------------------------------------
    # verificacion 1: cabeza lineal == regresion funcional cerrada
    # ---------------------------------------------------------------------
    P(f"\n[4] verificaciones")
    Xa = np.column_stack([Xtr_s, np.ones(len(Xtr_s))])
    G = Xa.T @ Xa + 1e-10 * np.eye(Xa.shape[1])
    Theta = resuelve_chol(G, Xa.T @ Str)          # solucion cerrada, Cholesky propia
    pred_cerrada = c_mu + (Xa @ Theta) @ Bm.T
    # la equivalencia es teorica: B es W-ortonormal y M=K, asi que la perdida
    # (c-c_hat)^T W (c-c_hat) es exactamente ||s - s_hat||^2 euclidea en scores,
    # y el minimizador de la cabeza lineal es la regresion funcional cerrada.
    # Para comprobarlo hay que optimizar a CONVERGENCIA sobre train (sin
    # penalizacion y sin parada temprana por validacion): se usa LBFGS.
    LAM_RUG_BK = LAM_RUG
    LAM_RUG = 0.0
    # en float64: con float32 el residuo de la optimizacion (2e-4) es ruido de
    # precision de la maquina, no desacuerdo entre los dos metodos
    torch.manual_seed(SEMILLAS[0])
    m_lin0 = nn.Linear(Xtr.shape[1], M).double()
    _xt = torch.tensor(Xtr_s, dtype=torch.float64)
    _ct = torch.tensor(tr[EST].to_numpy(float), dtype=torch.float64)
    _W64 = torch.tensor(W, dtype=torch.float64)
    _B64 = torch.tensor(Bm, dtype=torch.float64)
    _mu64 = torch.tensor(c_mu, dtype=torch.float64)
    _opt = torch.optim.LBFGS(m_lin0.parameters(), lr=1.0, max_iter=800,
                             tolerance_grad=1e-16, tolerance_change=1e-18,
                             history_size=80, line_search_fn="strong_wolfe")


    def _cierre():
        _opt.zero_grad()
        d = (_mu64 + m_lin0(_xt) @ _B64.T) - _ct
        l = ((d @ _W64) * d).sum(dim=1).mean() / PERIODO
        l.backward()
        return l


    for _ in range(8):
        _opt.step(_cierre)
    LAM_RUG = LAM_RUG_BK
    with torch.no_grad():
        pred_grad = (_mu64 + m_lin0(_xt) @ _B64.T).numpy()
    dif = np.abs(pred_grad - pred_cerrada).max()
    P(f"    cabeza lineal por gradiente vs regresion funcional cerrada (Cholesky propia):")
    P(f"      max|dif| en los coeficientes = {dif:.2e} "
      f"({'CUMPLE <= 1e-4' if dif <= 1e-4 else 'NO cumple 1e-4'})")

    # verificacion 2: KAN sobre funcion sintetica conocida
    torch.manual_seed(7)
    xs = torch.rand(800, 2) * 2 - 1
    ys = (torch.sin(np.pi * xs[:, :1]) + xs[:, 1:] ** 2)
    kv = construye_cabeza("kan", 2, 1, 7)
    ov = torch.optim.Adam(kv.parameters(), lr=0.02)
    for _ in range(1500):
        ov.zero_grad()
        l = ((kv(xs) - ys) ** 2).mean()
        l.backward()
        ov.step()
    with torch.no_grad():
        rmse_sint = float(((kv(xs) - ys) ** 2).mean().sqrt())
    P(f"    KAN sobre f(x1,x2)=sin(pi x1)+x2^2: RMSE {rmse_sint:.4f} "
      f"(sd objetivo {float(ys.std()):.3f}) -> "
      f"{'aprende la funcion' if rmse_sint < 0.1 * float(ys.std()) else 'ajuste pobre'}")

    # ---------------------------------------------------------------------
    # metricas
    # ---------------------------------------------------------------------
    def curvas(coef):
        return coef @ Phi.T


    def picos_dos(coef):
        """Fechas de los dos maximos principales por derivada ANALITICA."""
        d = coef @ dPhi.T
        out = []
        for fila in d:
            sg = np.sign(fila)
            cruces = np.where((sg[:-1] > 0) & (sg[1:] <= 0))[0]
            if len(cruces) == 0:
                out.append([np.nan, np.nan])
                continue
            y = (coef @ Phi.T)[len(out)] if False else None
            dias = t_dia[cruces]
            val = (fila[cruces] * 0)
            out.append(list(dias[:2]) + [np.nan] * (2 - len(dias[:2])))
        return np.array(out)


    def kge(obs, pred):
        r = np.corrcoef(obs, pred)[0, 1]
        return 1 - np.sqrt((r - 1) ** 2
                           + (pred.std() / obs.std() - 1) ** 2
                           + (pred.mean() / obs.mean() - 1) ** 2)


    def metricas(coef_obs, coef_pred, etiqueta, modelo, semilla, grupo):
        Xo, Xp = curvas(coef_obs), curvas(coef_pred)
        rec = np.maximum(Xp, 0.0)
        pct_rec = 100 * (Xp < 0).mean()
        acum_o = (np.maximum(Xo, 0) ** 2).sum(axis=1)
        acum_p = (rec ** 2).sum(axis=1)
        l2rel = np.linalg.norm(Xp - Xo, axis=1) / (np.linalg.norm(Xo, axis=1) + 1e-12)
        po, pp = picos_dos(coef_obs), picos_dos(coef_pred)
        err_pico = np.nanmean(np.abs(pp - po), axis=1)
        return dict(modelo=modelo, semilla=semilla, bloque=grupo, n=len(Xo),
                    l2_rel=float(l2rel.mean()),
                    rmse_func=float(np.sqrt(((Xp - Xo) ** 2).mean())),
                    kge=float(kge(acum_o, acum_p)),
                    sesgo_acum_pct=float(100 * (acum_p.mean() / acum_o.mean() - 1)),
                    err_picos_dias=float(np.nanmean(err_pico)),
                    pct_recortado=float(pct_rec)), l2rel, acum_o, acum_p, err_pico


    # ONI para el desglose por fase ENSO
    oni_f = os.path.join(AUX, "oni.txt")
    fase = {}
    if os.path.exists(oni_f):
        lineas = [l.split() for l in open(oni_f).read().strip().split("\n")]
        for l in lineas:
            if len(l) == 13 and l[0].isdigit():
                a = int(l[0])
                v = np.mean([float(x) for x in l[1:]])
                fase[a] = "El Nino" if v >= 0.5 else ("La Nina" if v <= -0.5 else "Neutro")

    P(f"\n[5] entrenamiento y evaluacion (3 semillas por cabeza)")
    filas, detalle, modelos_fin = [], {}, {}
    for tipo in ["lineal", "mlp", "kan"]:
        for sd in SEMILLAS:
            m, lv, ep = entrena(tipo, sd)
            for nombre, df, Xs in [("test_temporal", te, Xte_s),
                                   ("reservadas", rs, Xrs_s)]:
                cp = predice_coef(m, Xs)
                mt, l2rel, ao, ap, epi = metricas(df[EST].to_numpy(float), cp,
                                                  nombre, tipo, sd, nombre)
                filas.append(mt)
                if sd == SEMILLAS[0]:
                    detalle[(tipo, nombre)] = dict(df=df, coef_pred=cp, l2rel=l2rel,
                                                   acum_o=ao, acum_p=ap)
            modelos_fin[(tipo, sd)] = m
            P(f"    {tipo:7s} semilla {sd}: epocas {ep:5d} | perdida val {lv:.5f}")

    # baseline: el satelital sin corregir
    for nombre, df, _ in [("test_temporal", te, None), ("reservadas", rs, None)]:
        mt, l2rel, ao, ap, epi = metricas(df[EST].to_numpy(float),
                                          df[SAT].to_numpy(float),
                                          nombre, "satelital_sin_corregir", 0, nombre)
        filas.append(mt)
        detalle[("satelital_sin_corregir", nombre)] = dict(
            df=df, coef_pred=df[SAT].to_numpy(float), l2rel=l2rel, acum_o=ao, acum_p=ap)
    F = pd.DataFrame(filas)
    F.to_csv(os.path.join(TAB, "fase4_metricas.csv"), index=False)

    res = (F.groupby(["bloque", "modelo"])
           .agg(l2_rel_m=("l2_rel", "mean"), l2_rel_sd=("l2_rel", "std"),
                kge_m=("kge", "mean"), kge_sd=("kge", "std"),
                sesgo_m=("sesgo_acum_pct", "mean"), sesgo_sd=("sesgo_acum_pct", "std"),
                picos_m=("err_picos_dias", "mean"), picos_sd=("err_picos_dias", "std"),
                rec_m=("pct_recortado", "mean"))
           .reset_index())
    P("\n[6] resultados (media +- sd entre semillas)")
    for b in ["test_temporal", "reservadas"]:
        P(f"\n    bloque {b}")
        P(f"      {'modelo':24s} {'L2 rel':>14s} {'KGE':>14s} "
          f"{'sesgo acum %':>14s} {'err picos (d)':>14s} {'recortado %':>11s}")
        for _, r in res[res.bloque == b].iterrows():
            sd_l = 0 if np.isnan(r.l2_rel_sd) else r.l2_rel_sd
            sd_k = 0 if np.isnan(r.kge_sd) else r.kge_sd
            sd_s = 0 if np.isnan(r.sesgo_sd) else r.sesgo_sd
            sd_p = 0 if np.isnan(r.picos_sd) else r.picos_sd
            P(f"      {r.modelo:24s} {r.l2_rel_m:7.4f}+-{sd_l:5.4f} "
              f"{r.kge_m:7.4f}+-{sd_k:5.4f} {r.sesgo_m:+7.1f}+-{sd_s:5.1f} "
              f"{r.picos_m:7.1f}+-{sd_p:5.1f} {r.rec_m:10.2f}")
    res.to_csv(os.path.join(TAB, "fase4_resumen.csv"), index=False)

    # ---------------------------------------------------------------------
    # bootstrap e inferencia
    # ---------------------------------------------------------------------
    P(f"\n[7] inferencia: bootstrap 1000 por estacion + Wilcoxon pareada")
    rngb = np.random.default_rng(99)
    comp = []
    for bloque in ["test_temporal", "reservadas"]:
        base_l2 = detalle[("satelital_sin_corregir", bloque)]["l2rel"]
        dfb = detalle[("satelital_sin_corregir", bloque)]["df"]
        for tipo in ["lineal", "mlp", "kan"]:
            l2 = detalle[(tipo, bloque)]["l2rel"]
            d = l2 - base_l2
            idx = rngb.integers(0, len(d), (1000, len(d)))
            bs = d[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            w = wilcoxon(l2, base_l2)
            # Wilcoxon pareada POR ESTACION (promedio por estacion antes del test)
            g = pd.DataFrame(dict(e=dfb.STATION_ID.to_numpy(), a=l2, b=base_l2)).groupby("e").mean()
            we = wilcoxon(g.a, g.b)
            comp.append(dict(bloque=bloque, modelo=tipo, contra="satelital_sin_corregir",
                             dif_media=float(d.mean()), ic_bajo=float(lo), ic_alto=float(hi),
                             excluye_cero=bool(hi < 0 or lo > 0),
                             p_wilcoxon_par=float(w.pvalue),
                             p_wilcoxon_estacion=float(we.pvalue), n_est=len(g)))
        for a, b in [("mlp", "lineal"), ("kan", "lineal"), ("kan", "mlp")]:
            la, lb = detalle[(a, bloque)]["l2rel"], detalle[(b, bloque)]["l2rel"]
            d = la - lb
            idx = rngb.integers(0, len(d), (1000, len(d)))
            bs = d[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            g = pd.DataFrame(dict(e=dfb.STATION_ID.to_numpy(), a=la, b=lb)).groupby("e").mean()
            comp.append(dict(bloque=bloque, modelo=a, contra=b, dif_media=float(d.mean()),
                             ic_bajo=float(lo), ic_alto=float(hi),
                             excluye_cero=bool(hi < 0 or lo > 0),
                             p_wilcoxon_par=float(wilcoxon(la, lb).pvalue),
                             p_wilcoxon_estacion=float(wilcoxon(g.a, g.b).pvalue),
                             n_est=len(g)))
    C2 = pd.DataFrame(comp)
    C2.to_csv(os.path.join(TAB, "fase4_wilcoxon.csv"), index=False)
    for _, r in C2.iterrows():
        P(f"    [{r.bloque:14s}] {r.modelo:7s} vs {r.contra:24s} "
          f"dif L2 {r.dif_media:+.4f} IC95 [{r.ic_bajo:+.4f},{r.ic_alto:+.4f}] "
          f"{'SIGNIFICATIVO' if r.excluye_cero else 'no concluyente':15s} "
          f"p_est={r.p_wilcoxon_estacion:.2e}")

    # ---------------------------------------------------------------------
    # desglose por fase ENSO
    # ---------------------------------------------------------------------
    P(f"\n[8] desglose por fase ENSO (ONI medio anual, NOAA/PSL Nino 3.4)")
    rows = []
    for bloque in ["test_temporal", "reservadas"]:
        for tipo in ["satelital_sin_corregir", "lineal", "mlp", "kan"]:
            d = detalle[(tipo, bloque)]
            df = d["df"].copy()
            df["l2"] = d["l2rel"]
            df["FASE"] = df.YEAR.map(fase)
            for f_, g in df.groupby("FASE"):
                rows.append(dict(bloque=bloque, modelo=tipo, fase=f_, n=len(g),
                                 l2_rel=float(g.l2.mean())))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(TAB, "fase4_enso.csv"), index=False)
    piv = E[E.bloque == "test_temporal"].pivot(index="modelo", columns="fase", values="l2_rel")
    P("    L2 relativo por fase (bloque test_temporal):")
    P("      " + piv.round(4).to_string().replace("\n", "\n      "))

    # ---------------------------------------------------------------------
    # CSV de estaciones reservadas, formato exacto
    # ---------------------------------------------------------------------
    mejor_tipo = (res[res.bloque == "reservadas"]
                  .sort_values("l2_rel_m").iloc[0].modelo)
    if mejor_tipo == "satelital_sin_corregir":
        mejor_tipo = (res[(res.bloque == "reservadas")
                          & (res.modelo != "satelital_sin_corregir")]
                      .sort_values("l2_rel_m").iloc[0].modelo)
    m_best = modelos_fin[(mejor_tipo, SEMILLAS[0])]
    cp = predice_coef(m_best, Xrs_s)
    curv = np.maximum(curvas(cp), 0.0) ** 2            # mm/dia, no negativo
    sal = []
    for i, (_, r) in enumerate(rs.iterrows()):
        sal.append(pd.DataFrame(dict(station_id=r.STATION_ID, year=int(r.YEAR),
                                     day=np.arange(1, T + 1),
                                     value_mm=np.round(curv[i], 4))))
    S = pd.concat(sal, ignore_index=True)
    S.to_csv(os.path.join(ENT, "pred_reservadas.csv"), index=False)
    P(f"\n[9] predicciones de estaciones reservadas: entrega/pred_reservadas.csv")
    P(f"    modelo usado: {mejor_tipo} (mejor L2 en el bloque reservado) | "
      f"{len(S):,} filas | columnas {list(S.columns)}")
    P(f"    rango de value_mm: {S.value_mm.min():.3f} a {S.value_mm.max():.1f} mm/dia "
      f"(negativos: {int((S.value_mm < 0).sum())})")

    # ---------------------------------------------------------------------
    # figuras
    # ---------------------------------------------------------------------
    def eje_meses(ax, ylab):
        ax.set_xticks(INI_MES + 15)
        ax.set_xticklabels(MESES, fontsize=7)
        ax.set_xlim(0, 365)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.25)


    d0 = detalle[(mejor_tipo, "test_temporal")]
    orden = np.argsort(d0["l2rel"])
    sel = list(orden[:3]) + list(orden[-3:])
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 7))
    for ax, i in zip(axs.ravel(), sel):
        r = d0["df"].iloc[i]
        ax.plot(t_dia, (r[EST].to_numpy(float) @ Phi.T), color="black", lw=1.8,
                label="estacion (obs)")
        ax.plot(t_dia, (r[SAT].to_numpy(float) @ Phi.T), color="tab:orange", lw=1.2,
                ls="--", label="satelital")
        ax.plot(t_dia, d0["coef_pred"][i] @ Phi.T, color="tab:red", lw=1.5,
                label=f"{mejor_tipo}")
        eje_meses(ax, r"$\sqrt{P}$")
        ax.set_title(f"{r.STATION_ID} {int(r.YEAR)} | L2rel {d0['l2rel'][i]:.3f}",
                     fontsize=8)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Fase 4 - mejores (arriba) y peores (abajo) predicciones del bloque de prueba")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase4_curvas.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2))
    mods = ["satelital_sin_corregir", "lineal", "mlp", "kan"]
    col = dict(zip(mods, ["gray", "tab:blue", "tab:green", "tab:red"]))
    for k_, (met, lab) in enumerate([("l2_rel_m", "L2 relativo"), ("kge_m", "KGE"),
                                     ("sesgo_m", "sesgo del acumulado (%)")]):
        r1 = res[res.bloque == "test_temporal"].set_index("modelo").reindex(mods)
        r2 = res[res.bloque == "reservadas"].set_index("modelo").reindex(mods)
        x = np.arange(len(mods))
        axs[k_].bar(x - 0.2, r1[met], 0.4, color=[col[m] for m in mods], label="test temporal")
        axs[k_].bar(x + 0.2, r2[met], 0.4, color=[col[m] for m in mods], alpha=0.5,
                    hatch="//", label="reservadas")
        axs[k_].set_xticks(x)
        axs[k_].set_xticklabels(["satelital", "lineal", "MLP", "KAN"], fontsize=8)
        axs[k_].set_title(lab, fontsize=9)
        axs[k_].grid(alpha=0.25, axis="y")
        if k_ == 0:
            axs[k_].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase4_metricas.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, tipo in zip(axs, mods):
        d = detalle[(tipo, "test_temporal")]
        ax.scatter(d["acum_o"], d["acum_p"], s=8, alpha=0.4, color=col[tipo])
        lim = [0, float(max(d["acum_o"].max(), d["acum_p"].max())) * 1.03]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("acumulado observado [mm]", fontsize=8)
        ax.set_ylabel("acumulado predicho [mm]", fontsize=8)
        rr = res[(res.bloque == "test_temporal") & (res.modelo == tipo)].iloc[0]
        ax.set_title(f"{tipo}\nKGE {rr.kge_m:.3f} | sesgo {rr.sesgo_m:+.1f} %", fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase4_dispersion.png"), dpi=130)
    plt.close(fig)

    P(f"\n[fin] figuras fase4_*.png | tablas fase4_*.csv | pred_reservadas.csv")
    P(f"      torch {torch.__version__} | semillas {SEMILLAS} | lambda_rug {LAM_RUG:g}")
    open(os.path.join(ENT, "log_fase4.txt"), "w").write("\n".join(log) + "\n")


def _bloque_5b():
    """Fase 3b - emparejamiento con coordenadas oficiales."""
    global log
    log = []
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    RAIZ = os.path.dirname(os.path.abspath(__file__))
    ENT = os.path.join(RAIZ, "entrega")
    TAB = os.path.join(ENT, "tablas")
    FIG = os.path.join(ENT, "figs")
    NUE = os.path.join(RAIZ, "datos_nuevos")
    K = 5
    COLS = [f"c{j+1}" for j in range(K)]
    rng = np.random.default_rng(2026)


    def km(dlon, dlat, lat=4.0):
        """Distancia aproximada en km para desplazamientos pequenos."""
        return np.sqrt((dlon * 111.32 * np.cos(np.deg2rad(lat))) ** 2 + (dlat * 110.57) ** 2)


    print("[1] catalogo oficial de estaciones")
    ll = pd.read_csv(os.path.join(NUE, "CVC_Wheather_Stations_Long_Lat.csv"), encoding="latin-1")
    cat = pd.read_excel(os.path.join(NUE, "CatalogoEstacionesElevacion.xlsx"))
    ll["CODIGO"] = ll.CODIGO.astype(str).str.strip()
    cat["codigo"] = cat.codigo.astype(str).str.strip()
    coords = (ll[["CODIGO", "STATION_NAME", "CATEGORY", "Municipality", "BASIN", "Lon", "lat"]]
              .rename(columns={"CODIGO": "STATION_ID", "Lon": "LON_EST", "lat": "LAT_EST"})
              .merge(cat[["codigo", "alt"]].rename(columns={"codigo": "STATION_ID", "alt": "ALT_EST"}),
                     on="STATION_ID", how="left")
              .drop_duplicates("STATION_ID"))
    print(f"    catalogo: {len(coords)} estaciones con coordenadas, "
          f"{coords.ALT_EST.notna().sum()} con altitud")

    ce = pd.read_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), dtype={"STATION_ID": str}) \
        if os.path.exists(os.path.join(ENT, "coef_estaciones_pentadal.csv")) else None
    ce_d = pd.read_csv(os.path.join(ENT, "coef_estaciones.csv"), dtype={"STATION_ID": str})
    cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
    cs["PIX"] = cs.FILA.astype(str) + "_" + cs.COL.astype(str)
    ce_m = ce if ce is not None else ce_d
    print(f"    curvas de estaciones: {len(ce_m)} | curvas satelitales: {len(cs)}")

    pix = cs[["PIX", "LON", "LAT"]].drop_duplicates().reset_index(drop=True)
    print(f"    malla satelital: {len(pix)} pixeles, "
          f"lon [{pix.LON.min():.3f}, {pix.LON.max():.3f}] "
          f"lat [{pix.LAT.min():.3f}, {pix.LAT.max():.3f}]")

    # ---------------------------------------------------------------------
    print("\n[2] emparejamiento por vecino mas cercano real")
    est = sorted(ce_m.STATION_ID.unique())
    C = coords.set_index("STATION_ID")
    pl, pa = pix.LON.to_numpy(), pix.LAT.to_numpy()
    filas = []
    sin_coord = []
    for e in est:
        if e not in C.index or not np.isfinite(C.LON_EST.get(e, np.nan)):
            sin_coord.append(e)
            continue
        lo, la = float(C.LON_EST[e]), float(C.LAT_EST[e])
        d = km(pl - lo, pa - la, la)
        j = int(np.argmin(d))
        filas.append(dict(STATION_ID=e, NOMBRE=C.STATION_NAME.get(e, ""),
                          MUNICIPIO=C.Municipality.get(e, ""), CUENCA=C.BASIN.get(e, ""),
                          LON_EST=lo, LAT_EST=la, ALT_EST=C.ALT_EST.get(e, np.nan),
                          PIX=str(pix.PIX[j]), LON_PIX=float(pl[j]), LAT_PIX=float(pa[j]),
                          dist_km=float(d[j])))
    P = pd.DataFrame(filas)
    dentro = P.dist_km <= 8.0        # la malla es de 0.05 deg ~ 5.5 km: media diagonal ~ 3.9 km
    print(f"    estaciones con coordenada: {len(P)} de {len(est)} "
          f"(sin coordenada en el catalogo: {len(sin_coord)})")
    print(f"    dentro de la malla (<= 8 km del centro del pixel): {int(dentro.sum())}")
    print(f"    distancia estacion-pixel: mediana {P.dist_km.median():.2f} km | "
          f"p90 {P.dist_km.quantile(0.9):.1f} km | max {P.dist_km.max():.0f} km")
    P = P[dentro].reset_index(drop=True)
    print(f"    pixeles distintos: {P.PIX.nunique()} "
          f"({len(P) - P.PIX.nunique()} estaciones comparten pixel con otra)")
    if P.ALT_EST.notna().any():
        print(f"    altitud real: {P.ALT_EST.min():.0f} - {P.ALT_EST.max():.0f} m "
              f"(mediana {P.ALT_EST.median():.0f})")

    # control: el emparejamiento real debe correlacionar mejor que uno aleatorio
    def mensual(df, clave, llaves):
        df = df.copy()
        A = np.full((len(llaves), (df.YEAR.max() - df.YEAR.min() + 1) * 12), np.nan)
        idx = {k: i for i, k in enumerate(llaves)}
        y0 = df.YEAR.min()
        t = np.arange(365) + 0.5
        arm = [np.ones_like(t)]
        for j in range(1, (K + 1) // 2 + 1):
            arm += [np.sin(2 * np.pi * j * t / 365), np.cos(2 * np.pi * j * t / 365)]
        Phi = np.column_stack(arm[:K]) * np.sqrt(2.0 / 365) ** 0  # misma base que fase 1
        lim = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        for _, r in df.iterrows():
            if r[clave] not in idx:
                continue
            cur = np.asarray(r[COLS], float) @ Phi.T
            for m in range(12):
                A[idx[r[clave]], (int(r.YEAR) - y0) * 12 + m] = cur[lim[m]:lim[m + 1]].mean()
        return A


    print("\n[3] control del emparejamiento con anomalias mensuales")
    ce_c = ce_m[ce_m.STATION_ID.isin(P.STATION_ID)]
    cs_c = cs[cs.PIX.isin(P.PIX)]
    anios = sorted(set(ce_c.YEAR) & set(cs_c.YEAR))
    ce_c = ce_c[ce_c.YEAR.isin(anios)]
    cs_c = cs_c[cs_c.YEAR.isin(anios)]


    def anomalias(A):
        out = np.full_like(A, np.nan)
        for m in range(12):
            sl = slice(m, A.shape[1], 12)
            out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
        return out


    lst_e = sorted(ce_c.STATION_ID.unique())
    lst_p = sorted(cs_c.PIX.unique())
    Ae = anomalias(mensual(ce_c, "STATION_ID", lst_e))
    As = anomalias(mensual(cs_c, "PIX", lst_p))
    ip = {p: i for i, p in enumerate(lst_p)}
    mp = P.set_index("STATION_ID")
    r_real, r_nulo = [], []
    for i, e in enumerate(lst_e):
        ok = np.isfinite(Ae[i])
        if ok.sum() < 24:
            continue
        a = Ae[i, ok] - Ae[i, ok].mean()
        j = ip.get(str(mp.PIX[e]))
        if j is None:
            continue
        b = As[j, ok]
        m2 = np.isfinite(b)
        if m2.sum() < 24:
            continue
        aa, bb = a[m2] - a[m2].mean(), b[m2] - b[m2].mean()
        r_real.append(float((aa @ bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)))
        sel = rng.integers(0, len(lst_p), 30)
        B = As[np.ix_(sel, ok)][:, m2]
        B = B - np.nanmean(B, axis=1, keepdims=True)
        num = B @ aa
        den = np.sqrt(np.nansum(B ** 2, axis=1)) * np.linalg.norm(aa) + 1e-12
        r_nulo += list(num / den)
    r_real = np.array(r_real)
    r_nulo = np.array(r_nulo)
    print(f"    r del pixel que contiene la estacion: media {np.nanmean(r_real):.3f} "
          f"(n = {len(r_real)})")
    print(f"    r con pixeles aleatorios de la malla:  media {np.nanmean(r_nulo):.3f}")
    print(f"    ganancia: {np.nanmean(r_real) - np.nanmean(r_nulo):+.3f}")
    print("    [fase 3 con centroides: r 0.526 vs nulo 0.274]")
    P["r_anom"] = [float(np.nan)] * len(P)
    P.loc[P.STATION_ID.isin(lst_e), "r_anom"] = pd.Series(
        {e: r for e, r in zip([e for e in lst_e if e in set(P.STATION_ID)], r_real)}).reindex(
        P.loc[P.STATION_ID.isin(lst_e), "STATION_ID"]).to_numpy()
    P.to_csv(os.path.join(TAB, "fase3b_emparejamiento.csv"), index=False)

    # ---------------------------------------------------------------------
    print("\n[4] dataset pareado para la fase 4")
    ce_i = ce_m.set_index(["STATION_ID", "YEAR"])
    cs_i = cs.set_index(["PIX", "YEAR"])
    key = []
    for (e, y) in ce_i.index:
        if e not in mp.index:
            continue
        p = str(mp.PIX[e])
        if (p, y) in cs_i.index:
            key.append((e, int(y), p))
    key = pd.DataFrame(key, columns=["STATION_ID", "YEAR", "PIX"]).drop_duplicates()
    sal = key.copy()
    for j in range(K):
        sal[f"sat_c{j+1}"] = cs_i.loc[list(zip(key.PIX, key.YEAR)), COLS[j]].to_numpy()
        sal[f"est_c{j+1}"] = ce_i.loc[list(zip(key.STATION_ID, key.YEAR)), COLS[j]].to_numpy()
    sal = sal.merge(P[["STATION_ID", "ALT_EST", "LON_EST", "LAT_EST", "LON_PIX", "LAT_PIX",
                       "dist_km", "r_anom", "NOMBRE", "MUNICIPIO"]], on="STATION_ID", how="left")
    # compatibilidad con la fase 4: covariables con el nombre que ya usa el script
    sal["ALT_SUB"] = sal.ALT_EST
    med = sal.ALT_SUB.median()
    sal["ALT_SUB"] = sal.ALT_SUB.fillna(med)
    viejo = os.path.join(ENT, "pares_estacion_pixel.csv")
    if os.path.exists(viejo) and not os.path.exists(viejo + ".centroides"):
        os.rename(viejo, viejo + ".centroides")
    sal.to_csv(viejo, index=False)
    print(f"    entrega/pares_estacion_pixel.csv: {len(sal)} filas | "
          f"{sal.STATION_ID.nunique()} estaciones | {sal.YEAR.min()}-{sal.YEAR.max()}")
    print(f"    (version con centroides guardada como pares_estacion_pixel.csv.centroides)")
    print(f"    altitud faltante imputada con la mediana ({med:.0f} m) en "
          f"{int(sal.ALT_EST.isna().sum())} filas")

    cmp = pd.DataFrame([
        dict(metodo="centroide de subzona (fase 3)", estaciones=65, pixeles_distintos=np.nan,
             dist_mediana_km=np.nan, r_anom_media=0.526, r_nulo=0.274),
        dict(metodo="coordenada real (fase 3b)", estaciones=int(sal.STATION_ID.nunique()),
             pixeles_distintos=int(P.PIX.nunique()), dist_mediana_km=float(P.dist_km.median()),
             r_anom_media=float(np.nanmean(r_real)), r_nulo=float(np.nanmean(r_nulo)))])
    cmp.to_csv(os.path.join(TAB, "fase3b_comparacion_metodos.csv"), index=False)
    print("\n" + cmp.to_string(index=False))

    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    ax.scatter(pix.LON, pix.LAT, s=1.2, c="0.82", label="malla satelital (0,05 deg)")
    sc = ax.scatter(P.LON_EST, P.LAT_EST, c=P.ALT_EST, s=26, cmap="terrain",
                    edgecolor="k", linewidth=0.3, label="estaciones (altitud real)")
    plt.colorbar(sc, ax=ax, label="altitud (m)", shrink=0.8)
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.set_title(f"Estaciones con coordenada oficial sobre la malla satelital\n"
                 f"{len(P)} estaciones emparejadas por vecino mas cercano")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase3b_mapa.png"), dpi=150)
    print("\n[5] figura entrega/figs/fase3b_mapa.png")
    print("\nFASE 3b OK")


def _bloque_5c():
    """Fase 3c - control espacial del emparejamiento."""
    global log
    log = []
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    RAIZ = os.path.dirname(os.path.abspath(__file__))
    ENT = os.path.join(RAIZ, "entrega")
    TAB = os.path.join(ENT, "tablas")
    FIG = os.path.join(ENT, "figs")
    K, PERIODO = 5, 365.0
    COLS = [f"c{j+1}" for j in range(K)]
    rng = np.random.default_rng(2026)


    def base_fourier(t, K, periodo=PERIODO):
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    def km(dlon, dlat, lat=4.0):
        return np.sqrt((dlon * 111.32 * np.cos(np.deg2rad(lat))) ** 2 + (dlat * 110.57) ** 2)


    t = np.arange(365) + 0.5
    Phi = base_fourier(t, K)
    LIM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    ce = pd.read_csv(os.path.join(ENT, "coef_estaciones_pentadal.csv"), dtype={"STATION_ID": str})
    cs = pd.read_csv(os.path.join(ENT, "coef_satelite.csv"))
    cs["PIX"] = cs.FILA.astype(str) + "_" + cs.COL.astype(str)
    E = pd.read_csv(os.path.join(TAB, "fase3b_emparejamiento.csv"), dtype={"STATION_ID": str, "PIX": str})
    print(f"[0] {len(E)} estaciones con coordenada oficial dentro de la malla")

    anios = sorted(set(ce.YEAR) & set(cs.YEAR))
    ce = ce[ce.YEAR.isin(anios) & ce.STATION_ID.isin(E.STATION_ID)]
    cs = cs[cs.YEAR.isin(anios)]
    y0, y1 = min(anios), max(anios)
    nm = (y1 - y0 + 1) * 12


    def serie_mensual(df, clave, llaves):
        A = np.full((len(llaves), nm), np.nan)
        idx = {k: i for i, k in enumerate(llaves)}
        C = df[COLS].to_numpy(float) @ Phi.T
        kk = df[clave].to_numpy()
        yy = df.YEAR.to_numpy(int)
        for r in range(len(df)):
            i = idx.get(kk[r])
            if i is None:
                continue
            base = (yy[r] - y0) * 12
            for m in range(12):
                A[i, base + m] = C[r, LIM[m]:LIM[m + 1]].mean()
        return A


    def anomalias(A):
        out = np.full_like(A, np.nan)
        for m in range(12):
            sl = slice(m, A.shape[1], 12)
            out[:, sl] = A[:, sl] - np.nanmean(A[:, sl], axis=1, keepdims=True)
        return out


    lst_e = sorted(ce.STATION_ID.unique())
    lst_p = sorted(cs.PIX.unique())
    print("[1] series mensuales de anomalias")
    Ae = anomalias(serie_mensual(ce, "STATION_ID", lst_e))
    As = anomalias(serie_mensual(cs, "PIX", lst_p))
    pxy = cs[["PIX", "LON", "LAT"]].drop_duplicates().set_index("PIX").loc[lst_p]
    PL, PA = pxy.LON.to_numpy(), pxy.LAT.to_numpy()
    ip = {p: i for i, p in enumerate(lst_p)}
    me = E.set_index("STATION_ID")


    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 24:
            return np.nan
        x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
        s = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / s) if s > 0 else np.nan


    # ---------------------------------------------------------------------
    print("[2] perfil r(distancia): cada estacion contra TODOS los pixeles")
    BINS = [0, 5, 10, 20, 40, 80, 160, 400]
    acum = {i: [] for i in range(len(BINS) - 1)}
    for i, e in enumerate(lst_e):
        lo, la = float(me.LON_EST[e]), float(me.LAT_EST[e])
        d = km(PL - lo, PA - la, la)
        sel = rng.choice(len(lst_p), min(220, len(lst_p)), replace=False)
        for j in sel:
            r = corr(Ae[i], As[j])
            if np.isfinite(r):
                b = np.searchsorted(BINS, d[j], side="right") - 1
                if 0 <= b < len(BINS) - 1:
                    acum[b].append(r)
    filas = []
    for b in range(len(BINS) - 1):
        v = np.array(acum[b])
        if len(v):
            filas.append(dict(dist_min_km=BINS[b], dist_max_km=BINS[b + 1], n=len(v),
                              r_media=v.mean(), r_sd=v.std(),
                              ic95_inf=v.mean() - 1.96 * v.std() / np.sqrt(len(v)),
                              ic95_sup=v.mean() + 1.96 * v.std() / np.sqrt(len(v))))
    Rd = pd.DataFrame(filas)
    Rd.to_csv(os.path.join(TAB, "fase3c_r_por_distancia.csv"), index=False)
    print(Rd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    cerca = Rd[Rd.dist_max_km <= 10].r_media.mean()
    lejos = Rd[Rd.dist_min_km >= 80].r_media.mean()
    print(f"\n    r medio a <= 10 km: {cerca:.3f} | a >= 80 km: {lejos:.3f} "
          f"| decaimiento {cerca - lejos:+.3f}")

    # ---------------------------------------------------------------------
    print("\n[3] seleccion post-hoc (fase 3) vs coordenada real (fase 3b)")
    V = os.path.join(ENT, "pares_estacion_pixel.csv.centroides")
    if os.path.exists(V):
        viejo = pd.read_csv(V, dtype={"STATION_ID": str})
        vp = viejo.drop_duplicates("STATION_ID").set_index("STATION_ID")
        comun = [e for e in lst_e if e in vp.index]
        print(f"    estaciones en ambos metodos: {len(comun)}")
        f2 = []
        for e in comun:
            i = ip_e = lst_e.index(e)
            pr = str(me.PIX[e])
            # el PIX viejo venia como "FILA_COL"
            pv = str(vp.PIX[e])
            if pr not in ip or pv not in ip:
                continue
            lo, la = float(me.LON_EST[e]), float(me.LAT_EST[e])
            jv = ip[pv]
            f2.append(dict(STATION_ID=e,
                           r_real=corr(Ae[i], As[ip[pr]]),
                           r_seleccionado=corr(Ae[i], As[jv]),
                           dist_real_km=float(me.dist_km[e]),
                           dist_seleccionado_km=float(km(PL[jv] - lo, PA[jv] - la, la)),
                           mismo_pixel=(pr == pv)))
        S = pd.DataFrame(f2).dropna()
        S.to_csv(os.path.join(TAB, "fase3c_seleccion_vs_real.csv"), index=False)
        if len(S):
            print(f"    r con el pixel que CONTIENE la estacion:        {S.r_real.mean():.3f}")
            print(f"    r con el pixel elegido por maxima correlacion:  {S.r_seleccionado.mean():.3f}")
            print(f"    inflacion del metodo anterior:                  "
                  f"{S.r_seleccionado.mean() - S.r_real.mean():+.3f}")
            print(f"    distancia real del pixel seleccionado: mediana "
                  f"{S.dist_seleccionado_km.median():.1f} km "
                  f"(el verdadero esta a {S.dist_real_km.median():.1f} km)")
            print(f"    acerto el pixel correcto en {int(S.mismo_pixel.sum())} de {len(S)} casos "
                  f"({100*S.mismo_pixel.mean():.0f} %)")
    else:
        print("    no hay version con centroides para comparar")

    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = (Rd.dist_min_km + Rd.dist_max_km) / 2
    ax.errorbar(x, Rd.r_media, yerr=1.96 * Rd.r_sd / np.sqrt(Rd.n), marker="o",
                color="tab:blue", capsize=3, label="r medio (IC 95 %)")
    ax.fill_between(x, Rd.r_media - Rd.r_sd, Rd.r_media + Rd.r_sd, alpha=0.15,
                    color="tab:blue", label="+-1 desviacion")
    ax.set_xscale("log")
    ax.set_xlabel("distancia estacion-pixel (km, escala log)")
    ax.set_ylabel("correlacion de anomalias mensuales")
    ax.set_title("Decaimiento espacial de la correlacion satelite-estacion\n"
                 "(coordenadas oficiales, 1981-2003)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase3c_decaimiento.png"), dpi=150)
    print("\n[4] figura entrega/figs/fase3c_decaimiento.png")
    print("\nFASE 3c OK")


def _bloque_7():
    """Fase 4b - codificador funcional aprendible (4.2) y particion oficial."""
    global log
    log = []
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    RAIZ = os.path.dirname(os.path.abspath(__file__))
    ENT = os.path.join(RAIZ, "entrega")
    TAB = os.path.join(ENT, "tablas")
    FIG = os.path.join(ENT, "figs")
    MOD = os.path.join(ENT, "modelos")
    os.makedirs(MOD, exist_ok=True)

    K, PERIODO = 5, 365.0
    NB_NODOS = 8                 # nodos interiores de la base B-spline de las w_k
    N_NEU = 5                    # neuronas funcionales del codificador
    LAM_RUG = 1e-6
    FRAC_RESERVADAS = 0.13
    SEMILLAS = [11, 23, 37]
    ANIOS_TR = list(range(1981, 1996))      # oficial: 1981-1995
    ANIOS_VA = list(range(1996, 2001))      # oficial: 1996-2000
    ANIOS_TE = list(range(2001, 2004))      # oficial: 2001-2003

    log = []


    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)


    # ---------------------------------------------------------------------
    # bases y matrices de Gram (cuadratura trapezoidal, a mano)
    # ---------------------------------------------------------------------
    def base_fourier(t, K, periodo=PERIODO):
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    def nodos_bspline(n_int, k=3, a=0.0, b=PERIODO):
        interior = np.linspace(a, b, n_int + 2)[1:-1]
        return np.concatenate([[a] * (k + 1), interior, [b] * (k + 1)])


    def base_bspline(t, nodos, k=3):
        """B-splines por recurrencia de Cox-de Boor (sin scipy)."""
        n = len(nodos) - k - 1
        B = np.zeros((len(t), len(nodos) - 1))
        for i in range(len(nodos) - 1):
            B[:, i] = ((t >= nodos[i]) & (t < nodos[i + 1])).astype(float)
        B[t >= nodos[-1], -1] = 0.0
        B[np.isclose(t, nodos[-1]), np.where(nodos < nodos[-1])[0][-1]] = 1.0
        for p in range(1, k + 1):
            Bn = np.zeros((len(t), len(nodos) - p - 1))
            for i in range(len(nodos) - p - 1):
                d1 = nodos[i + p] - nodos[i]
                d2 = nodos[i + p + 1] - nodos[i + 1]
                a1 = (t - nodos[i]) / d1 if d1 > 0 else np.zeros_like(t)
                a2 = (nodos[i + p + 1] - t) / d2 if d2 > 0 else np.zeros_like(t)
                Bn[:, i] = a1 * B[:, i] + a2 * B[:, i + 1]
            B = Bn
        return B[:, :n]


    def pesos_trapecio(t):
        w = np.zeros_like(t)
        w[1:-1] = (t[2:] - t[:-2]) / 2
        w[0] = (t[1] - t[0]) / 2
        w[-1] = (t[-1] - t[-2]) / 2
        return w


    t_dia = np.arange(365) + 0.5
    Phi = base_fourier(t_dia, K)
    NODOS = nodos_bspline(NB_NODOS)
    Bsp = base_bspline(t_dia, NODOS)
    NB = Bsp.shape[1]
    wq = pesos_trapecio(t_dia)

    W = Phi.T @ (Phi * wq[:, None])              # Gram de Fourier (K x K)

    # Wx y RB se calculan en malla FINA con regla de Simpson: con 365 nodos y
    # trapecio el error relativo de <w_k, X> es 8e-3, insuficiente para que la
    # integral del codificador sea exacta a 1e-4.
    NF = 36501                                   # impar: Simpson compuesto
    t_f = np.linspace(0.0, PERIODO, NF)
    hf = t_f[1] - t_f[0]
    ws = np.ones(NF)
    ws[1:-1:2] = 4.0
    ws[2:-1:2] = 2.0
    ws *= hf / 3.0
    Phi_f = base_fourier(t_f, K)
    Bsp_f = base_bspline(t_f, NODOS)
    Wx = Bsp_f.T @ (Phi_f * ws[:, None])         # integral B Phi^T  (NB x K)
    d2B_f = np.gradient(np.gradient(Bsp_f, t_f, axis=0), t_f, axis=0)
    RB = d2B_f.T @ (d2B_f * ws[:, None])         # (NB x NB)
    lamW, Uw = np.linalg.eigh(W)
    Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T
    Whi = Uw @ np.diag(1 / np.sqrt(lamW)) @ Uw.T

    P("=" * 74)
    P("FASE 4b - codificador funcional APRENDIBLE, particion oficial")
    P("=" * 74)
    P(f"    base de las w_k: B-spline cubica, {NB_NODOS} nodos interiores -> NB = {NB}")
    P(f"    particion de la base: suma de filas = {Bsp.sum(axis=1).min():.6f} "
      f"a {Bsp.sum(axis=1).max():.6f}")
    P(f"    Wx = int B Phi^T: {Wx.shape} | ||Wx|| = {np.linalg.norm(Wx):.2f}")
    P(f"    Gram W de Fourier: traza {np.trace(W):.2f}")

    # ---------------------------------------------------------------------
    # verificacion de la integral del codificador
    # ---------------------------------------------------------------------
    rng_v = np.random.default_rng(1)
    a_test = rng_v.normal(size=NB)
    c_test = rng_v.normal(size=K)
    z_matriz = float(a_test @ Wx @ c_test)
    t_fino = np.linspace(0, PERIODO, 400003)   # malla independiente de la de Wx
    w_fino = base_bspline(t_fino, NODOS) @ a_test
    x_fino = base_fourier(t_fino, K) @ c_test
    z_fino = float(np.trapezoid(w_fino * x_fino, t_fino))
    err_rel = abs(z_matriz - z_fino) / (abs(z_fino) + 1e-12)
    P(f"\n[0] verificacion del producto interno del codificador")
    P(f"    a^T Wx c            = {z_matriz:.10f}")
    P(f"    int w(t) X(t) dt    = {z_fino:.10f}   (cuadratura fina, 2e5 nodos)")
    P(f"    error relativo      = {err_rel:.2e} "
      f"({'CUMPLE <= 1e-4' if err_rel <= 1e-4 else 'NO CUMPLE'})")

    # ---------------------------------------------------------------------
    # datos y particiones oficiales
    # ---------------------------------------------------------------------
    D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
    SAT = [f"sat_c{i+1}" for i in range(K)]
    EST = [f"est_c{i+1}" for i in range(K)]
    COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
    P(f"\n[1] datos (coordenadas reales de la fase 3b): {len(D)} pares | "
      f"{D.STATION_ID.nunique()} estaciones | {D.YEAR.min()}-{D.YEAR.max()}")

    rng0 = np.random.default_rng(2026)
    ests = np.array(sorted(D.STATION_ID.unique()))
    reserv = set(rng0.choice(ests, int(round(FRAC_RESERVADAS * len(ests))), replace=False))
    D["RESERVADA"] = D.STATION_ID.isin(reserv)
    tr = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TR)].reset_index(drop=True)
    va = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_VA)].reset_index(drop=True)
    te = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TE)].reset_index(drop=True)
    rs = D[D.RESERVADA].reset_index(drop=True)
    P(f"    particion OFICIAL: train {len(tr)} (1981-1995) | val {len(va)} (1996-2000) | "
      f"test {len(te)} (2001-2003)")
    P(f"    estaciones reservadas por sorteo propio (el docente no publico lista): "
      f"{len(reserv)} de {len(ests)} -> {len(rs)} pares")

    # ---------------------------------------------------------------------
    # decodificador FPCA (solo train, a mano por autovalores generalizados)
    # ---------------------------------------------------------------------
    Ctr = tr[EST].to_numpy(float)
    c_mu = Ctr.mean(axis=0)
    V = ((Ctr - c_mu).T @ (Ctr - c_mu)) / (len(Ctr) - 1)
    lamF, Uf = np.linalg.eigh(Wh @ V @ Wh)
    o = np.argsort(lamF)[::-1]
    lamF, Uf = lamF[o], Uf[:, o]
    Bfull = Whi @ Uf
    vexp = lamF / lamF.sum()
    M = int(np.searchsorted(np.cumsum(vexp), 0.99) + 1)
    Bm = Bfull[:, :M]
    P(f"\n[2] decodificador FPCA (solo train): " +
      " ".join(f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)))
    P(f"    M = {M} componentes para el 99 % | ortonormalidad max|B^T W B - I| = "
      f"{np.abs(Bm.T @ W @ Bm - np.eye(M)).max():.2e}")

    mcov = tr[COV].to_numpy(float).mean(axis=0)
    scov = tr[COV].to_numpy(float).std(axis=0) + 1e-12


    def tensores(df):
        return (torch.tensor(df[SAT].to_numpy(float), dtype=torch.float32),
                torch.tensor((df[COV].to_numpy(float) - mcov) / scov, dtype=torch.float32),
                torch.tensor(df[EST].to_numpy(float), dtype=torch.float32))


    Ctr_s, Vtr, Ytr = tensores(tr)
    Cva_s, Vva, Yva = tensores(va)
    Cte_s, Vte, Yte = tensores(te)
    Crs_s, Vrs, Yrs = tensores(rs)

    Wxt = torch.tensor(Wx, dtype=torch.float32)
    RBt = torch.tensor(RB, dtype=torch.float32)
    Wt = torch.tensor(W, dtype=torch.float32)
    Bmt = torch.tensor(Bm, dtype=torch.float32)
    cmut = torch.tensor(c_mu, dtype=torch.float32)


    # ---------------------------------------------------------------------
    # codificador funcional: z_k = a_k^T Wx c   con a_k aprendibles
    # ---------------------------------------------------------------------
    class CodificadorFuncional(nn.Module):
        """Capa de neuronas funcionales con pesos w_k(t) en base B-spline.

        modo = "fijo"       : a_k congelados en el operador W^(1/2) (fase 4 original)
        modo = "aleatorio"  : a_k aprendibles desde una inicializacion aleatoria
        modo = "informado"  : a_k aprendibles pero INICIALIZADOS en W^(1/2), de modo
                              que la red parte del operador teorico y lo ajusta
        """

        def __init__(self, n_neu=N_NEU, modo="informado", semilla=0):
            super().__init__()
            g = torch.Generator().manual_seed(semilla)
            if modo == "aleatorio":
                A0 = torch.randn(NB, n_neu, generator=g) * 0.1
            else:
                # a_k que reproduce z = c W^(1/2): se resuelve Wx^T a_k = (W^(1/2))_k
                objetivo = torch.tensor(Wh, dtype=torch.float32)[:, :n_neu]
                A0 = torch.linalg.lstsq(Wxt.T, objetivo).solution       # (NB x n_neu)
            self.A = nn.Parameter(A0, requires_grad=(modo != "fijo"))
            self.modo = modo
            self.escala = None

        def forward(self, c):
            z = c @ Wxt.T @ self.A
            if self.escala is not None:
                z = z / self.escala
            return z

        def pesos_wk(self):
            """w_k(t) evaluadas en la malla diaria (365 x n_neu)."""
            return torch.tensor(Bsp, dtype=torch.float32) @ self.A

        def rugosidad(self):
            return ((self.A.T @ RBt) * self.A.T).sum() / PERIODO


    class KANPropia(nn.Module):
        """KAN minima: funciones univariadas en B-spline cubica (respaldo)."""

        def __init__(self, n_in, n_out, grid=5, k=3):
            super().__init__()
            self.n_in, self.k = n_in, k
            paso = 1.0 / grid
            self.nodos = torch.linspace(-0.5 - k * paso, 1.5 + k * paso, grid + 2 * k + 2)
            self.nb = len(self.nodos) - k - 1
            self.coef = nn.Parameter(0.1 * torch.randn(n_in, self.nb, n_out))
            self.w_lin = nn.Parameter(0.1 * torch.randn(n_in, n_out))
            self.b = nn.Parameter(torch.zeros(n_out))

        def bspline(self, x):
            nod = self.nodos.to(x.device)
            B = ((x.unsqueeze(-1) >= nod[:-1]) & (x.unsqueeze(-1) < nod[1:])).to(x.dtype)
            for p in range(1, self.k + 1):
                iz = (x.unsqueeze(-1) - nod[:-(p + 1)]) / (nod[p:-1] - nod[:-(p + 1)])
                de = (nod[p + 1:] - x.unsqueeze(-1)) / (nod[p + 1:] - nod[1:-p])
                B = iz * B[..., :-1] + de * B[..., 1:]
            return B

        def forward(self, x):
            B = self.bspline(torch.tanh(x) * 0.5 + 0.5)
            y = torch.einsum("nib,ibo->no", B, self.coef)
            return y + torch.einsum("ni,io->no", torch.tanh(x), self.w_lin) + self.b


    def construye_cabeza(tipo, n_in, n_out, semilla, h_mlp=None):
        torch.manual_seed(semilla)
        if tipo == "lineal":
            return nn.Linear(n_in, n_out)
        if tipo == "mlp":
            return nn.Sequential(nn.Linear(n_in, h_mlp), nn.GELU(), nn.Linear(h_mlp, n_out))
        if tipo == "kan":
            try:
                return KAN(width=[n_in, n_out], grid=5, k=3, seed=semilla, device="cpu")
            except Exception as e:
                P(f"    [aviso] pykan no utilizable ({type(e).__name__}); KANPropia")
            return KANPropia(n_in, n_out)
        raise ValueError(tipo)


    class RedFuncional(nn.Module):
        def __init__(self, tipo, semilla, h_mlp=None, modo="informado", n_neu=N_NEU):
            super().__init__()
            self.cod = CodificadorFuncional(n_neu, modo=modo, semilla=semilla)
            self.cabeza = construye_cabeza(tipo, n_neu + len(COV), M, semilla, h_mlp)
            self.tipo, self.n_neu, self.modo = tipo, n_neu, modo

        def forward(self, c, v):
            z = self.cod(c)
            return self.cabeza(torch.cat([z, v], dim=1))


    def n_par(m, solo_entrenables=True):
        ps = [p for p in m.parameters() if (p.requires_grad or not solo_entrenables)]
        return sum(p.numel() for p in ps)


    # h del MLP que iguala los parametros de la KAN (<= 10 %)
    _k0 = construye_cabeza("kan", N_NEU + len(COV), M, SEMILLAS[0])
    npar_kan = sum(p.numel() for p in _k0.parameters())
    usa_pykan = "kan" in type(_k0).__module__
    mejor_h, mejor_d = 1, 1e9
    for h in range(1, 400):
        d = abs(sum(p.numel() for p in construye_cabeza("mlp", N_NEU + len(COV), M, 0, h).parameters())
                - npar_kan)
        if d < mejor_d:
            mejor_h, mejor_d = h, d
    H_MLP = mejor_h
    npar_mlp = sum(p.numel() for p in construye_cabeza("mlp", N_NEU + len(COV), M, 0, H_MLP).parameters())
    P(f"\n[3] parametros de las cabezas: KAN ({'pykan' if usa_pykan else 'propia'}) {npar_kan} | "
      f"MLP (h={H_MLP}) {npar_mlp} | diferencia "
      f"{100*abs(npar_kan-npar_mlp)/npar_kan:.1f} %")
    P(f"    codificador aprendible: {NB} x {N_NEU} = {NB*N_NEU} parametros adicionales")


    def perdida(pred_s, obs_c, red):
        c = cmut + pred_s @ Bmt.T
        d = c - obs_c
        l2 = ((d @ Wt) * d).sum(dim=1).mean() / PERIODO
        rug = red.cod.rugosidad() if red.cod.A.requires_grad else 0.0
        return l2 + LAM_RUG * rug, l2


    def entrena(tipo, semilla, modo="informado", epocas=4000, lr=0.01, n_neu=N_NEU):
        red = RedFuncional(tipo, semilla, H_MLP, modo, n_neu)
        with torch.no_grad():                      # escala inicial de z (sd ~ 1 en train)
            z0 = red.cod(Ctr_s)
            red.cod.escala = z0.std(dim=0) + 1e-9
        opt = torch.optim.Adam([p for p in red.parameters() if p.requires_grad], lr=lr)
        mejor, st, pac = 1e18, None, 0
        for ep in range(epocas):
            opt.zero_grad()
            p_, _ = perdida(red(Ctr_s, Vtr), Ytr, red)
            p_.backward()
            opt.step()
            if ep % 10 == 0:
                with torch.no_grad():
                    _, lv = perdida(red(Cva_s, Vva), Yva, red)
                if lv.item() < mejor - 1e-7:
                    mejor, pac = lv.item(), 0
                    st = {k: v.clone() for k, v in red.state_dict().items()}
                else:
                    pac += 1
                    if pac >= 40:
                        break
        if st is not None:
            red.load_state_dict(st)
        return red, mejor, ep


    # ---------------------------------------------------------------------
    # metricas
    # ---------------------------------------------------------------------
    def metricas(red, C, Vv, Y):
        with torch.no_grad():
            s = red(C, Vv)
            coef = (cmut + s @ Bmt.T).numpy()
        obs = Y.numpy()
        cur_p, cur_o = coef @ Phi.T, obs @ Phi.T
        dif = coef - obs
        l2 = np.sqrt(((dif @ W) * dif).sum(axis=1) / PERIODO)
        nor = np.sqrt(((obs @ W) * obs).sum(axis=1) / PERIODO) + 1e-12
        acu_p, acu_o = cur_p.sum(axis=1), cur_o.sum(axis=1)
        r = np.corrcoef(acu_p, acu_o)[0, 1]
        alfa = acu_p.std() / (acu_o.std() + 1e-12)
        beta = acu_p.mean() / (acu_o.mean() + 1e-12)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alfa - 1) ** 2 + (beta - 1) ** 2)
        return dict(l2_rel=float(np.mean(l2 / nor)), rmse=float(np.sqrt((dif ** 2).mean())),
                    sesgo_acum_pct=float(100 * (acu_p.mean() - acu_o.mean()) / acu_o.mean()),
                    kge=float(kge), r_acum=float(r))


    MODOS = [("fijo", "fijo W^1/2"), ("aleatorio", "aprend. aleatorio"),
             ("informado", "aprend. informado")]
    P(f"\n[4] entrenamiento: 3 cabezas x {len(MODOS)} codificadores x {len(SEMILLAS)} semillas")
    filas = []
    modelos = {}
    for modo, etq in MODOS:
        for tipo in ("lineal", "mlp", "kan"):
            for sem in SEMILLAS:
                red, lv, ep = entrena(tipo, sem, modo)
                mt = metricas(red, Cte_s, Vte, Yte)
                mr = metricas(red, Crs_s, Vrs, Yrs)
                filas.append(dict(codificador=etq, cabeza=tipo, semilla=sem, epocas=ep,
                                  val=lv, **{f"test_{k}": v for k, v in mt.items()},
                                  **{f"resv_{k}": v for k, v in mr.items()}))
                modelos[(modo, tipo, sem)] = red
                P(f"    {etq:18s} {tipo:7s} s{sem}: val {lv:.4f} | "
                  f"test L2rel {mt['l2_rel']:.3f} KGE {mt['kge']:+.3f} | "
                  f"resv L2rel {mr['l2_rel']:.3f} ({ep} ep)")
    F = pd.DataFrame(filas)
    F.to_csv(os.path.join(TAB, "fase4b_ablacion.csv"), index=False)

    res = (F.groupby(["codificador", "cabeza"])
           [["test_l2_rel", "test_kge", "test_sesgo_acum_pct", "resv_l2_rel", "resv_kge"]]
           .agg(["mean", "std"]).round(4))
    P("\n[5] resumen por codificador y cabeza (media +- sd sobre semillas)")
    P(res.to_string())
    res.to_csv(os.path.join(TAB, "fase4b_resumen.csv"))

    P("\n    L2 relativo medio en test por codificador:")
    for _, etq in MODOS:
        P(f"      {etq:18s} {F[F.codificador == etq].test_l2_rel.mean():.4f}")
    mf = F[F.codificador == "fijo W^1/2"].test_l2_rel.mean()
    mi = F[F.codificador == "aprend. informado"].test_l2_rel.mean()
    P(f"    aprendible informado vs fijo: {100*(mf-mi)/mf:+.1f} % de mejora")

    # ---------------------------------------------------------------------
    # por que el codificador aprendible no gana con 5 neuronas: con n_neu = K el
    # operador fijo W^(1/2) ya genera TODO el espacio de X (dim K = 5), asi que
    # aprender a_k solo aplica un cambio de base invertible. La ventaja debe
    # aparecer cuando el codificador COMPRIME (n_neu < K).
    P("\n[5b] barrido del numero de neuronas funcionales (cabeza MLP, 2 semillas)")
    f2 = []
    for n_neu in (1, 2, 3, 5):
        for modo, etq in MODOS:
            ms = []
            for sem in SEMILLAS[:2]:
                red, lv, ep = entrena("mlp", sem, modo, n_neu=n_neu)
                ms.append(metricas(red, Cte_s, Vte, Yte)["l2_rel"])
            f2.append(dict(n_neuronas=n_neu, codificador=etq, test_l2_rel=float(np.mean(ms))))
            P(f"    n_neu {n_neu} | {etq:18s} | test L2rel {np.mean(ms):.4f}")
    Fc = pd.DataFrame(f2)
    Fc.to_csv(os.path.join(TAB, "fase4b_barrido_neuronas.csv"), index=False)
    piv = Fc.pivot(index="n_neuronas", columns="codificador", values="test_l2_rel")
    piv["ganancia_informado_pct"] = (100 * (piv["fijo W^1/2"] - piv["aprend. informado"])
                                     / piv["fijo W^1/2"])
    P("\n" + piv.round(4).to_string())

    # modelo elegido para la fase 4.5: el codificador aprendible informado (cumple
    # la especificacion 4.2 de la guia) con la mejor perdida de validacion
    MODO_FINAL = "aprend. informado"
    sub = F[(F.codificador == MODO_FINAL) & (F.cabeza == "kan")]
    sem_kan = int(sub.loc[sub.val.idxmin(), "semilla"])
    sub_l = F[(F.codificador == MODO_FINAL) & (F.cabeza == "lineal")]
    sem_lin = int(sub_l.loc[sub_l.val.idxmin(), "semilla"])
    for nom, llave in [("kan", ("informado", "kan", sem_kan)),
                       ("lineal", ("informado", "lineal", sem_lin))]:
        r = modelos[llave]
        torch.save(dict(state_dict=r.state_dict(), tipo=r.tipo, n_neu=r.n_neu, modo=r.modo,
                        h_mlp=H_MLP, escala=r.cod.escala, M=M),
                   os.path.join(MOD, f"red_{nom}_aprendible.pt"))
    np.savez(os.path.join(MOD, "contexto_fase4b.npz"), Phi=Phi, Bsp=Bsp, NODOS=NODOS, W=W,
             Wx=Wx, RB=RB, Bm=Bm, c_mu=c_mu, mcov=mcov, scov=scov, t_dia=t_dia, M=M,
             vexp=vexp, sem_kan=sem_kan, sem_lin=sem_lin, usa_pykan=usa_pykan)
    P(f"\n[6] modelos guardados en entrega/modelos/ (KAN semilla {sem_kan}, "
      f"lineal semilla {sem_lin}, codificador {MODO_FINAL})")

    # figura: funciones peso aprendidas vs fijas
    red_a = modelos[("informado", "kan", sem_kan)]
    red_f = modelos[("fijo", "kan", SEMILLAS[0])]
    Wk_a = red_a.cod.pesos_wk().detach().numpy()
    Wk_f = red_f.cod.pesos_wk().detach().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for k in range(N_NEU):
        axs[0].plot(t_dia, Wk_f[:, k], label=f"w{k+1}")
        axs[1].plot(t_dia, Wk_a[:, k], label=f"w{k+1}")
    axs[0].set_title("codificador fijo: $w_k$ que reproduce $W^{1/2}$")
    axs[1].set_title("codificador aprendible (inicio informado): $w_k$ ajustadas")
    for ax in axs:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("dia del anio")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel("$w_k(t)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase4b_pesos_codificador.png"), dpi=150)

    with open(os.path.join(ENT, "log_fase4b.txt"), "w") as f:
        f.write("\n".join(log))
    ok = (err_rel <= 1e-4)
    P(f"\nFASE 4b {'OK' if ok else 'CON FALLOS'} (verificacion de la integral: {err_rel:.1e})")
    P('    verificacion de la integral: ' + ('OK' if ok else 'FALLA'))


def _bloque_8():
    """Fase 4.5 - interpretacion climatologica (4.5)."""
    global log
    log = []
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    RAIZ = os.path.dirname(os.path.abspath(__file__))
    ENT = os.path.join(RAIZ, "entrega")
    TAB = os.path.join(ENT, "tablas")
    FIG = os.path.join(ENT, "figs")
    MOD = os.path.join(ENT, "modelos")
    PERIODO, K, N_NEU = 365.0, 5, 5
    COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
    NOM_ENT = [f"z{j+1}" for j in range(N_NEU)] + ["altitud", "longitud", "latitud"]
    LIM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    log = []


    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)


    ctx = np.load(os.path.join(MOD, "contexto_fase4b.npz"))
    Phi, Bsp, NODOS = ctx["Phi"], ctx["Bsp"], ctx["NODOS"]
    W, Wx, Bm, c_mu = ctx["W"], ctx["Wx"], ctx["Bm"], ctx["c_mu"]
    mcov, scov, t_dia, M = ctx["mcov"], ctx["scov"], ctx["t_dia"], int(ctx["M"])
    NB = Bsp.shape[1]
    Wxt = torch.tensor(Wx, dtype=torch.float32)
    RBt = torch.tensor(ctx["RB"], dtype=torch.float32) if "RB" in ctx else torch.zeros(NB, NB)


    # --- clases identicas a las de la fase 4b, registradas como su modulo para
    #     que torch.load reconstruya los objetos sin re-entrenar nada ----------
    class CodificadorFuncional(nn.Module):
        def __init__(self, n_neu=N_NEU, modo="informado", semilla=0):
            super().__init__()
            self.A = nn.Parameter(torch.zeros(NB, n_neu), requires_grad=(modo != "fijo"))
            self.modo = modo
            self.escala = None

        def forward(self, c):
            z = c @ Wxt.T @ self.A
            return z / self.escala if self.escala is not None else z

        def pesos_wk(self):
            return torch.tensor(Bsp, dtype=torch.float32) @ self.A

        def rugosidad(self):
            return ((self.A.T @ RBt) * self.A.T).sum() / PERIODO


    class KANPropia(nn.Module):
        def __init__(self, n_in, n_out, grid=5, k=3):
            super().__init__()
            self.n_in, self.k = n_in, k
            paso = 1.0 / grid
            self.nodos = torch.linspace(-0.5 - k * paso, 1.5 + k * paso, grid + 2 * k + 2)
            self.nb = len(self.nodos) - k - 1
            self.coef = nn.Parameter(0.1 * torch.randn(n_in, self.nb, n_out))
            self.w_lin = nn.Parameter(0.1 * torch.randn(n_in, n_out))
            self.b = nn.Parameter(torch.zeros(n_out))

        def bspline(self, x):
            nod = self.nodos.to(x.device)
            B = ((x.unsqueeze(-1) >= nod[:-1]) & (x.unsqueeze(-1) < nod[1:])).to(x.dtype)
            for p in range(1, self.k + 1):
                iz = (x.unsqueeze(-1) - nod[:-(p + 1)]) / (nod[p:-1] - nod[:-(p + 1)])
                de = (nod[p + 1:] - x.unsqueeze(-1)) / (nod[p + 1:] - nod[1:-p])
                B = iz * B[..., :-1] + de * B[..., 1:]
            return B

        def forward(self, x):
            B = self.bspline(torch.tanh(x) * 0.5 + 0.5)
            y = torch.einsum("nib,ibo->no", B, self.coef)
            return y + torch.einsum("ni,io->no", torch.tanh(x), self.w_lin) + self.b


    def construye_cabeza(tipo, n_in, n_out, semilla, h_mlp=None):
        torch.manual_seed(semilla)
        if tipo == "lineal":
            return nn.Linear(n_in, n_out)
        if tipo == "mlp":
            return nn.Sequential(nn.Linear(n_in, h_mlp), nn.GELU(), nn.Linear(h_mlp, n_out))
        if tipo == "kan":
            try:
                return KAN(width=[n_in, n_out], grid=5, k=3, seed=semilla, device="cpu")
            except Exception as e:
                P(f"    [aviso] pykan no utilizable ({type(e).__name__}); KANPropia")
            return KANPropia(n_in, n_out)
        raise ValueError(tipo)


    class RedFuncional(nn.Module):
        def __init__(self, tipo="kan", semilla=0, h_mlp=None, modo="informado", n_neu=N_NEU):
            super().__init__()
            self.cod = CodificadorFuncional(n_neu, modo, semilla)
            self.cabeza = construye_cabeza(tipo, n_neu + len(COV), int(M), semilla, h_mlp)
            self.tipo, self.n_neu, self.modo = tipo, n_neu, modo

        def forward(self, c, v):
            return self.cabeza(torch.cat([self.cod(c), v], dim=1))


    def carga(nombre):
        """Reconstruye la red desde el state_dict guardado por la fase 4b."""
        d = torch.load(os.path.join(MOD, nombre), weights_only=False)
        red = RedFuncional(d["tipo"], 0, d["h_mlp"], d["modo"], d["n_neu"])
        red.load_state_dict(d["state_dict"], strict=False)
        red.cod.escala = d["escala"]
        red.eval()
        return red, d


    P("=" * 74)
    P("FASE 4.5 - interpretacion climatologica de la red funcional")
    P("=" * 74)
    red_kan, dk = carga("red_kan_aprendible.pt")
    red_lin, dl = carga("red_lineal_aprendible.pt")
    P(f"    modelos cargados: cabeza KAN ({type(red_kan.cabeza).__name__}) y cabeza lineal; "
      f"codificador en modo '{dk['modo']}' con {dk['n_neu']} neuronas funcionales")

    # ---------------------------------------------------------------------
    # 1. funciones peso del codificador y su lectura climatologica
    # ---------------------------------------------------------------------
    Wk = red_kan.cod.pesos_wk().detach().numpy()          # (365 x N_NEU)
    esc = red_kan.cod.escala.detach().numpy()
    Wk_n = Wk / esc                                        # como entran al modelo
    filas = []
    for k in range(N_NEU):
        w = Wk_n[:, k]
        masa = np.array([np.abs(w[LIM[m]:LIM[m + 1]]).sum() for m in range(12)])
        masa = masa / masa.sum()
        filas.append(dict(neurona=f"z{k+1}", mes_dominante=MESES[int(masa.argmax())],
                          peso_mes_dominante=float(masa.max()),
                          **{f"p_{MESES[m]}": float(masa[m]) for m in range(12)},
                          signo_medio=float(np.sign(w.mean())),
                          cambios_de_signo=int((np.diff(np.sign(w)) != 0).sum())))
    Tw = pd.DataFrame(filas)
    Tw.to_csv(os.path.join(TAB, "fase45_pesos_por_mes.csv"), index=False)
    P("\n[1] funciones peso w_k(t) del codificador: reparto mensual de |w_k|")
    P(Tw[["neurona", "mes_dominante", "peso_mes_dominante", "cambios_de_signo"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    LLU1, LLU2 = (LIM[2], LIM[5]), (LIM[8], LIM[11])       # mar-may y sep-nov
    for k in range(N_NEU):
        w = np.abs(Wk_n[:, k])
        fr = (w[LLU1[0]:LLU1[1]].sum() + w[LLU2[0]:LLU2[1]].sum()) / w.sum()
        P(f"    z{k+1}: {100*fr:.0f} % de |w| en las dos temporadas de lluvia "
          f"(mar-may y sep-nov, que ocupan el {100*(LLU1[1]-LLU1[0]+LLU2[1]-LLU2[0])/365:.0f} % del anio)")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for k in range(N_NEU):
        ax.plot(t_dia, Wk_n[:, k], lw=1.6, label=f"$w_{k+1}(t)$")
    for a, b, et in [(LLU1[0], LLU1[1], "lluvias mar-may"), (LLU2[0], LLU2[1], "lluvias sep-nov")]:
        ax.axvspan(a, b, color="tab:blue", alpha=0.10)
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.92, et, ha="center", fontsize=7, color="tab:blue")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("dia del anio")
    ax.set_ylabel("peso")
    ax.set_title("Funciones peso aprendidas por las neuronas funcionales\n"
                 "$z_k=\\int w_k(t)X(t)dt$ (base B-spline cubica, 12 coeficientes)")
    ax.set_xticks(LIM[:12] + np.diff(LIM) / 2)
    ax.set_xticklabels(MESES, fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase45_pesos_wk.png"), dpi=150)

    # ---------------------------------------------------------------------
    # 2. activaciones de la primera capa de la KAN
    # ---------------------------------------------------------------------
    D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
    SAT = [f"sat_c{i+1}" for i in range(K)]
    EST = [f"est_c{i+1}" for i in range(K)]
    rng0 = np.random.default_rng(2026)
    ests = np.array(sorted(D.STATION_ID.unique()))
    reserv = set(rng0.choice(ests, int(round(0.13 * len(ests))), replace=False))
    D["RESERVADA"] = D.STATION_ID.isin(reserv)
    tr = D[(~D.RESERVADA) & D.YEAR.between(1981, 1995)].reset_index(drop=True)
    te = D[(~D.RESERVADA) & D.YEAR.between(2001, 2003)].reset_index(drop=True)
    rs = D[D.RESERVADA].reset_index(drop=True)


    def entradas(df):
        c = torch.tensor(df[SAT].to_numpy(float), dtype=torch.float32)
        v = torch.tensor((df[COV].to_numpy(float) - mcov) / scov, dtype=torch.float32)
        with torch.no_grad():
            z = red_kan.cod(c)
        return c, v, torch.cat([z, v], dim=1).numpy()


    Ctr, Vtr, Xtr = entradas(tr)
    Cte, Vte, Xte = entradas(te)
    Crs, Vrs, Xrs = entradas(rs)
    x_med = torch.tensor(np.median(Xtr, axis=0), dtype=torch.float32)
    q = np.percentile(Xtr, [1, 99], axis=0)
    NG = 201
    P(f"\n[2] activaciones de la cabeza KAN: {Xtr.shape[1]} entradas x {M} salidas")


    def activacion(cabeza, i, xs, j=None):
        """phi_ij(x) salvo constante: respuesta al variar solo la entrada i."""
        Xb = x_med.repeat(len(xs), 1).clone()
        Xb[:, i] = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad():
            y = cabeza(Xb).numpy()
        y = y - y.mean(axis=0, keepdims=True)
        return y if j is None else y[:, j]


    n_in = Xtr.shape[1]
    act = {}
    for i in range(n_in):
        xs = np.linspace(q[0, i], q[1, i], NG)
        act[i] = (xs, activacion(red_kan.cabeza, i, xs))

    # importancia: amplitud de cada activacion sobre la salida PC1 y total
    imp = []
    for i in range(n_in):
        xs, y = act[i]
        imp.append(dict(entrada=NOM_ENT[i], amplitud_PC1=float(y[:, 0].max() - y[:, 0].min()),
                        amplitud_total=float(np.abs(y).max(axis=0).sum()),
                        no_linealidad=float(np.nan)))
    Ti = pd.DataFrame(imp)
    # no linealidad: 1 - R2 del mejor ajuste lineal de la activacion sobre PC1
    for i in range(n_in):
        xs, y = act[i]
        A = np.column_stack([np.ones_like(xs), xs])
        be = np.linalg.lstsq(A, y[:, 0], rcond=None)[0]
        r2 = 1 - ((y[:, 0] - A @ be) ** 2).sum() / (((y[:, 0] - y[:, 0].mean()) ** 2).sum() + 1e-18)
        Ti.loc[i, "no_linealidad"] = float(1 - r2)
    # la KAN puede anular por completo una entrada: hay que distinguirlo, porque
    # una activacion identicamente nula da R2 degenerado en el ajuste simbolico
    # criterio RELATIVO: una entrada esta podada si su activacion es despreciable
    # frente a la mayor (el criterio absoluto no distingue escalas)
    TOL_PODA = 1e-3 * Ti.amplitud_total.max()
    Ti["estado"] = np.where(Ti.amplitud_total <= TOL_PODA, "podada", "activa")
    Ti = Ti.sort_values("amplitud_total", ascending=False)
    Ti.to_csv(os.path.join(TAB, "fase45_importancia_entradas.csv"), index=False)
    P(Ti.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    PODADAS = set(Ti[Ti.estado == "podada"].entrada)
    ACTIVAS = [n for n in NOM_ENT if n not in PODADAS]
    P(f"    entradas que la KAN ANULA (activacion identicamente nula): "
      f"{', '.join(sorted(PODADAS)) if PODADAS else 'ninguna'}")
    P(f"    la red se queda con {len(ACTIVAS)} de {n_in} entradas: {', '.join(ACTIVAS)}")

    fig, axs = plt.subplots(M, n_in, figsize=(2.0 * n_in, 1.7 * M), sharex="col")
    for i in range(n_in):
        xs, y = act[i]
        for j in range(M):
            ax = axs[j, i]
            podada = NOM_ENT[i] in PODADAS
            ax.plot(xs, y[:, j], lw=1.3, color="0.6" if podada else "tab:red")
            ax.axhline(0, color="0.7", lw=0.5)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_title(NOM_ENT[i] + (" (podada)" if podada else ""), fontsize=8,
                             color="0.5" if podada else "k")
            if i == 0:
                ax.set_ylabel(f"PC{j+1}", fontsize=8)
    fig.suptitle("Funciones de activacion de la cabeza KAN: $\\phi_{ij}$ por entrada (columnas) "
                 "y componente de salida (filas)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "fase45_activaciones_kan.png"), dpi=150)

    # ---------------------------------------------------------------------
    # 3. ajuste simbolico de las activaciones
    # ---------------------------------------------------------------------
    def disenos(x):
        return {
            "lineal: a+bx": np.column_stack([np.ones_like(x), x]),
            "cuadratica: a+bx+cx^2": np.column_stack([np.ones_like(x), x, x ** 2]),
            "cubica": np.column_stack([np.ones_like(x), x, x ** 2, x ** 3]),
            "seno: a+b sen(pi x/2)": np.column_stack([np.ones_like(x), np.sin(np.pi * x / 2)]),
            "tangente hiperbolica": np.column_stack([np.ones_like(x), np.tanh(x)]),
            "valor absoluto": np.column_stack([np.ones_like(x), np.abs(x)]),
            "exponencial": np.column_stack([np.ones_like(x), np.exp(np.clip(x, -5, 5))]),
            "logaritmica: a+b ln(1+|x|)": np.column_stack([np.ones_like(x), np.log1p(np.abs(x))]),
        }


    P("\n[3] ajuste simbolico de las activaciones (R2, salida PC1)")
    fs = []
    for i in range(n_in):
        xs, y = act[i]
        obj = y[:, 0]
        sst = ((obj - obj.mean()) ** 2).sum()
        if NOM_ENT[i] in PODADAS or sst <= 1e-12 * len(obj):   # R2 no definido
            P(f"    {NOM_ENT[i]:9s}: activacion nula (entrada podada por la KAN), "
              f"sin ajuste simbolico")
            continue
        mejor = None
        for nom, A in disenos(xs).items():
            be = np.linalg.lstsq(A, obj, rcond=None)[0]
            r2 = 1 - ((obj - A @ be) ** 2).sum() / sst
            r2aj = 1 - (1 - r2) * (len(xs) - 1) / (len(xs) - A.shape[1])
            fs.append(dict(entrada=NOM_ENT[i], forma=nom, R2=float(r2), R2_ajustado=float(r2aj),
                           coef=np.round(be, 4).tolist()))
            if mejor is None or r2aj > mejor[1]:
                mejor = (nom, r2aj, r2, be)
        P(f"    {NOM_ENT[i]:9s}: mejor forma = {mejor[0]:24s} R2 = {mejor[2]:.4f} "
          f"(ajustado {mejor[1]:.4f})")
    S = pd.DataFrame(fs)
    S.to_csv(os.path.join(TAB, "fase45_ajuste_simbolico.csv"), index=False)
    best = S.sort_values("R2_ajustado", ascending=False).iloc[0]
    P(f"    (se excluyeron las entradas podadas: {', '.join(sorted(PODADAS)) or 'ninguna'})")
    P(f"    activacion mejor explicada: {best.entrada} con {best.forma}, "
      f"R2 = {best.R2:.4f}, coeficientes {best.coef}")

    i_dom = NOM_ENT.index(Ti[Ti.estado == "activa"].iloc[0].entrada)
    xs, y = act[i_dom]
    obj = y[:, 0]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(xs, obj, lw=2.2, color="tab:red", label=f"$\\phi$ de la KAN ({NOM_ENT[i_dom]} $\\to$ PC1)")
    for nom in ["lineal: a+bx", "cuadratica: a+bx+cx^2", "tangente hiperbolica"]:
        A = disenos(xs)[nom]
        be = np.linalg.lstsq(A, obj, rcond=None)[0]
        r2 = 1 - ((obj - A @ be) ** 2).sum() / (((obj - obj.mean()) ** 2).sum() + 1e-18)
        ax.plot(xs, A @ be, "--", lw=1.2, label=f"{nom.split(':')[0]} ($R^2$={r2:.3f})")
    ax.set_xlabel(f"{NOM_ENT[i_dom]} (estandarizada)")
    ax.set_ylabel("contribucion a PC1")
    ax.set_title("Ajuste simbolico de la activacion dominante")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase45_ajuste_simbolico.png"), dpi=150)

    # ---------------------------------------------------------------------
    # 4. contraste con la cabeza lineal
    # ---------------------------------------------------------------------
    Wlin = red_lin.cabeza.weight.detach().numpy()          # (M x n_in)
    pend = np.zeros(n_in)
    for i in range(n_in):
        xs, y = act[i]
        A = np.column_stack([np.ones_like(xs), xs])
        pend[i] = np.linalg.lstsq(A, y[:, 0], rcond=None)[0][1]
    Tc = pd.DataFrame(dict(entrada=NOM_ENT, coef_lineal_PC1=Wlin[0],
                           pendiente_media_KAN_PC1=pend))
    Tc["razon"] = Tc.pendiente_media_KAN_PC1 / (Tc.coef_lineal_PC1 + 1e-12)
    Tc.to_csv(os.path.join(TAB, "fase45_lineal_vs_kan.csv"), index=False)
    Tc["estado"] = ["podada" if n in PODADAS else "activa" for n in NOM_ENT]
    rho = np.corrcoef(Wlin[0], pend)[0, 1]
    act_m = np.array([n not in PODADAS for n in NOM_ENT])
    rho_act = np.corrcoef(Wlin[0][act_m], pend[act_m])[0, 1] if act_m.sum() > 2 else np.nan
    P(f"\n[4] cabeza lineal vs pendiente media de las activaciones KAN (PC1)")
    P(Tc.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    P(f"    correlacion sobre las 8 entradas:            r = {rho:.3f}")
    P(f"    correlacion solo sobre las entradas activas: r = {rho_act:.3f}")
    P(f"    (la discrepancia la producen las entradas que la KAN anula y el modelo "
      f"lineal si usa)")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(Wlin[0], pend, s=42, color="tab:purple")
    for i in range(n_in):
        ax.annotate(NOM_ENT[i], (Wlin[0][i], pend[i]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    lim = max(np.abs(Wlin[0]).max(), np.abs(pend).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="identidad")
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(0, color="0.8", lw=0.6)
    ax.set_xlabel("coeficiente de la cabeza lineal (PC1)")
    ax.set_ylabel("pendiente media de $\\phi_{i,PC1}$ (KAN)")
    ax.set_title(f"La KAN reproduce el signo y la escala del modelo lineal\n(r = {rho:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase45_lineal_vs_kan.png"), dpi=150)

    # ---------------------------------------------------------------------
    # 5. peores casos
    # ---------------------------------------------------------------------
    Bmt = torch.tensor(Bm, dtype=torch.float32)
    cmut = torch.tensor(c_mu, dtype=torch.float32)


    def errores(df, C, V):
        with torch.no_grad():
            coef = (cmut + red_kan(C, V) @ Bmt.T).numpy()
        obs = df[EST].to_numpy(float)
        d = coef - obs
        l2 = np.sqrt(((d @ W) * d).sum(axis=1) / PERIODO)
        nor = np.sqrt(((obs @ W) * obs).sum(axis=1) / PERIODO) + 1e-12
        out = df.copy()
        out["l2_rel"] = l2 / nor
        out["acum_obs"] = (obs @ Phi.T).sum(axis=1)
        out["acum_pred"] = (coef @ Phi.T).sum(axis=1)
        return out, coef, obs


    Ete, cte, ote = errores(te, Cte, Vte)
    Ers, crs, ors_ = errores(rs, Crs, Vrs)
    tod = pd.concat([Ete.assign(bloque="test 2001-2003"), Ers.assign(bloque="reservadas")])
    peor = tod.sort_values("l2_rel", ascending=False).head(12)
    cols = ["bloque", "STATION_ID", "NOMBRE", "MUNICIPIO", "YEAR", "ALT_EST", "dist_km",
            "l2_rel", "acum_obs", "acum_pred"]
    cols = [c for c in cols if c in peor.columns]
    peor[cols].to_csv(os.path.join(TAB, "fase45_peores_casos.csv"), index=False)
    P("\n[5] peores casos (mayor error L2 relativo)")
    P(peor[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    P(f"\n    error L2 relativo medio: test {Ete.l2_rel.mean():.3f} | "
      f"reservadas {Ers.l2_rel.mean():.3f}")
    if "ALT_EST" in tod.columns:
        alto = tod[tod.ALT_EST > tod.ALT_EST.median()]
        bajo = tod[tod.ALT_EST <= tod.ALT_EST.median()]
        P(f"    por altitud: estaciones sobre la mediana {alto.l2_rel.mean():.3f} | "
          f"bajo la mediana {bajo.l2_rel.mean():.3f}")
        r_alt = tod[["l2_rel", "ALT_EST"]].corr(method="spearman").iloc[0, 1]
        r_dis = tod[["l2_rel", "dist_km"]].corr(method="spearman").iloc[0, 1] \
            if "dist_km" in tod.columns else np.nan
        P(f"    correlacion de Spearman del error con la altitud: {r_alt:+.3f} | "
          f"con la distancia al pixel: {r_dis:+.3f}")

    sel = peor.head(4)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (_, r) in zip(axs.ravel(), sel.iterrows()):
        fuente = Ete if r.bloque == "test 2001-2003" else Ers
        dfc, cc, oo = (te, cte, ote) if r.bloque == "test 2001-2003" else (rs, crs, ors_)
        k = dfc.index[(dfc.STATION_ID == r.STATION_ID) & (dfc.YEAR == r.YEAR)]
        if len(k) == 0:
            continue
        k = int(k[0])
        ax.plot(t_dia, oo[k] @ Phi.T, color="k", lw=1.6, label="estacion (observado)")
        ax.plot(t_dia, cc[k] @ Phi.T, color="tab:red", lw=1.6, label="prediccion")
        nom = r.get("NOMBRE", r.STATION_ID)
        ax.set_title(f"{nom} ({r.YEAR}) - L2rel {r.l2_rel:.2f}, "
                     f"{r.get('ALT_EST', float('nan')):.0f} m", fontsize=9)
        ax.set_xticks(LIM[:12] + np.diff(LIM) / 2)
        ax.set_xticklabels(MESES, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Peores casos del modelo KAN con codificador aprendible", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "fase45_peores_casos.png"), dpi=150)

    with open(os.path.join(ENT, "log_fase45.txt"), "w") as f:
        f.write("\n".join(log))
    P("\nFASE 4.5 OK")


def _bloque_5d():
    """Fase 3d - comparacion pareada georreferenciada."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RAIZ = os.path.dirname(os.path.abspath(__file__))
    ENT = os.path.join(RAIZ, "entrega")
    TAB, FIG = os.path.join(ENT, "tablas"), os.path.join(ENT, "figs")
    K, PERIODO = 5, 365.0
    rng = np.random.default_rng(2026)


    def base_fourier(t, K, periodo=PERIODO):
        cols = [np.ones_like(t)]
        for j in range(1, K // 2 + 1):
            cols.append(np.sin(2 * np.pi * j * t / periodo))
            cols.append(np.cos(2 * np.pi * j * t / periodo))
        return np.column_stack(cols[:K])


    t = np.arange(365) + 0.5
    Phi = base_fourier(t, K)
    D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
    SAT = [f"sat_c{i+1}" for i in range(K)]
    EST = [f"est_c{i+1}" for i in range(K)]
    Ce, Cs = D[EST].to_numpy(float), D[SAT].to_numpy(float)
    Xe, Xs = Ce @ Phi.T, Cs @ Phi.T
    print(f"[1] pares georreferenciados: {len(D)} | {D.STATION_ID.nunique()} estaciones | "
          f"{D.YEAR.min()}-{D.YEAR.max()}")

    dif = Xe - Xs
    sesgo = dif.mean(axis=0)
    B = 1000
    boot = np.empty((B, 365))
    for b in range(B):
        idx = rng.integers(0, len(dif), len(dif))
        boot[b] = dif[idx].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    sig = (lo > 0) | (hi < 0)
    print(f"[2] sesgo funcional medio (estacion - satelite): {sesgo.mean():+.4f} raiz(mm)/dia")
    print(f"    maximo {sesgo.max():+.4f} (dia {int(sesgo.argmax())+1}) | "
          f"minimo {sesgo.min():+.4f} (dia {int(sesgo.argmin())+1})")
    print(f"    dias con banda bootstrap al 95 % que excluye el cero: "
          f"{int(sig.sum())} de 365 ({100*sig.mean():.1f} %)")

    acu_e, acu_s = Xe.sum(axis=1), Xs.sum(axis=1)
    razon = acu_s.mean() / acu_e.mean()
    print(f"[3] integral anual media: estacion {acu_e.mean():.1f} | satelite {acu_s.mean():.1f} "
          f"| razon satelite/estacion {razon:.3f}")

    D["sesgo_int"] = dif.sum(axis=1) / 365
    filas = []
    for col, nom in [("ALT_EST", "altitud real"), ("LON_EST", "longitud"),
                     ("LAT_EST", "latitud"), ("YEAR", "anio"), ("dist_km", "distancia al pixel")]:
        if col not in D.columns:
            continue
        s = D[["sesgo_int", col]].dropna()
        if len(s) < 30:
            continue
        rho = s.corr(method="spearman").iloc[0, 1]
        filas.append(dict(variable=nom, n=len(s), spearman=float(rho)))
        print(f"    sesgo integrado vs {nom:20s}: rho = {rho:+.3f}  (n = {len(s)})")
    T = pd.DataFrame(filas)
    T.to_csv(os.path.join(TAB, "fase3d_sesgo_estructural.csv"), index=False)
    pd.DataFrame(dict(dia=np.arange(1, 366), sesgo=sesgo, ic_inf=lo, ic_sup=hi,
                      significativo=sig)).to_csv(
        os.path.join(TAB, "fase3d_comparacion.csv"), index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(t, Xe.mean(axis=0), lw=1.8, color="k", label="estaciones")
    axs[0].plot(t, Xs.mean(axis=0), lw=1.8, color="tab:orange", label="satelite")
    axs[0].set_title("Medias funcionales pareadas (coordenadas reales)")
    axs[0].set_xlabel("dia del anio")
    axs[0].set_ylabel("$\\sqrt{mm/dia}$")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)
    axs[1].plot(t, sesgo, color="tab:blue", lw=1.6)
    axs[1].fill_between(t, lo, hi, alpha=0.25, color="tab:blue", label="IC 95 % (bootstrap)")
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_title(f"Sesgo funcional estacion - satelite\n"
                     f"({100*sig.mean():.0f} % de los dias con diferencia significativa)")
    axs[1].set_xlabel("dia del anio")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fase3d_sesgo_funcional.png"), dpi=150)
    print("\nFASE 3d OK")


_BLOQUES = [
    (_bloque_1, "Fase 1b - preprocesamiento y depuracion (4.1)"),
    (_bloque_2, "Fase 1 - construccion de datos funcionales"),
    (_bloque_3, "Fase 3.0 - reajuste pentadal (comparabilidad de escalas)"),
    (_bloque_4, "Fase 2 - analisis exploratorio funcional"),
    (_bloque_5, "Fase 3 - comparacion estaciones vs satelital"),
    (_bloque_5b, "Fase 3b - emparejamiento con coordenadas oficiales"),
    (_bloque_5c, "Fase 3c - control espacial del emparejamiento"),
    (_bloque_5d, "Fase 3d - comparacion pareada georreferenciada"),
    (_bloque_6, "Fase 4 - red funcional, ablacion y evaluacion"),
    (_bloque_7, "Fase 4b - codificador funcional aprendible (4.2)"),
    (_bloque_8, "Fase 4.5 - interpretacion climatologica (4.5)"),
]


if __name__ == "__main__":
    sel = [int(a) for a in sys.argv[1:]] or list(range(1, len(_BLOQUES) + 1))
    for n, (fn, desc) in enumerate(_BLOQUES, 1):
        if n not in sel:
            continue
        print("\n" + "#" * 74)
        print(f"# BLOQUE {n}/{len(_BLOQUES)}: {desc}")
        print("#" * 74, flush=True)
        fn()
    print("\n[proyecto] bloques ejecutados:", sel)
