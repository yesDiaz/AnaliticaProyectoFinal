# INVENTARIO DE DATOS — Proyecto final Analítica de Datos (precipitación, Valle del Cauca)

Generado el 21-nov-2026 a partir de las corridas guardadas en `cache/log_fase0_v2.txt` y las tablas de `tablas/`.
Todo número de este archivo sale de una tabla del pipeline; la tabla de origen va entre corchetes.

## 1. Fuentes

| Fuente | Archivo | Resolución | Periodo útil | Notas |
|---|---|---|---|---|
| Estaciones en tierra (red CVC) | `datos/P_1930_1979.xlsx`, `P_1980_1990.csv`, `P_1991_2003.csv` | diaria, mm/día, puntual | 1981-2003 (no hay datos de estación después de 2003) | 323 estaciones con algún dato en 1981-2003 [Tab02] |
| Coordenadas | `datos/CVC_Wheather_Stations_Long_Lat.csv` | lon/lat | — | 238 estaciones; 178 de las 323 caen dentro de la malla satelital |
| Altitud (catálogo) | `datos/CatalogoEstacionesElevacion.xlsx` | m s. n. m. | — | 229 registros |
| Altitud faltante | `entrega/datos_externos/srtm90m_estaciones.csv` | SRTM 90 m (3″) | — | OpenTopoData `srtm90m` consultado el 21-nov-2026 con las coordenadas CVC (ver `srtm90m_LEEME.txt`) |
| Satélite | `datos/ImagenesSatelitePentadal.zip` (GeoTIFF `pptAAAAPP.tif`) | péntada, 0,05°, 39×37 píxeles | 1981-2012 completos (2013 trae 24 péntadas) | **Total por péntada (mm/péntada)**, ver §4 |
| ENSO | `entrega/datos_externos/oni.ascii.txt` | trimestral móvil | 1950-2026 | ONI de NOAA-CPC; fase del año = media de las 12 estaciones (±0,5 °C) |

## 2. Inventario de estaciones idóneas (P1 #1)

- Tabla de **% de días faltantes por estación y por año** (calendario completo 1981-2003, 365 días; 0 mm cuenta como dato): **`tablas/Tab06_faltantes_pct_estacion_anio.csv`** (323 filas × 23 años, más región, coordenadas y % de datos del periodo). Figura: `figuras/Fig01_inventario_faltantes_trazabilidad.png`.
- Resumen por estación con la decisión de cada regla: `tablas/Tab06b_resumen_estaciones_reglas.csv`.

### Comparación de reglas [Tab06c]

| Regla | Regla ANUAL | Regla del PERIODO | Estaciones con coordenadas | Pacífico | Andina | Pares estación-año |
|---|---|---|---|---|---|---|
| OMM (como se citó en clase) | ≤10 faltantes y ≤5 seguidos | ≥80 % de años válidos (≥19 de 23) | 77 | 6 | 71 | 1634 |
| NOAA | igual que OMM | ≥10 años válidos | 141 | 20 | 121 | 2605 |
| **Nacional (ELEGIDA)** | **≤25 % de faltantes (se imputa)** | **>45 % de datos en 1981-2003** | **141** | **22** | **119** | **2945** |
| Universo con coordenadas | — | — | 178 | 28 | 150 | — |

- La regla OMM deja **6 de 28** estaciones del Pacífico; la elegida deja **22 de 28**. Es la razón principal de la elección (cobertura de todo el departamento y no solo de la zona andina), en línea con lo que recomendó el profesor el 20-nov.
- La regla anual y la del periodo se aplican por separado: primero se decide si la **estación** entra (periodo), y luego, dentro de ella, qué **años** se usan (anual, ≤25 % de faltantes). Los años con más del 25 % de faltantes se descartan y **no** se imputan (el profesor: entre 25 y 40 % ya no se rescata).
- ⚠️ La fuente del criterio "nacional >45 %" que se mencionó en clase **no se pudo verificar** (ver `REFERENCIAS_VERIFICADAS.md`, sección "No encontrado"). En el informe se sustenta con el umbral anual del profesor (25 %), la cobertura del Pacífico y umbrales colombianos sí verificados (IDEAM 2018: 15 % por mes calendario; Cruz-Roa y Barrios 2018: <20 %).
- La OMM (WMO-No. 1203, §4.4.1) aplica la regla de 10/5 días **por mes**, no por año; en clase se citó como regla anual. El inventario usa la versión anual citada en clase y lo dice explícitamente.

## 3. Control de calidad y trazabilidad [Tab02]

| Paso | Estaciones | Pares estación-año | Días válidos | Afectados |
|---|---|---|---|---|
| P0 crudo 1981-2003 (29-feb excluido) | 323 | 5514 | 1 929 453 | — |
| P1 valores imposibles (<0, no numérico, >1000) | 323 | 5514 | 1 929 452 | 1 |
| P2 totales mensuales en un solo día | 323 | 5514 | 1 929 121 | 331 |
| P3 errores de digitación | 323 | 5514 | 1 929 116 | 5 |
| P3b meses con escala errónea | 323 | 5514 | 1 929 087 | 29 |
| P3c bloques de 10 días copiados | 323 | 5511 | 1 926 943 | 2144 |
| P4 rachas ≥30 ceros con 2 vecinas ≤25 km | 323 | 5511 | 1 924 909 | 2034 |
| P6 regla del periodo (>45 % y con coordenadas) | 141 | 3057 | 1 082 653 | 182 estaciones |
| P7 regla anual (≤25 % faltantes) | 141 | 2945 | 1 062 509 | 112 pares |
| P8 imputación (CSW; respaldo kriging 40 km) | 141 | 2945 | 1 074 820 | 12 311 días |

## 4. Cero ≠ faltante (P1 #3) [Tab02c]

- 0 mm es dato (no llovió). En el código ningún paso convierte un cero en NaN por ser cero. Prueba automática (`assert` en `fase0`): todo cero del crudo que terminó en NaN debe tener una bandera documentada. Resultado: **0 ceros sin regla**.
- De 1 126 184 ceros del crudo 1981-2003 se conservan 1 123 213. Los 2971 que pasan a NaN vienen de: rachas declaradas falla (2034), bloques copiados (649) y meses de "total mensual en un día" (288).
- Se conservan rachas secas largas: la más larga, **96 días** (estación 2613000102, 1982, año Niño); 49 estaciones conservan alguna racha seca de 60 días o más.

## 5. Rachas de 30 días en cero (P1 #4) [Tab05]

- 889 rachas de ≥30 días en 0 mm. Se comparan con las **2 estaciones más cercanas a ≤25 km** que tengan dato en ≥80 % de esos días.
- Falla del sensor (→ NaN → se imputa) si las vecinas llovieron: media ≥1 mm/día y ≥20 % de días con >1 mm. Si no, es sequía real y se conserva. Sin vecinas: falla solo con "alzas" (≥10 mm en los 5 días antes **y** en los 5 días después).
- Resultado: **49 fallas** (20 de ellas también con alzas antes y después) y **840 sequías reales** conservadas. Ejemplos en `figuras/Fig02_rachas_cero_vecinas.png`.

## 6. Altitud (P1 #7) [Tab06d]

- 87 estaciones con coordenadas no tienen altitud en el catálogo; se toma del SRTM 90 m (38 de ellas quedaron entre las 141 elegidas).
- Validación con las **126** estaciones que tienen las dos altitudes: sesgo +54 m, MAE 129 m, RMSE 276 m, mediana del error absoluto 36,5 m, r = 0,93, 70,6 % dentro de ±100 m, error máximo 1660 m. No se verificó la causa de los errores grandes de unas pocas estaciones (posibles: coordenadas imprecisas en ladera, píxel de 90 m en pendiente fuerte o error del catálogo); queda como limitación.

## 7. Relación lluvia-altura (P1 #6) [Tab06e]

- Total anual medio vs altitud (141 estaciones): **no lineal**. R² lineal 0,06 vs cuadrático 0,24 (término cuadrático p = 4·10⁻⁸).
- Pacífico: la lluvia **baja** con la altura (−210 mm por cada 100 m; R² lineal 0,40, cuadrático 0,75). Andina: sube débilmente (+17 mm/100 m; Spearman 0,35).
- Correlación diaria entre estaciones (1981-1996) [Tab06f]: a igual distancia baja con la diferencia de altura (0-10 km: r = 0,46 con |Δz| ≤400 m, 0,37 con |Δz| >900 m).

## 8. Imputación (P1 #5 y #6) [Tab06g, Tab06h]

- Discriminación previa: 1) |Δz| ≤ 400 m (distancia en el eje vertical); 2) distancia horizontal ≤ 20 km (radio donde la correlación diaria media de pares con |Δz| ≤400 m sigue ≥0,30). Donantes: estaciones con dato **ese mismo día** (máx. 6).
- Métodos: IDW (1/d²), CSW (peso = r² de la correlación diaria 1981-1996) y kriging ordinario (correlograma exponencial r(h) = 0,345·e^(−h/201,5 km), pepita 0,655; radio 40 km).
- Validación dejar-uno-fuera en 1981-1996 (729 282 días):

| Método | RMSE diario (mm) | r diaria | RMSE total 30 d (mm) | sesgo total 30 d | Cobertura |
|---|---|---|---|---|---|
| IDW | 9,45 | 0,532 | 69,2 | +1,3 % | 89,3 % |
| **CSW** | 9,12 | 0,547 | **67,2** | **+0,1 %** | 89,3 % |
| Kriging ordinario | **8,94** | **0,549** | 69,7 | −2,3 % | 98,0 % |

- Se elige CSW por el menor error en el total de 30 días (la escala que conserva la curva funcional con K = 6) y el menor sesgo. El kriging, con su radio doble, se usa como respaldo donde no hay donantes a 20 km.
- Imputado: 1,13 % de los días de los pares elegidos en la zona andina y 1,23 % en el Pacífico; quedan 105 días sin donantes (NaN, la fase 1 los omite en los mínimos cuadrados).

## 9. Resolución temporal y satélite (P1 #8 y #9) [Tab06i]

- Se trabaja con **lluvia diaria puntual** de estación (no acumulada): curva anual en [0, 365] con K = 6 B-splines, ajustada sobre la raíz de la lluvia diaria en la fase 4. No se usa la lluvia acumulada en ningún paso.
- Satélite: GeoTIFF enteros, 0,05°, **sin valores −9999** en el recorte entregado, con un **tope en 200 mm/péntada** (0,24 % de los valores son exactamente 200: censura del recorte). Comparación con 208 910 péntadas de estación completas en el píxel más cercano: media del satélite 25,4 mm frente a 24,4 mm de **suma** de la estación y 4,8 mm/día de **promedio** (razón 1,04 con la suma y 5,3 con el promedio). Por tanto el valor es **total por péntada (mm/péntada)**, como dice la cabecera oficial de CHIRPS (`units = "mm/pentad"`). En clase se dijo que era promedio; los datos muestran lo contrario. El pipeline divide cada péntada entre su número de días (5, o 3-6 la última de cada mes) para llevarla a mm/día.
- Denominador de las reglas: con lluvia diaria es 365 días por año. Si se agregara a péntadas serían 72 por año y "5 seguidos" serían 25 días; por eso las reglas se aplican sobre el dato diario, antes de cualquier agregación.

## 10. Partición (fase 4)

- Entrenamiento + validación: 1981-1996, con validación cruzada temporal de ventana creciente (entrena hasta 1988/90/92/94 y valida en los 2 años siguientes).
- Prueba: 1997-2003 en tres tramos: T1 Niño 1997-1998, T2 Niña 1999-2000, T3 neutro 2001-2003 (ONI anual: 1997 +1,14; 1998 +0,04; 1999 −1,08; 2000 −0,77; 2001 −0,29; 2002 +0,51; 2003 +0,24). Por el ONI, 2002 fue un Niño moderado, aunque en clase se dijo "neutro".
- Estaciones reservadas: el docente no entregó lista; se sortean 14 estaciones (10 %) con semilla 2026 (`tablas/Tab42b_estaciones_reservadas.csv`).
