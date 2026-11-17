# Fase 3 — Comparación estación terrestre vs producto satelital

Proyecto Final de Analítica de Datos · UAO 2026-2S
Scripts: `fase3_0_pentadas.py` (reajuste de escala) y `fase3_comparacion.py`
Log: `entrega/log_fase3.txt` · Figuras: `figs/fase3_*.png` · Tablas: `tablas/fase3_*.csv`

---

## 1. El problema que había que resolver primero: las coordenadas

El proyecto no entrega catálogo de estaciones y los 186 `STATION_ID` de 10 dígitos **no
coinciden** con el campo `Codigo` del Catálogo Nacional de Estaciones del IDEAM (CNE,
`datos.gov.co` / `hp9r-jxuu`, 29.320 registros descargados): 0 de 186 emparejan de forma
literal, y la mejor regla de reescritura probada (134/186) resulta espuria.

Lo que **sí** es válido es la estructura del código: **los 4 primeros dígitos son la subzona
hidrográfica del IDEAM**, y eso se verifica contra el CNE (429 estaciones del CNE en la
subzona 2610, 513 en 2612, 398 en 2613…). De ahí la estrategia de esta fase, en tres pasos
auditables:

1. **Coordenada aproximada** = centroide de la subzona según el CNE, con su dispersión como
   incertidumbre declarada: **±10 km en longitud, ±24 km en latitud** (mediana).
2. **Universo de análisis** = solo las estaciones cuya subzona cae dentro de la malla satelital.
3. **Emparejamiento** = entre los píxeles a menos de 35 km del centroide, el de máxima
   correlación de **anomalías** mensuales.

### Hallazgo 1 — las dos fuentes del proyecto no cubren la misma región

| Subzona | n en CNE | Centroide | Altitud | Departamento | ¿En la malla? |
|---|---|---|---|---|---|
| 2610 | 429 | (−76,100, 4,267) | 952 m | Valle del Cauca | **sí** |
| 2611 | 210 | (−76,100, 4,533) | 975 m | Valle del Cauca | **sí** |
| 2612 | 513 | (−75,717, 4,556) | 1.480 m | Quindío | **sí** |
| 2613 | 398 | (−75,652, 4,850) | 1.520 m | Risaralda | **sí** |
| 2614, 2615 | 755 | lat 5,05–5,15 | 1.430–2.032 m | Risaralda, Caldas | no |
| 2620–2625 | 590 | lat 6,06–7,69 | 88–1.739 m | Antioquia | no |
| 5301 | 6 | (−78,411, 2,519) | 115 m | Nariño | no |
| 5300, 5400, 5420, 5421 | — | sin registros en el CNE | — | Patía / Pacífico sur | no |

**Solo 65 de las 186 estaciones (35 %) están dentro del dominio del producto satelital**
(lon −77,525 a −75,725; lat 3,075 a 4,975). El resto pertenece al Cauca medio-bajo
(Antioquia, hasta lat 7,7°) y al Patía/Pacífico sur (hasta lon −78,4°). La comparación
pareada solo es legítima en la intersección de dominios, y eso explica además las curvas
atípicas del Pacífico detectadas en la Fase 2: son estaciones que el satelital de este
proyecto nunca ve.

---

## 2. El segundo problema: las dos fuentes no estaban en la misma escala

La Fase 1 ajustó las estaciones sobre `√P` de valores **diarios** y el satelital sobre `√P`
de promedios **pentadales** (72 pentadas/año, mm/día). La raíz es cóncava, así que por la
desigualdad de Jensen

```
media_5días( √P )  ≤  √( media_5días(P) )
```

y la brecha crece con la intermitencia: una estación con días secos y chubascos la sufre
mucho más que un producto satelital ya suavizado. Comparar las dos fuentes en esa escala
**no mide el sensor, mide el preprocesamiento**.

Corrección aplicada (`fase3_0_pentadas.py`): se reconstruyeron los coeficientes de estación
con **exactamente el mismo preprocesamiento del satelital** — agregar a 72 pentadas, pasar a
mm/día, raíz, y ajustar la misma base de Fourier K=5 por mínimos cuadrados ponderados por
días observados, con **Cholesky propia** (sin `lm()`, sin `lstsq`). 1.299.927 registros
diarios → 256.409 pentadas válidas (≥3 días con dato) → 3.567 curvas reajustadas
(R² mediano 0,184 sobre 72 puntos, RMSE 1,09 √(mm/día)).

### Hallazgo 2 — la mitad del desacuerdo entre fuentes era artefacto de representación

| Representación | Integral anual media | Sesgo vs acumulado observado |
|---|---|---|
| Ajuste **diario** (Fase 1) | 810 mm/año | **−58,2 %** |
| Ajuste **pentadal** (Fase 3) | 1.483 mm/año | **−23,5 %** |
| Acumulado observado real | 1.939 mm/año | — |

Razón pentadal/diario = **2,14**. Trasladado a la comparación con el satélite: la razón
satélite/estación pasa de **3,32** (escala diaria) a **1,62** (escala pentadal), es decir
**el 51 % del desacuerdo original era efecto Jensen, no sesgo del producto satelital**.

---

## 3. Emparejamiento estación–píxel y su validación

- 65 estaciones emparejadas sobre 25 píxeles distintos; 67 píxeles candidatos por estación
  (mediana), distancia al centroide 26,2 km (mediana).
- Correlación de anomalías del píxel elegido: **mediana 0,526** (rango 0,158–0,674).
- **Control de aleatoriedad**: correlación con píxeles aleatorios de la malla
  **0,274 ± 0,179** → el emparejamiento gana **+0,245** de correlación. La señal es real
  pero no identifica la estación de forma unívoca: la asignación es *informativa*, no
  *exacta*, y así debe leerse todo lo que sigue.
- Figura `figs/fase3_emparejamiento.png`: mapa de centroides y píxeles asignados,
  distribución nula vs asignada, y totales pareados.

Por qué anomalías y no la serie directa: el ciclo estacional bimodal es casi idéntico en
toda la región (correlaciones >0,8 con cualquier píxel), de modo que no discrimina posición.
La variabilidad interanual sí es local, y es la que se usa.

---

## 4. Comparación funcional pareada (1.243 pares estación-año)

| Resultado | Valor |
|---|---|
| Sesgo medio integrado (estación − satélite) | **−0,497** √(mm)/día |
| Sesgo máximo / mínimo | −0,221 (día 20) / −0,696 (día 156) |
| Fracción del año con diferencia significativa al 95 % (bootstrap 1000) | **100 %** |
| RMSE funcional medio / máximo | 0,720 / 0,867 (día 179) |
| Correlación entre fuentes día a día | mediana 0,373 (0,174–0,538) |
| Integral anual: estación vs satélite | 1.224 vs 1.834 mm (razón 1,62) |
| Máximos de la media: estación vs satélite | días 108 y 308 vs 114 y 302 |

### Hallazgo 3 — el satélite sobreestima y el sesgo es estructural, no aleatorio

| Subzona | n | Sesgo | Razón sat/est | Altitud |
|---|---|---|---|---|
| 2611 (Valle, 975 m) | 198 | −0,207 | 1,18 | 975 m |
| 2610 (Valle, 952 m) | 178 | −0,328 | 1,41 | 952 m |
| 2613 (Risaralda) | 248 | −0,528 | 1,66 | 1.520 m |
| 2612 (Quindío) | 619 | −0,627 | 1,82 | 1.480 m |

- **Spearman sesgo vs altitud: −0,308** · **vs longitud: −0,363** → el satélite sobreestima
  más en la montaña y hacia el occidente. Es el comportamiento documentado de los productos
  satelitales de precipitación en terreno complejo: el algoritmo infrarrojo/microondas
  confunde nubosidad orográfica persistente con lluvia.
- **Spearman sesgo vs año: −0,457** → hay **deriva temporal**: el desacuerdo crece a lo largo
  del período. Esto es relevante para la Fase 4: un modelo entrenado en los años ochenta y
  evaluado en los dos mil parte con desventaja, y por eso la validación incluye partición
  temporal explícita.
- La diferencia es significativa en el **100 %** del año, con el máximo en junio (día 156),
  es decir en la **transición seca entre las dos temporadas**, cuando la lluvia real es baja
  y el error relativo del satélite es mayor.
- El satélite **atrasa la primera temporada** (día 114 vs 108) y **adelanta la segunda**
  (día 302 vs 308): comprime el ciclo bimodal en ~12 días.

---

## 5. Qué queda habilitado y qué queda bloqueado

**Habilitado** (producto de esta fase): `entrega/pares_estacion_pixel.csv`, 1.243 pares
estación-año con los 5 coeficientes de cada fuente, altitud, longitud, latitud y calidad del
emparejamiento. Es el insumo directo de la Fase 4 (modelo satélite → estación).

**Bloqueado, y no se disimula:**

1. Sin el catálogo real de coordenadas, la posición de cada estación tiene incertidumbre de
   decenas de kilómetros. Con 25 píxeles para 65 estaciones, el emparejamiento es de
   *vecindad*, no de *punto*. Variograma funcional y kriging sobre la red terrestre siguen
   fuera de alcance: un variograma con distancias que tienen ±24 km de error no es
   interpretable.
2. La altitud usada es la **mediana de la subzona** (dispersión intra-subzona de cientos de
   metros, p. ej. σ=491 m en 2612), así que la relación sesgo–altitud está atenuada: el
   −0,308 observado es una cota inferior del efecto real.
3. Las subzonas 5300, 5400, 5420 y 5421 (Patía y Pacífico sur) no tienen registros en el CNE,
   por lo que ni siquiera admiten coordenada aproximada.

**Lo que hay que pedir al docente o al grupo**: el archivo de metadatos de las 186 estaciones
(código, nombre, latitud, longitud, altitud). Con él, `fase3_comparacion.py` se re-ejecuta sin
cambios estructurales — solo se sustituye el bloque [1] por la lectura del catálogo y el
emparejamiento pasa a ser vecino más cercano real.
