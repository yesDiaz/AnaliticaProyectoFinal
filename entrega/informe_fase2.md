# Fase 2 — Análisis exploratorio funcional

Proyecto Final de Analítica de Datos · Precipitación en el suroccidente colombiano
(estaciones terrestres 1981–2003 y producto satelital 1981–2012)

Script: `fase2_eda_funcional.py` · log de ejecución: `log_fase2.txt`
Figuras: `figs/fase2_*.png` · Tablas: `tablas/fase2_*.csv`
Entorno: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, matplotlib 3.11.2, scikit-fda 0.10.1 (semilla 20262)

---

## 0. Insumos y reglas de cálculo

| Fuente | Curvas | Unidad muestral | Base |
|---|---|---|---|
| Estaciones (`coef_estaciones.csv`) | 3.567 | estación-año (186 estaciones, 1981–2003) | Fourier K=5, T=365 |
| Satélite (`coef_satelite.csv`) | 46.176 | píxel-año (1.443 píxeles, 1981–2012) | Fourier K=5, T=365 |
| ONI Niño 3.4 (`aux/oni.txt`, NOAA/PSL) | 1948–2026 | mes | — |

Todo estadístico funcional se calculó **a mano** y se verificó contra una
referencia independiente, como exige la rúbrica:

- Media funcional: acumulador explícito curva por curva (sin `np.mean`), verificada
  contra la identidad analítica `Φ(t)·c̄`, contra `numpy.mean` y contra
  `skfda FDataGrid.mean()` → error máximo **6.2e-15** (estaciones) y **4.6e-14** (satélite).
- Varianza, desviación, covarianza y correlación funcional: dobles bucles sobre la malla,
  sin funciones de alto nivel.
- Derivadas: **analíticas** sobre la base de Fourier (no diferencias finitas).
- Profundidades Fraiman–Muniz y Modified Band Depth: implementadas desde la definición,
  con pesos de integración trapezoidal; verificadas contra `skfda` en una submuestra de 400
  curvas → `max|MBD_manual − MBD_skfda| = 8.1e-05`, `max|FM_manual − FM_skfda| = 1.1e-04`
  (diferencia residual atribuible al tratamiento de empates en la función de distribución
  empírica, no a la fórmula).
- Mínimos cuadrados de la Fase 1: Cholesky propia, sin `lm()` ni `lstsq`.

- FPCA: problema de autovalores generalizado resuelto a mano con la matriz de Gram de la base
  (`W = ∫Φ(t)Φ(t)ᵗdt`) y simetrización por `W^{1/2}`; verificado contra
  `skfda.preprocessing.dim_reduction.FPCA` → misma varianza explicada y
  `|corr(scores)| = 1.000000` en las tres primeras componentes.

---

## 1. Validación de la representación funcional (bloque [9], `figs/fase2_validacion.png`)

Antes de interpretar, se contrastó la curva ajustada contra el dato observado depurado
(`estaciones_depuradas.csv`, 37 MB, acumulado por estación-año):

| Métrica | Valor |
|---|---|
| n | 3.567 curvas |
| Acumulado observado medio | 1.938 mm/año |
| Integral de la curva ajustada al cuadrado | 810 mm/año |
| Sesgo | −1.128 mm (−58,2 %) |
| RMSE | 1.234 mm |
| r de Pearson | 0,9730 |
| ρ de Spearman | 0,9271 |

**Lectura.** La curva está en escala √P (transformación de varianza-estabilización de la
Fase 1) y con K=5 retiene solo los dos primeros armónicos: por construcción elimina la
varianza diaria de alta frecuencia, de modo que `∫(curva)²` **no** reproduce el total anual
—lo subestima en un factor ≈2,39— pero preserva casi íntegro el **ordenamiento** entre
estaciones-años (ρ=0,93) y la **forma estacional**, que es lo que explotan las fases 2 a 4.
Por eso, en todo análisis de magnitud (acumulados, ENSO, gradiente) se usa el acumulado
**observado**, y la representación funcional se reserva para forma, derivadas y profundidad.
Esto se documenta como limitación explícita, no se corrige subiendo K: aumentar K reintroduce
ruido diario y desvirtúa el objeto funcional «ciclo anual».

---

## 2. Media y dispersión funcional (`figs/fase2_media_sd.png`)

- Ciclo **bimodal** nítido en ambas fuentes, firma del doble paso de la ZCIT sobre el
  suroccidente colombiano.
- Desviación estándar funcional de las estaciones: mínimo **0,606** en el día 77 (mediados de
  marzo, subida de la primera temporada) y máximo **0,785** en el día 316 (mediados de
  noviembre). La variabilidad entre estaciones-años es mayor en la **segunda** temporada que
  en la primera: la segunda temporada es la que responde al ENSO.
- Varianza total integrada (traza de la covarianza): estaciones **184,55**, satélite **273,68**
  (escala (√mm)²). El producto satelital es más variable porque su malla cubre la vertiente
  pacífica hiperhúmeda, ausente de la red de estaciones.

## 3. Covarianza y correlación funcional (`figs/fase2_covarianza_correlacion.png`)

| Par de días | Estaciones | Satélite |
|---|---|---|
| 15 ene vs 15 abr (día 15 vs 105) | 0,736 | 0,889 |
| 15 abr vs 27 oct (día 105 vs 300) | 0,775 | 0,923 |

La superficie de correlación es alta y de bandas anchas: dentro de un mismo año las dos
temporadas de lluvia se mueven juntas (forzamiento común de gran escala). La correlación es
sistemáticamente **mayor en el satélite**, coherente con que un píxel promedia 0,05° × 0,05° y
amortigua el ruido pluviométrico puntual.

## 4. Derivadas e inicio/fin de temporadas (`figs/fase2_derivadas.png`)

Calculadas por derivación analítica de la base de Fourier y lectura de ceros de μ′(t):

| | Máximos | Mínimos | Inicios | Finales |
|---|---|---|---|---|
| Estaciones | 22-abr, 01-nov | 28-ene, 26-jul | 06-mar, 15-sep | 03-jun, 22-dic |
| Satélite | 07-may, 28-oct | 03-feb, 30-jul | 26-mar, 04-sep | 21-jun, 17-dic |

- Las dos fuentes coinciden en el régimen bimodal y en el orden de los eventos, con un
  **desfase de ~2 semanas** en la primera temporada (satélite más tardío) y de ~4 días en la
  segunda.
- Nivel medio anual: estaciones **1,273** vs satélite **2,867** (√mm/día) — otra consecuencia
  de la cobertura espacial distinta, no de un error de calibración.

## 5. Profundidad funcional y boxplot funcional (`figs/fase2_boxplot_funcional.png`)

Boxplot funcional de Sun & Genton (2011) construido sobre MBD, con factor 1,5 del
rango central del 50 %.

- FM: 0,5003–0,9646 · MBD: 0,0012–0,4965 · ρ(FM, MBD) = **0,9893** (las dos profundidades
  ordenan las curvas casi igual; MBD discrimina mejor en la cola).
- Curva mediana (MBD máxima): estación **2613600103**, año **2003**.
- Atípicas: **54 de 3.567** curvas (**1,51 %**); en el satélite **0 de 46.176** (0,00 %).
- Las atípicas no se reparten al azar: se concentran en pocas estaciones —
  `5300000203` (15 años), `5300900102` (12), `5301100101` (11), `5400000201` (6) — todas de
  la zona hidrográfica del Pacífico/Patía, con acumulados de 15.000 a 23.800 mm/año.
  **No son errores de dato: son el régimen hiperhúmedo del Chocó biogeográfico**, que la
  profundidad marca como extremo respecto a la masa de estaciones andinas.
- Por año, los picos de atipicidad son 1993 y 1996 (5 curvas), 1988, 1994, 1998 y 2002 (4).

**Consecuencia metodológica:** la población de estaciones no es homogénea; cualquier modelo
de la Fase 4 debe o estratificar por régimen, o usar estimadores robustos basados en
profundidad, o incluir una covariable de régimen. Descartar esas curvas sería descartar la
señal climática más fuerte del área.

## 6. Variabilidad interanual y ENSO (`figs/fase2_interanual_enso.png`)

- Acumulado anual medio **1.939 mm**, mínimo **1.556 mm (1992)**, máximo **2.398 mm (1984)**.
- Spearman acumulado vs ONI DJF: **−0,270**; vs **ONI medio del año civil: −0,403**
  (n=23 años). Se reportan ambas: la clasificación DJF hereda el evento del invierno previo y
  desalinea años como 1988 (Niño en DJF, Niña fuerte desde mayo), por lo que para un acumulado
  de año civil el ONI medio anual es el criterio pertinente.
- Medias por fase (ONI medio anual):

| Fase | n | Acumulado medio | Años |
|---|---|---|---|
| El Niño | 6 | **1.812 mm** | 1982, 1987, 1991, 1992, 1997, 2002 |
| Neutro | 11 | 1.914 mm | 1981, 1983, 1986, 1990, 1993, 1994, 1995, 1996, 1998, 2001, 2003 |
| La Niña | 6 | **2.113 mm** | 1984, 1985, 1988, 1989, 1999, 2000 |

La Niña llueve **+17 %** frente a El Niño (301 mm/año de diferencia), con el signo negativo
esperado para los Andes colombianos. Con n=23 años la correlación es moderada y no significativa
al 5 % de forma concluyente; se presenta como señal consistente, no como prueba.

## 7. Gradiente orográfico (`figs/fase2_orografia.png`)

Explotando los 1.443 píxeles del producto satelital (los únicos con coordenadas disponibles,
ver § 8):

- Acumulado anual por píxel: mínimo **1.013 mm**, mediana **3.074 mm**, máximo **6.170 mm**.
- Vertiente pacífica (lon ≤ −76,8°): **5.080 mm/año** · valle interandino (−76,5 a −75,9°):
  **1.855 mm/año** → razón **2,74**.
- Spearman acumulado vs longitud: **−0,813** (más lluvia hacia el occidente).

El gradiente zonal es el factor dominante de la variabilidad espacial, por encima del año:
en 1.800 km² el acumulado cambia por un factor de 6. Esto justifica que la Fase 4 modele la
posición (o un proxy orográfico) antes que el tiempo.

## 7bis. FPCA — componentes principales funcionales (`figs/fase2_fpca.png`)

Resuelto a mano (ver § 0) y verificado contra `skfda` con coincidencia exacta
(`|corr(scores)| = 1,000000`; ortonormalidad `8.9e-16`; varianza de los scores igual a los
autovalores con error `2.8e-14`).

| Componente | Varianza explicada | Acumulada |
|---|---|---|
| PC1 | **79,6 %** | 79,6 % |
| PC2 | 10,0 % | **89,6 %** |
| PC3 | 5,0 % | 94,6 % |
| PC4 | 3,4 % | 98,0 % |
| PC5 | 2,0 % | 100 % |

- **PC1 (79,6 %) es un modo de nivel**: eigenfunción del mismo signo en todo el año, de modo que
  un score alto desplaza la curva completa hacia arriba. Lo confirma su relación con el total
  observado: **Spearman(PC1, acumulado observado) = 0,921**. Es decir, cuatro quintos de la
  variabilidad funcional son «cuánto llueve», no «cuándo».
- **PC2 (10,0 %) es un modo de contraste entre temporadas**: cambia de signo entre la primera y
  la segunda temporada, y describe años en los que una temporada compensa a la otra.
- Con **dos componentes se capta el 89,6 %** de la variabilidad: la Fase 4 puede modelar las
  3.567 curvas con dos escalares por curva en lugar de cinco coeficientes, sin pérdida
  apreciable. En el plano de scores, el gradiente de color del acumulado es monótono a lo largo
  de PC1, y las curvas atípicas por profundidad (§ 5) ocupan el extremo derecho: las dos
  técnicas —profundidad y FPCA— identifican el mismo bloque hiperhúmedo.

Tablas: `tablas/fase2_fpca_varianza.csv`, `tablas/fase2_fpca_scores.csv`
(scores PC1–PC3 por estación-año, con acumulado observado y MBD).

---

## 8. Limitación bloqueante: coordenadas de las estaciones

Los 186 `STATION_ID` de 10 dígitos **no corresponden** al campo `Codigo` del Catálogo Nacional
de Estaciones del IDEAM (`datos.gov.co`, dataset `hp9r-jxuu`, 29.320 registros descargados y
verificados): **0 de 186** coinciden de forma literal, con o sin ceros a la izquierda. La mejor
regla de reescritura probada (`id[:4] + id[6:9] + "0"`) empareja 134/186, pero es **espuria**:
sitúa las estaciones en Antioquia, Quindío y Risaralda y solo 21 de las 134 caen dentro de la
malla satelital del proyecto (lon −77,525 a −75,725; lat 3,075 a 4,975). Evidencia completa en
`aux/NOTA_CODIGOS_ESTACIONES.md`, `aux/coords_estaciones.py` y `aux/estaciones_cne_match.csv`.

**Efecto:** sin latitud/longitud/altitud de las estaciones no es posible la covarianza espacial,
el variograma funcional ni el kriging sobre la red terrestre, ni usar la altitud como covariable.
El análisis espacial de este informe (§ 7) se sostiene por completo sobre la malla satelital,
que sí tiene geo-referencia. Se optó por documentar el fallo de emparejamiento como hallazgo
metodológico en vez de usar coordenadas falsas.

---

## 9. Síntesis

1. El ciclo anual de precipitación es bimodal y las dos fuentes lo reproducen con un desfase
   máximo de dos semanas; el objeto funcional es el adecuado para esta estructura.
2. La variabilidad entre años se concentra en la segunda temporada (σ máxima en noviembre),
   que es también la sensible a ENSO.
3. La heterogeneidad espacial domina: el gradiente Pacífico→valle (×2,74) supera cualquier
   efecto interanual (×1,17 entre fases ENSO).
4. Las curvas atípicas por profundidad son régimen, no error: identifican el bloque de
   estaciones hiperhúmedas del Pacífico.
5. La representación con Fourier K=5 es fiel en forma y orden (ρ=0,93) pero sesgada en
   magnitud (−58 %): los análisis de total usan el dato observado.
6. El FPCA reduce el problema a dos dimensiones (89,6 % de la varianza): PC1 = nivel de lluvia
   (ρ=0,92 con el acumulado), PC2 = contraste entre las dos temporadas.
