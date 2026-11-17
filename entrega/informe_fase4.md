# Fase 4 — Red neuronal funcional, ablación y evaluación

Proyecto Final de Analítica de Datos · UAO 2026-2S
Script: `fase4_modelos.py` · Log: `entrega/log_fase4.txt`
Tablas: `tablas/fase4_metricas.csv`, `fase4_resumen.csv`, `fase4_wilcoxon.csv`, `fase4_enso.csv`
Figuras: `figs/fase4_curvas.png`, `fase4_metricas.png`, `fase4_dispersion.png`
Predicciones: `entrega/pred_reservadas.csv` (89.790 filas, `station_id, year, day, value_mm`)

---

## 1. Problema y arquitectura (4.2)

**Tarea.** Predecir la curva anual de precipitación de una estación terrestre a partir de la
curva del producto satelital del píxel emparejado en la Fase 3, más tres covariables de
contexto (altitud de la subzona, longitud y latitud del píxel). Es decir: *corregir el
satélite usando la red terrestre como verdad*.

```
curva satelital  --[codificador L2]-->  z = W^½ c_sat (5)  ┐
altitud, lon, lat ------------------------------------------┤--> cabeza --> 5 scores
                                                            ┘      (lineal | MLP | KAN)
                                                                      |
                                                 decodificador FPCA (solo train)
                                                                      |
                                                            coeficientes --> curva
```

- **Codificador funcional**: `z = W^½ c_sat` con `W = ∫Φ(t)Φ(t)ᵗdt` por cuadratura
  trapezoidal (traza 1.092,00; `max|W − Wᵗ| = 5,7e-14`). Se usa el producto interno de L²,
  no el euclídeo de los coeficientes: así la distancia en el espacio latente es la distancia
  real entre curvas.
- **Decodificador FPCA** estimado **solo con el conjunto de entrenamiento** (sin fuga):
  varianza explicada PC1 42,7 %, PC2 30,3 %, PC3 10,6 %, PC4 9,9 %, PC5 6,5 %. Para el
  umbral del 99 % se retienen **M = 5** componentes; ortonormalidad en L²
  `max|BᵗWB − I| = 1,9e-15`.
- **Pérdida**: norma L² funcional `(ĉ − c)ᵗ W (ĉ − c)/T` + `λ·rugosidad`, con
  `R = ∫Φ''(t)Φ''(t)ᵗdt` analítica (traza 5,43e-4) sobre las funciones peso `w_k(t)`;
  `λ = 1e-6`.
- **Escala y no negatividad**: se modela en `√P`; la reconstrucción en mm es
  `max(curva,0)²`. Días recortados: **0,00 %** (lineal), **0,01 %** (MLP), **0,15 %** (KAN);
  `pred_reservadas.csv` no contiene ningún valor negativo.

## 2. Particiones, sin fuga de información

- **Estaciones reservadas por sorteo: 13 de 65** (20 %, semilla 2026). Nunca entran al
  entrenamiento ni a la validación: miden generalización **espacial** (estación no vista).
- **Partición temporal** sobre las 52 estaciones restantes: entrenamiento 1981–1996 (724
  pares), validación 1997–1999 (111), prueba 2000–2003 (162).
- La guía original planteaba 1981–2005 / 2006–2009 / 2010–2013, pero la **fe de erratas del
  docente** (noviembre de 2026) confirmó que la base no se pudo recuperar hasta 2013 y que el
  proyecto usa la serie verificada **hasta 2003**. La partición aplica la misma lógica
  —extrapolar hacia adelante en el tiempo— a la ventana vigente.
- Tres semillas (20264, 20265, 20266); se reporta media ± desviación.

## 3. Verificaciones exigidas

| Verificación | Resultado |
|---|---|
| Cabeza lineal (gradiente, LBFGS float64) ≡ regresión funcional en forma cerrada por **Cholesky propia** | `max|dif| = 6,7e-09` ✔ (tolerancia 1e-4) |
| KAN sobre función sintética `f(x₁,x₂)=sin(πx₁)+x₂²` | RMSE 0,0026 frente a sd 0,764 ✔ |
| Parámetros igualados KAN vs MLP (`sum(p.numel())`) | KAN (pykan) **756**, MLP (h=54) **761** → **0,7 %** de diferencia ✔ (≤10 %) |
| Cabeza lineal (referencia) | 45 parámetros |
| Gram y FPCA | simetría 5,7e-14; ortonormalidad 1,9e-15 ✔ |

La equivalencia lineal ≡ regresión funcional no es casual: como `B` es W-ortonormal y M=K,
la pérdida `(c−ĉ)ᵗW(c−ĉ)` es exactamente `‖s−ŝ‖²` en scores, así que el mínimo de la cabeza
lineal **es** la solución de mínimos cuadrados funcional. En `float32` el residuo se queda en
2e-4 (ruido de precisión); en `float64` baja a 7e-9.

## 4. Resultados (4.4)

### Bloque de prueba temporal (2000–2003, estaciones vistas)

| Modelo | L² relativo | KGE | Sesgo acum. (%) | Error picos (días) |
|---|---|---|---|---|
| Satelital sin corregir | 0,5045 | −0,4192 | **+72,3** | 23,9 |
| Cabeza lineal | 0,2754 ± 0,0001 | 0,1521 | +2,8 | 22,3 |
| MLP | 0,2830 ± 0,0099 | 0,1224 | −0,1 | 26,4 |
| **KAN** | **0,2746 ± 0,0009** | **0,3383** | +2,3 | 23,0 |

### Bloque de estaciones reservadas (13 estaciones no vistas, 1981–2003)

| Modelo | L² relativo | KGE | Sesgo acum. (%) | Error picos (días) |
|---|---|---|---|---|
| Satelital sin corregir | 0,4028 | −0,0631 | **+50,0** | 18,4 |
| Cabeza lineal | 0,2161 | 0,2149 | −7,6 | 16,8 |
| **MLP** | **0,2095 ± 0,0016** | **0,3277** | −8,8 | 16,8 |
| KAN | 0,2318 ± 0,0008 | 0,1884 | −11,6 | 17,2 |

### Inferencia (bootstrap 1000 + Wilcoxon pareada por estación)

| Comparación | Bloque | Δ L² | IC 95 % | Veredicto |
|---|---|---|---|---|
| lineal vs satelital | temporal | −0,2290 | [−0,2559, −0,2000] | **significativo** (p=7,1e-11) |
| MLP vs satelital | temporal | −0,2285 | [−0,2570, −0,2028] | **significativo** (p=3,4e-11) |
| KAN vs satelital | temporal | −0,2289 | [−0,2585, −0,2014] | **significativo** (p=4,4e-12) |
| MLP vs lineal | temporal | +0,0006 | [−0,0047, +0,0060] | no concluyente |
| KAN vs lineal | temporal | +0,0002 | [−0,0121, +0,0111] | no concluyente |
| KAN vs MLP | temporal | −0,0004 | [−0,0125, +0,0114] | no concluyente |
| MLP vs lineal | reservadas | −0,0084 | [−0,0119, −0,0051] | **significativo** (p=2,4e-04) |
| KAN vs lineal | reservadas | +0,0164 | [+0,0074, +0,0268] | **significativo a favor del lineal** |
| KAN vs MLP | reservadas | +0,0248 | [+0,0151, +0,0343] | **significativo a favor del MLP** |

Aplicando la regla de la rúbrica (una cabeza supera a otra **solo si** el IC 95 % de la
diferencia excluye el cero):

1. **Las tres cabezas baten al satélite sin corregir de forma contundente**: el error L²
   relativo cae de 0,50 a ~0,27 (**−45 %**) y el sesgo del acumulado anual de **+72 % a
   ±3 %**. Esto es el resultado principal de la fase.
2. **Entre cabezas, en el bloque temporal no hay ganador**: los tres IC incluyen el cero.
   Con 8 entradas y 5 salidas, la relación satélite→estación es esencialmente **lineal**, y
   756 parámetros no compran nada.
3. **En estaciones no vistas sí hay orden**: MLP < lineal < KAN (MLP mejor). La KAN es la
   que peor extrapola a estaciones nuevas, consistente con que sus splines se ajustan al
   rango de entradas visto en entrenamiento.

### Desglose por fase ENSO (ONI medio anual, NOAA/PSL Niño 3.4)

L² relativo, bloque temporal:

| Modelo | El Niño | La Niña | Neutro |
|---|---|---|---|
| Satelital sin corregir | 0,5792 | 0,3441 | 0,5599 |
| Lineal | 0,3089 | 0,2466 | 0,2758 |
| MLP | 0,3321 | 0,2489 | 0,2645 |
| KAN | 0,2996 | 0,2481 | 0,2798 |

## 5. Interpretación climatológica (4.5)

- **El satélite necesita corrección estructural, no cosmética.** Sobreestima el acumulado un
  +72 % en el bloque de prueba y su KGE es **negativo** (−0,42): usado crudo es peor que
  predecir la media. Tras la corrección el KGE pasa a positivo y el sesgo a ±3 %.
- **La corrección que aprenden los modelos es la que la Fase 3 había diagnosticado**: sesgo
  creciente con la altitud (Spearman −0,31) y hacia el occidente (−0,36). Las covariables
  altitud/longitud/latitud son las que permiten corregir sin ver la estación.
- **El error es mayor en El Niño** (0,30–0,33) que en La Niña (0,25) para todos los modelos,
  y el satélite crudo se degrada aún más (0,58 vs 0,34). En años secos de El Niño la lluvia
  es intermitente y débil; el satélite interpreta nubosidad orográfica sin precipitación como
  lluvia, y el error relativo crece porque el denominador (lluvia real) es pequeño. En La Niña,
  con lluvia abundante y persistente, las dos fuentes concuerdan mejor.
- **La fecha de los picos apenas mejora** (23,9 → 22,3 días de error en el bloque temporal).
  La corrección es de **magnitud**, no de **fase**: el desfase de ~12 días del ciclo bimodal
  que la Fase 3 detectó es un sesgo del producto satelital que una transformación coeficiente
  a coeficiente, con sólo dos armónicos, no puede reparar. Para corregir fase haría falta
  permitir desplazamientos temporales explícitos (warping) o una base con más armónicos.
- **Ganancia práctica**: con 13 estaciones nunca vistas, el modelo reduce el error un 48 % y
  deja el sesgo en −9 %. Es decir, el esquema sirve para **generar series de precipitación
  en sitios sin estación** dentro de la misma región, que es exactamente para lo que se usan
  los productos satelitales corregidos en hidrología operativa.

## 6. Limitaciones honestas

1. El emparejamiento estación–píxel de la Fase 3 es *informativo, no exacto* (65 estaciones
   sobre 25 píxeles, correlación de anomalías 0,53 contra 0,27 del nulo). Parte del error
   residual (L² ≈ 0,27) es error de emparejamiento, no de modelo.
2. La altitud es la mediana de la subzona, con dispersión de cientos de metros: la covariable
   más informativa del problema entra degradada.
3. `R²` funcional y KGE están calculados sobre curvas suavizadas con K=5; la variabilidad
   diaria real no está en el objetivo y por tanto no se evalúa.
4. Sólo 162 pares en el bloque temporal de prueba y 13 estaciones reservadas: los IC son
   anchos y las diferencias entre cabezas, pequeñas por construcción.
5. pykan se usó tal cual (756 parámetros, grid 5, k 3); no se exploró su regularización
   específica ni la poda simbólica, que es donde una KAN suele aportar interpretabilidad.
