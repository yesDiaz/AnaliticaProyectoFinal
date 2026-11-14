# Fase 4.1 y Fase 1 — Preprocesamiento y construcción de datos funcionales

Proyecto Final Analítica de Datos 2026-2S — precipitación en el Valle del Cauca
Fuentes: red de estaciones en tierra (precipitación diaria) e imágenes satelitales pentadales
(GeoTIFF, EPSG:4326, resolución 0,05°, malla 39×37, extensión lon −77,55…−75,70 y lat 3,05…5,00).

Scripts: `preproc_4_1.py` (depuración) y `fase1_datos_funcionales.py` (construcción funcional).
Semilla fija 42; rutas relativas; numpy 1.26.4, pandas 2.1.4, matplotlib 3.6.3.

---

## 0. Hallazgos sobre la disponibilidad de los datos

Antes de cualquier decisión metodológica hay tres restricciones que condicionan todo el proyecto:

1. **Las estaciones terminan en 2003.** El archivo unificado contiene 2.992.694 registros diarios
   de 344 estaciones entre 1930 y 2003. En el periodo del proyecto (1981-2013) hay 323 estaciones,
   pero **ningún registro posterior a 2003**.
2. **El satélite llega hasta abril de 2013.** Hay 2.328 rásteres: 72 péntadas/año para 1981-2012
   (32 años completos) y solo 24 péntadas de 2013 (hasta el 30 de abril).
3. **Periodo común efectivo estación↔satélite: 1981-2003** (23 años).

Consecuencia directa sobre la partición exigida (entrenamiento 1981-2005, validación 2006-2009,
prueba 2010-2013): con los datos entregados **los conjuntos de validación y prueba quedan vacíos
en la fuente de estaciones**. Las alternativas son (a) conseguir los registros 2004-2013 de las
estaciones, o (b) redefinir la partición dentro de 1981-2003 (por ejemplo 1981-1995 / 1996-1999 /
2000-2003), documentando el cambio. Esta decisión debe tomarse con el docente antes de la Fase 4.

4. **No hay coordenadas de las estaciones.** Los archivos entregados solo traen `STATION_ID`
   (código IDEAM), sin latitud, longitud ni altitud. Sin ellas no se puede hacer el emparejamiento
   estación↔píxel de la Fase 3 ni usar la altitud como covariable del codificador de la Fase 4.
   Se requiere el catálogo de estaciones del IDEAM (CNE).
5. Por el código IDEAM, 164 de las 186 estaciones retenidas pertenecen a la zona hidrográfica 26
   (río Cauca); 11 a la zona 53 y 11 a la 54. Estas 22 estaciones deben verificarse contra sus
   coordenadas: si caen fuera de la malla satelital, no podrán emparejarse con un píxel.

---

## 1. Fase 4.1 — Preprocesamiento y selección de datos

Las reglas se fijaron con criterios físicos y climatológicos, **no** con estadísticos del conjunto
de prueba, de modo que ninguna decisión usa información que no esté disponible en entrenamiento.

### 1.1 Reglas aplicadas

| Nº | Regla | Criterio numérico | Justificación |
|----|-------|-------------------|---------------|
| 1 | Fechas imposibles | `MONTH`/`DAY` que no forman fecha válida | error de digitación |
| 1b | Duplicados estación-fecha | se colapsan con la mediana | criterio robusto frente a un valor aberrante |
| 2 | Valores imposibles | `P < 0` o `P > 300 mm/día` → faltante | no existe lluvia negativa; los máximos diarios registrados en el valle geográfico del Cauca no alcanzan 250 mm, se deja margen |
| 3a | Total mensual en un día | un día concentra ≥ 90 % del total del mes, con ≥ 50 mm, y ≥ 25 días del mes en cero | la planilla mensual se transcribió en una sola casilla |
| 3b | Total mensual al cierre | el **último día del mes** concentra ≥ 50 % del total y ≥ 100 mm | patrón de planilla cerrada el último día |
| 4 | Ceros que son faltantes | racha de ≥ 45 días consecutivos con `P = 0` → faltante | ni la temporada seca (jun-ago) del régimen bimodal del Valle alcanza 45 días seguidos sin lluvia medible: la racha indica estación fuera de operación con el 0 usado como relleno |
| 5 | Cobertura mínima | se conserva el par estación-año con ≥ 330 días válidos (90,4 %) **y** hueco máximo ≤ 15 días | una curva anual no puede reconstruirse si falta un trimestre; el límite de hueco evita extrapolar dentro del dominio |
| 6 | Historia mínima | estaciones con ≥ 15 años válidos | el FPCA y la partición temporal necesitan varias curvas de la misma estación |
| 7 | Imputación | interpolación lineal **dentro del año**, solo en huecos ≤ 15 días, truncando en 0 | los pares con huecos mayores ya se eliminaron; no se usa información de otros años ni del conjunto de prueba |

En los pasos 2-4 el dato sospechoso se marca como faltante en lugar de corregirse: no hay forma de
recuperar el reparto real de un acumulado mensual, y una corrección inventada contaminaría la curva.

### 1.2 Tabla de trazabilidad

| Paso | Registros | Registros válidos | Estaciones | Pares estación-año |
|------|-----------|-------------------|------------|--------------------|
| 0. crudo 1981-2013 | 1.930.611 | 1.930.611 | 323 | 5.514 |
| 1. fechas válidas | 1.930.611 | 1.930.611 | 323 | 5.514 |
| 1b. sin duplicados | 1.930.611 | 1.930.611 | 323 | 5.514 |
| 2. valores imposibles | 1.930.611 | 1.930.561 | 323 | 5.514 |
| 3. totales mensuales en un día | 1.930.611 | 1.928.583 | 323 | 5.514 |
| 4. ceros falsos | 1.930.611 | 1.910.369 | 323 | 5.514 |
| 5. cobertura mínima | 1.632.635 | 1.632.582 | 305 | 4.481 |
| 6. estaciones con historia | 1.299.927 | 1.299.875 | 186 | 3.567 |
| 7. imputación de huecos cortos | 1.299.927 | 1.299.927 | 186 | 3.567 |

Detalle de lo detectado:

- **50** valores > 300 mm/día anulados. El mayor era de **859.375 mm**; se reportan en
  `tablas/4_1_valores_extremos.csv` los 219 casos por encima de 200 mm para revisión.
- **80** estación-mes con el total mensual concentrado en un día: 53 por la regla 3a y
  **27 por la regla 3b** (cierre de mes). Ejemplo: la estación 5421200201 registró 943 mm el
  31 de octubre de 1986 y 896 mm el 31 de mayo de 1988. Sin la regla 3b estos casos sobreviven
  a la depuración porque el resto del mes **no** está en ceros.
- **321 rachas** de ≥ 45 días en cero (18.214 registros) convertidas a faltante.
- De los 5.514 pares estación-año, **1.033 se descartan** por cobertura: 764 fallan los dos
  criterios, 252 solo por hueco > 15 días y 17 solo por días válidos < 330. Los pares rechazados
  tienen en promedio 268,9 días válidos y un hueco máximo de 83,8 días; los aceptados, 364,3 días
  y 0,6 días de hueco.
- Solo **52** valores necesitaron interpolación: el filtro de cobertura deja curvas prácticamente
  completas, de modo que la imputación no influye en los resultados.

**Conjunto final:** 186 estaciones, 3.567 curvas estación-año, 1981-2003, 1.299.927 registros
diarios sin faltantes, media de 5,318 mm/día y 59,0 % de días secos (`entrega/estaciones_depuradas.csv`).

---

## 2. Fase 1 — Construcción de los datos funcionales

Todo se programó paso a paso: **no se usó ninguna librería de análisis de datos funcionales**
(ni `fda`, ni `smooth.basis`, ni `skfda`), y tampoco `lm()` ni `lstsq`.

### 2.1 Dominio y transformación

- Dominio común **t ∈ [0, 365]**: el día del año se reescala con `(DOY − 0,5)/días_del_año × 365`,
  de modo que los años bisiestos y las péntadas satelitales viven en el mismo intervalo.
- Variable modelada: **√P**. La lluvia diaria es fuertemente asimétrica (59 % de ceros y colas
  largas); la raíz estabiliza la varianza y evita que unos pocos días extremos dominen el ajuste.
  Advertencia para la interpretación: la media en escala √ **no** se retransforma a mm aplicando el
  cuadrado (desigualdad de Jensen); si se reportan milímetros hay que corregir el sesgo.

### 2.2 Dos bases candidatas y por qué se comparan

La curva climatológica anual es **periódica**: el 31 de diciembre empalma con el 1 de enero. Por eso
se construyeron y compararon dos bases, ambas programadas a mano:

1. **B-splines cúbicos** (orden 4) por la recursión de **Cox-de Boor** (`base_bspline`): se parte de
   las indicadoras de cada intervalo de nodos y se aplica
   B<sub>i,d</sub>(t) = (t−τ<sub>i</sub>)/(τ<sub>i+d−1</sub>−τ<sub>i</sub>)·B<sub>i,d−1</sub>(t) +
   (τ<sub>i+d</sub>−t)/(τ<sub>i+d</sub>−τ<sub>i+1</sub>)·B<sub>i+1,d−1</sub>(t), con nodos interiores
   equiespaciados y multiplicidad 4 en los extremos.
2. **Fourier** (`base_fourier`): 1, sen(2πjt/365), cos(2πjt/365) para j = 1…(K−1)/2. Es **periódica
   por construcción**, de modo que impone el empalme de fin de año sin restricciones adicionales.

La elección se decidió con VCG, no por preferencia. El criterio es relevante porque la base B-spline
**no** impone periodicidad y, al no tener datos a la derecha del 31 de diciembre, el último tramo
polinómico se desvía: en el perfil climatológico medio la curva B-spline (K = 6) cae a 0,61 al cerrar
el año frente a 1,03 al abrirlo, un **salto de 0,41 √mm** que es puro artefacto de borde. La curva de
Fourier cierra el año con un salto de **0,0000** (ver `figs/fase1_bimodalidad.png`).

### 2.3 Coeficientes

Ecuaciones normales (Φ'Φ + λR)c = Φ'y resueltas con una **factorización de Cholesky escrita a
mano** (`cholesky_propia`) y sustitución hacia adelante y hacia atrás (`resolver_chol`). La inversa
de la matriz de Gram se obtiene resolviendo el mismo sistema contra la identidad, lo que permite
construir la matriz de suavizado y los valores de apalancamiento h<sub>ii</sub> sin recurrir a
rutinas de regresión.

### 2.4 Selección de base y de K (estaciones)

Promedio sobre una muestra aleatoria de 60 curvas estación-año (semilla 42), con los tres criterios
del curso (`figs/fase1_seleccion_K_estaciones.png`, `tablas/fase1_seleccion_K_estaciones.csv`):

| Base | K | ECM | VC | VCG |
|------|---|-----|----|-----|
| B-spline | 4 | 3,0597 | 3,1262 | 3,1280 |
| B-spline | 6 | 2,9425 | 3,0390 | 3,0418 |
| B-spline | 10 | 2,8911 | 3,0520 | 3,0564 |
| B-spline | 30 | 2,5761 | 3,0526 | 3,0586 |
| B-spline | 73 | 2,0970 | 3,3316 | 3,2781 |
| Fourier | 3 | 3,0853 | 3,1367 | 3,1367 |
| **Fourier** | **5** | **2,9573** | **3,0401** | **3,0401** |
| Fourier | 7 | 2,9260 | 3,0417 | 3,0417 |
| Fourier | 11 | 2,8683 | 3,0496 | 3,0495 |
| Fourier | 31 | 2,5748 | 3,0759 | 3,0755 |
| Fourier | 73 | 2,0866 | 3,2641 | 3,2620 |

El ECM decrece siempre al aumentar K —es el comportamiento esperado del error de ajuste— mientras
VC y VCG tienen un **mínimo interior**: se probaron K por debajo y hasta K = 73 (un coeficiente por
péntada) por arriba, así que el óptimo no está en el borde de la rejilla.

**Base elegida: Fourier con K = 5** (constante + 2 armónicos anuales), VCG = 3,0401 frente a 3,0418
del mejor B-spline (K = 6). Gana por tres razones acumuladas: (i) mejor VCG, (ii) empalme de fin de
año exacto, y (iii) interpretación física directa — los dos armónicos son precisamente los dos
pasos anuales de la Zona de Convergencia Intertropical sobre el valle, de modo que cada coeficiente
tiene significado climatológico en vez de ser un peso local sin lectura.

Entre K = 5 y K = 21 el VCG varía menos del 0,4 %: el resultado es estable y K = 5 se adopta por
parsimonia.

### 2.5 Selección de λ (rugosidad)

Con la penalización R = ∫ Φ''(t)Φ''(t)' dt (segunda derivada numérica de la base sobre una malla de
3.651 puntos e integración por trapecios):

| λ | ν = tr(S<sub>λ</sub>) | ECM | VC | VCG |
|---|----|-----|----|-----|
| 0 | 5,000 | 2,0664 | 2,1242 | 2,1242 |
| 10² | 5,000 | 2,0664 | 2,1242 | 2,1242 |
| 10⁴ | 4,971 | 2,0665 | 2,1239 | 2,1239 |
| **10⁵** | **4,736** | 2,0671 | 2,1218 | **2,1218** |
| 10⁶ | 3,670 | 2,0804 | 2,1229 | 2,1229 |
| 10⁷ | 2,198 | 2,1059 | 2,1315 | 2,1315 |
| 10¹² | 0,785 | 2,1508 | 2,1600 | 2,1600 |

El mínimo de VCG es **interior**: λ = 10⁵, con **ν = tr(S<sub>λ</sub>) = 4,736** grados de libertad
efectivos frente a los 5 nominales. La penalización apenas actúa, y es coherente: con solo dos
armónicos la curva ya es suave, así que no queda rugosidad que castigar. Esto es consecuencia
directa de haber elegido una base parsimoniosa.

### 2.6 Verificación numérica de la matriz de suavizado

Para el ajuste sin penalización, S = Φ(Φ'Φ)⁻¹Φ' debe ser una proyección ortogonal de rango K:

- `max |S·S − S| = 3,47 × 10⁻¹⁷` → **S es idempotente** (dentro del error de máquina).
- `tr(S) = 5,000000` con K = 5 → `|tr(S) − K| = 0,0` exactamente → **tr(S) = K**.
- `max |S − S'| = 5,20 × 10⁻¹⁸` → **S es simétrica**.
- En la fuente satelital: `max |S·S − S| = 6,94 × 10⁻¹⁷` y `tr(S) = 5,0000` con K = 5.

Con penalización S<sub>λ</sub> ya no es idempotente —como debe ser— y su traza se interpreta como
grados de libertad efectivos (ν = 4,736 para λ = 10⁵). Valores en `tablas/fase1_verificacion_S.csv`.

### 2.7 Covarianza residual y bandas punto a punto

σ̂² = SSE/(n − tr(S)) = **2,0951** (escala √mm). Para cada t, Var(ŷ(t)) = σ̂²·φ(t)'(Φ'Φ)⁻¹φ(t) y la
banda es ŷ(t) ± 1,96·EE(t), con EE medio de **0,1694** y ancho medio de banda **0,6641**
(`figs/fase1_banda_confianza.png`, `tablas/fase1_banda_confianza.csv`).

**Interpretación correcta:** se trata de bandas **punto a punto**. Para cada día del año por
separado cubren el valor esperado de la curva con 95 % de confianza; **no** es una banda simultánea
para toda la curva, de modo que la probabilidad de que la curva verdadera quede contenida en la
banda *en todos los días a la vez* es menor que 0,95. Comparar dos curvas día por día con estas
bandas infla el error tipo I.

### 2.8 Control climatológico: bimodalidad y periodicidad

Un suavizado agresivo podría aplanar los dos picos de lluvia del Valle del Cauca y volver inútil el
dato funcional. Sobre el perfil climatológico medio de las 186 estaciones
(`figs/fase1_bimodalidad.png`, `tablas/fase1_bimodalidad.csv`):

| Base | K | nº de máximos locales | días de los picos | salto 31-dic/1-ene | ECM |
|------|---|----------------------|-------------------|--------------------|-----|
| Fourier | 5 | 2 | 112 y 304 | **0,0000** | 0,0379 |
| Fourier | 7 | 2 | 112 y 303 | **0,0000** | 0,0370 |
| B-spline | 6 | 2 | 113 y 317 | 0,4136 | 0,0495 |

La base elegida conserva exactamente **dos máximos**, hacia el **22 de abril** y el **31 de
octubre**, con mínimos en julio-agosto (el más seco) y enero-febrero: es el régimen bimodal del
doble paso de la ZCIT, y es la estructura que las fases siguientes deben explotar. Los picos son
estables al cambiar K dentro de la base de Fourier; la base B-spline, además del salto de fin de
año, **desplaza el segundo pico unas dos semanas** (día 317 en vez de 304) y aplana el primero, lo
que la descalifica para estimar fechas de máximo.

### 2.9 Las dos fuentes en el mismo dominio

- Conversión péntada → día del año: la péntada p (1…72) corresponde al mes m = ⌊(p−1)/6⌋+1 y a la
  sub-péntada j = (p−1) mod 6 + 1, con días 1-5, 6-10, 11-15, 16-20, 21-25 y 26-fin de mes. Se usa
  el **día central** del intervalo, reescalado al dominio [0, 365].
- Los valores satelitales se dividen por el número de días de la péntada (la última tiene 3 a 6 días
  según el mes) para pasar de mm/péntada a **mm/día**, de modo que las dos fuentes comparten unidades
  antes de aplicar la raíz cuadrada.
- Selección **independiente** de base y K para el satélite (72 observaciones por año): el mínimo de
  VCG también cae en **Fourier con K = 5** (ECM 0,2381; VC 0,2748; VCG 0,2749), frente a 0,2804 del
  mejor B-spline (K = 8). Tabla completa en `tablas/fase1_seleccion_K_satelite.csv`.
- **Resultado relevante para la Fase 3:** las dos fuentes, elegidas por separado, convergen en la
  misma base y la misma dimensión. Los coeficientes de una curva de estación y los de una curva de
  píxel son por tanto **directamente comparables componente a componente**, sin reproyección: la
  comparación estación↔píxel puede hacerse en el espacio de coeficientes (5 números) en vez de
  evaluar y comparar curvas día a día.
- `figs/fase1_dos_fuentes.png` muestra una curva de estación y una curva satelital sobre el mismo
  dominio anual.

### 2.10 Salidas

| Archivo | Contenido |
|---------|-----------|
| `entrega/estaciones_depuradas.csv` | serie diaria depurada (186 estaciones, 1981-2003) |
| `entrega/coef_estaciones.csv` | **3.567** curvas estación-año × 5 coeficientes (Fourier, λ = 10⁵) |
| `entrega/coef_satelite.csv` | **46.176** curvas píxel-año (1.443 píxeles × 32 años) × 5 coeficientes, con lon/lat de cada píxel |

---

## 3. Limitaciones declaradas

1. La partición oficial no es aplicable a las estaciones: no hay datos después de 2003.
2. Sin coordenadas de estaciones no hay emparejamiento estación↔píxel ni altitud como covariable.
3. El 59 % de días secos hace que la curva de √P no sea un modelo del proceso de ocurrencia de
   lluvia: describe el nivel medio estacional. Si la Fase 4 necesita modelar ocurrencia e intensidad
   por separado, debe construirse una segunda familia de curvas (probabilidad de día lluvioso).
4. La segunda derivada de la base para R se calcula por diferencias finitas sobre una malla de
   3.651 puntos, no analíticamente; el error de cuadratura es despreciable frente a la escala de λ,
   pero conviene mencionarlo. (Para la base de Fourier la penalización tiene forma cerrada —los
   valores propios son (2πj/365)⁴— lo que permitiría verificar la cuadratura si se quisiera.)
5. K y λ se eligieron sobre una muestra de 60 curvas y se aplicaron a las 3.567. Una selección
   curva por curva daría un ajuste marginalmente mejor a costa de que las curvas dejen de vivir en
   la misma base, lo que impediría comparar coeficientes en las fases siguientes.
6. Con K = 5 la curva capta el ciclo estacional (nivel + dos armónicos), no los eventos
   individuales de lluvia. Es la representación adecuada para comparar años y localizaciones, pero
   no para estudiar extremos diarios.
