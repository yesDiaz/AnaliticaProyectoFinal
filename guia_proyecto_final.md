# Proyecto Final — Guía completa paso a paso

> Guía de estudio hecha para ti, César. Resume el enunciado oficial y te da un plan de ataque. Todo lo técnico sale del enunciado; donde el enunciado no define algo, lo marco con ⚠️ **por confirmar con el profesor**. No inventes requisitos: cuando dudes, relee el enunciado antes que cualquier otra cosa.

---

## 1. Resumen del proyecto (lee esto primero)

Datos de **lluvia en el Valle del Cauca, 1981–2013**, de dos fuentes: **estaciones en tierra** (resolución diaria, puntual) y **producto satelital** (resolución péntada, píxel de malla regular). Cada año de lluvia en una localización se trata como un **dato funcional** (una curva), no como escalares sueltos.

El trabajo va en **fases secuenciales**: (1) construir las curvas funcionales a mano, (2) análisis exploratorio funcional, (3) comparar las dos fuentes, y (4) entrenar una **red KAN** que corrija el sesgo del satélite reconstruyendo la curva de estación a partir de la satelital, comparándola contra un MLP, una cabeza lineal (= regresión funcional) y la curva satelital sin corregir.

**Se entrega:** un informe **PDF** (≤20 páginas, una columna) + **un archivo R y un archivo Python** reproducibles (comprimidos aparte) + predicciones CSV de las estaciones reservadas. Luego hay **sustentación de 40 min** donde el profesor pregunta a cualquier integrante sobre cualquier fase.

---

## 2. Reglas críticas de entrega (EXACTAS del enunciado — no las adivines)

Estas son condiciones duras. Romper una te puede costar la nota entera. Repásalas con el grupo antes de entregar.

- **Formato PDF.** El nombre del archivo / asunto debe ser exactamente: **`Proyecto final AnalíticaDeDatos`** (regla 2).
- **Todos los integrantes dentro del documento.** No se aceptan informes compartidos por Google Drive, OneDrive, etc. (regla 3).
- **Un único archivo de R y un único archivo de Python**, separados del documento, con **todo** el código, **completamente reproducible** (regla 4). Se entregan comprimidos en **zip o rar** cargados en la plataforma (entregable "Script", 5 %).
- **R y Python obligatorios.** Solo se pueden usar scripts, procedimientos y metodologías **vistos en clase**. Las librerías adicionales se permiten **solo en los apartados mencionados** — en el ítem 4 (red funcional/KAN) se permiten **PyTorch y pykan**; para construir las curvas (ítem 1) usar librerías baja la nota a **máximo 2**; en el EDA (ítem 2) permiten librerías para los descriptivos pero la media funcional va verificada a mano (reglas 5, 18 y rúbrica).
- **Máximo 20 páginas**, incluidas las referencias, **a una sola columna**. Dos columnas = no se acepta. Solo se califican las primeras 20 páginas (regla 8 y rúbrica ítem 5).
- **No se acepta:** documento a mano, trabajo hecho **solo** en Google Colab, ni documento en **R Markdown** (regla 9).
- **Política de IA generativa:** se espera desarrollo autónomo. **Si usas IA, debes entregar un documento de Word aparte** con los **prompts usados** y el **nombre de la herramienta** (ChatGPT, Claude, Gemini, DeepSeek…). IA no declarada → el ítem del informe baja a **máximo 2** (regla 10 y rúbrica ítem 5). ⚠️ Esta guía cuenta como apoyo de IA sobre el *enunciado*, no sobre tu código ni tu texto; aun así, si llegas a usar IA para cualquier parte del trabajo, decláralo. Por confirmar con el profesor si una guía de estudio como esta debe declararse.
- **Plazo rígido:** no se aceptan trabajos después de la fecha/hora ni por otros medios distintos de la plataforma institucional. Informe PDF + scripts se entregan **antes de la sustentación** y **no se reciben versiones posteriores** (reglas 11 y 18). **Entrega: domingo 22-nov, 19:00.**
- **Plagio = 0,0.** Copia total o parcial de otro grupo, o código/texto ajeno sin citar, anula la nota (reglas 12 y 19).
- **Sustentación 40 min** de presentación + preguntas del profesor (regla 14). **Todos deben asistir**; quien falte sin justificación aceptada saca **0,0 en los ítems de sustentación** (regla 13). Única excepción: incapacidad médica o calamidad doméstica con soporte enviado **antes de la hora de inicio**.
- **El profesor pregunta a cualquiera sobre cualquier fase** (regla 15). Si respondes mal o no respondes, tu nota individual baja **−0,2** (regla 16); si otro responde por ti, cuenta como no respondida. La nota de sustentación **puede ser individual** si los desempeños son claramente distintos (regla 17).

> **Traducción práctica:** todos tienen que entender TODO el proyecto, no solo su parte. Eso define cómo repartimos el trabajo (ver diagrama 4) y cómo ensayamos la sustentación.

---

## 3. Cronograma sugerido (viernes 7-nov → domingo 22-nov 19:00)

Tienes 15 días, pero compártelos con Hacking Ético (Lab 07 + Lab 08 vencen **martes 10-nov 15:30**) y con la Práctica Extra OpenVAS (**22-nov 15:30**). Por eso el fin de semana 1 va a medias y el grueso de Analítica arranca el **miércoles 12**.

```
NOV  V7  S8  D9  L10 M11 M12 J13 V14 S15 D16 L17 M18 M19 J20 V21 S22
     │   │   │   │HE │   │═══ANALÍTICA, grueso═════════════════│TOPE│
     └HE labs (martes 10, 15:30)         ↑ margen de 24h antes del tope
```

- **V7–L10 (fin de semana 1 + lunes):** prioridad **Hacking Ético** (vence martes). En Analítica solo: conseguir/entender los datos, repartir roles, leer el enunciado **completo en grupo** (medio día máx). No empieces a codificar nada serio todavía.
- **M12–J13 (Fase 1 — construcción de datos funcionales, 10 %):** base B-spline a mano, mínimos cuadrados, selección de K/λ, matriz de suavizado y su verificación, bandas punto a punto. Curvas de **ambas** fuentes sobre el mismo dominio anual. **+ Preprocesamiento 4.1** (depuración, cobertura, tabla de trazabilidad) — hazlo aquí porque todo lo demás depende de qué datos entran.
- **V14–S15 (Fase 2 — EDA funcional, 15 %):** media/varianza/covarianza/correlación funcional, derivadas, profundidad (Fraiman–Muniz, MBD), boxplot funcional, atípicas. Interpretación climatológica.
- **D16–L17 (Fase 3 — comparación de fuentes, 15 %):** emparejamiento estación–píxel, curva de diferencia con banda, correlación cruzada funcional, comparación por altitud/subregión y por fase ENSO.
- **M18–J20 (Fase 4 — red funcional + ablación, 45 %):** codificador funcional, decodificador FPCA, las tres cabezas (lineal, KAN, MLP), 3 semillas, evaluación con las 4 métricas + bootstrap, interpretación. Es la fase más larga y la que más pesa — por eso tiene 3 días.
- **V21 (integración + informe):** redactar el PDF, numerar tablas/figuras, armar los dos scripts limpios y probar que corren **de principio a fin sin tocar nada** (rutas relativas, semillas fijas). Generar el CSV de estaciones reservadas.
- **S22 mañana–tarde (margen de seguridad):** revisión final, checklist, y **subir a plataforma antes de las 19:00**. ⚠️ No dejes la subida para las 18:45. Ten en cuenta que OpenVAS de HE vence el **mismo día 15:30** — súbelo antes para no chocar.

> Margen real: ~24 h entre terminar (V21 noche) y el tope (D22 19:00). Úsalo para imprevistos, no lo gastes antes.

---

## 4. Paso a paso técnico por fase

Cada fase lista **qué hacer**, **con qué** (R/Python) y **las trampas** (los "Tope" de la rúbrica son literalmente dónde pierdes puntos).

### Fase 1 — Construcción de datos funcionales (10 %)

**Qué hacer**
1. Montar la **matriz de la base Φ** (B-splines) y estimar los coeficientes por **mínimos cuadrados programados a mano, sin librerías** (ese es el punto del ítem 1).
2. Elegir el número de bases **K** (o el λ de penalización) con **ECM, validación cruzada (VC) y validación cruzada generalizada (VCG)**: muestra la **gráfica** y el **valor elegido para cada fuente** (diaria y péntada por separado).
3. **Verificar numéricamente la matriz de suavizado S**: que sea idempotente y que **tr(S) = K**; o, con penalización, los grados de libertad efectivos **ν = tr(S_λ)**.
4. Calcular **covarianza residual** y **bandas de confianza punto a punto**, e interpretarlas como punto a punto (no como banda simultánea).
5. Construir las curvas de **ambas fuentes sobre el mismo dominio anual** (ojo: estación es diaria, satélite es péntada → hay que llevarlas al mismo soporte temporal).

**Con qué**
- R: operaciones matriciales a mano (`solve`, `%*%`, `crossprod`), base spline construida explícitamente. Python: `numpy` para el álgebra.
- Construir las curvas con librerías (p. ej. `fda` en R, `scikit-fda` en Python) está permitido pero **baja a máximo 2** este ítem. Hazlo **a mano**.

**Trampas**
- Usar `fda`/`smooth.basis` para "ahorrar tiempo" → tope en 2. El ítem existe para demostrar que sabes construir la base.
- No reportar el valor de K/λ elegido por fuente, o mostrarlo sin la gráfica de ECM/VC/VCG.
- Olvidar la verificación de S (idempotencia, traza). Es un check numérico corto y da puntos seguros.
- Construir estación y satélite en dominios distintos: luego no se pueden comparar en la Fase 3.

### Fase 1b — Preprocesamiento y selección de datos (4.1, 6 %)

> Va de la mano con la Fase 1: decide qué estaciones y años entran **antes** de modelar. El enunciado lo exige explícitamente.

**Qué hacer**
1. Regla explícita para **valores imposibles** y **errores de digitación**.
2. Detectar **totales mensuales registrados en un solo día** y **ceros que en realidad son datos faltantes**.
3. Definir y **justificar** un **criterio de cobertura mínima** por estación y por año.
4. Decidir **justificadamente** si **agregas** datos (p. ej. registros fuera del periodo para estimar el FPCA, o imputación de vacíos) o si los **eliminas**.
5. **Tabla de trazabilidad:** número de estaciones y de **pares estación–año** antes y después de **cada** paso.
6. Ninguna decisión del preprocesamiento puede usar datos del **conjunto de prueba** (si no, hay fuga de información).

**Trampas**
- Eliminar/agregar sin criterio explícito o sin reportar cuántos → cae a 2.
- Tratar solo faltantes e imposibles, ignorando totales mensuales y ceros falsos → cae a 3.
- Usar los datos crudos sin depurar → 1.

### Fase 2 — Análisis exploratorio funcional (15 %)

**Qué hacer**
1. **Media funcional calculada paso a paso** y **verificada contra la función de librería** (sí: la calculas a mano y además compruebas que coincide con la librería).
2. **Varianza y desviación estándar funcional.**
3. **Covarianza y correlación funcional** de cada fuente.
4. **Derivadas** e interpretación del **inicio y fin de las temporadas de lluvia**.
5. **Profundidad** (**Fraiman–Muniz** y **MBD**) y **boxplot funcional**, con las curvas **atípicas identificadas por estación y año**.
6. **Interpretación climatológica** de cada resultado: régimen **bimodal**, **orografía**, **ENSO**.

**Con qué**
- R: `fda`, `fda.usc` (profundidades, boxplot funcional). Python: `scikit-fda`. Aquí sí puedes apoyarte en librerías para los descriptivos, **pero la media va verificada a mano**.

**Trampas**
- Analizar solo un subconjunto de años/estaciones **sin justificar** que es representativo de 1981–2013 → tope en 2.
- Describir las gráficas sin lectura climatológica → cae a 3.

### Fase 3 — Comparación funcional por fuente (15 %)

**Qué hacer**
1. **Emparejamiento estación–píxel documentado:** coordenadas, regla de asignación, y qué estaciones comparten píxel.
2. **Curva de diferencia** `d(t) = estación − satélite` con banda `d̄(t) ± 2·EE(t)`, e identificar los intervalos del año donde la banda **no contiene el cero** (ahí el sesgo es significativo).
3. **Correlación cruzada funcional** y su **diagonal**, interpretadas.
4. **Comparación de la covarianza** de cada fuente.
5. Comparación **por altitud o subregión** y **por fase ENSO**.
6. **Interpretación climatológica del sesgo satelital.**

**Trampas**
- Comparar **sin** el emparejamiento estación–píxel documentado → tope en 1 (es la trampa más cara de esta fase).

### Fase 4 — Red neuronal funcional, ablación, evaluación e interpretación (45 %)

Es casi la mitad de la nota. Se descompone en 4.2, 4.3, 4.4, 4.5 (y el 4.1 de arriba). La red tiene **tres bloques**; el codificador y el decodificador son **iguales en todos los modelos**, y **solo cambia la cabeza** — esa diferencia ES la ablación.

**4.2 — Red neuronal funcional (10 %)**
- **Codificador funcional:** la curva satelital `X(t)` se representa en base B-spline con coeficientes `c`. Cada **neurona funcional** calcula `z_k = ∫ w_k(t) X(t) dt = a_kᵀ W c`, donde `w_k(t)` es una función de pesos en B-splines con coeficientes `a_k` **aprendibles** y `W = ∫ φ(t) φ(t)ᵀ dt` es la **matriz de Gram** de la base (calculada por **cuadratura**). Usa entre **6 y 8 neuronas**. Al vector `z` se le agregan la **altitud de la estación** y el **índice ONI del año**.
- **Decodificador funcional FPCA:** `ŷ(t) = μ(t) + Σ_m s_m ψ_m(t)`, donde `μ(t)` y las funciones propias `ψ_m` salen del **FPCA de las curvas de estaciones del conjunto de entrenamiento** y se mantienen **congeladas**. **Reporta la varianza explicada** por las M componentes. **M entre 4 y 6**, justificado.
- **Pérdida:** norma L² funcional `(ĉ − c)ᵀ W (ĉ − c)` + **penalización de rugosidad** sobre las `w_k(t)`. Modela sobre `√P` o `log(1+P)`, y la reconstrucción final en mm debe ser **no negativa** (reporta el % de valores recortados).
- **Demostración + verificación numérica** (diferencia máxima **≤ 1e-4**) de que la **cabeza lineal con FPCA equivale a la regresión funcional función a función**: `β(s,t) = Σ_k Σ_m w_k(s) B_km ψ_m(t)`.

*Trampa:* presentar un **MLP sobre el vector de péntadas** como si fuera red funcional → tope en 1. La diferencia es que las neuronas integran la curva con la Gram, no toman péntadas sueltas.

**4.3 — Ablación KAN, MLP y cabeza lineal (11 %)**
- Las **tres cabezas** con la **misma arquitectura de codificador** y el **mismo decodificador FPCA**.
  - **Lineal:** `Σ_k β_k z_k` (equivale a la regresión funcional).
  - **KAN:** composición de funciones B-spline en las aristas, `φ(x) = w_b·SiLU(x) + w_s·Σ_i c_i B_i(x)`. Prueba si la corrección **no lineal** del sesgo mejora, manteniendo activaciones interpretables.
  - **MLP:** red densa. Techo de expresividad **sin** interpretabilidad; mide cuánto se pierde por exigir interpretabilidad en la KAN.
- **Parámetros de KAN y MLP igualados**: cuéntalos con `sum(p.numel())`, diferencia **≤ 10 %**, y **tabúlalos**.
- **3 semillas por configuración**, con media y desviación estándar.
- **Mismo presupuesto de entrenamiento** para todas: épocas máximas, parada temprana con validación, y nº de pruebas de hiperparámetros.
- **Verifica la KAN con una función sintética** (RMSE ≤ 1e-2) **antes** de usar datos reales.
- Una cabeza **supera** a otra **solo si** el IC bootstrap del 95 % de la **diferencia excluye el cero**.

*Trampa:* elegir hiperparámetros con el conjunto de prueba → tope en 1 (fuga). Cambiar otros bloques entre cabezas → cae a 3.

**4.4 — Evaluación (8 %)**
- **Partición temporal:** **1981–2005 / 2006–2009 / 2010–2013** + **estaciones reservadas por el docente**, sin fuga de información.
- **Comparar contra la curva satelital sin corregir** (es la línea base obligatoria — todo modelo se compara contra ella).
- **Las cuatro métricas:** error **L² relativo**, **KGE**, **sesgo del acumulado anual (%)** y **error en la fecha de los dos picos** de lluvia.
- **IC bootstrap del 95 %** (≥ **1000** remuestreos) + **Wilcoxon pareada por estación**.
- Resultados **por fase ENSO**, clasificada con el **ONI** y **citando la fuente** del ONI.
- **Predicciones de las estaciones reservadas** en CSV con el formato exacto **`station_id, year, day, value_mm`**, entregadas **antes de la sustentación**.

*Trampa:* **fuga de información** → arrastra 4.3 y 4.4 a máximo 2. Partición **aleatoria** de pares estación–año (en vez de temporal) → cae a 2.

**4.5 — Interpretación climatológica (10 %)**
- Grafica las **`w_k(t)`** (qué **ventana del año** usa cada neurona).
- Grafica las **funciones de activación de la primera capa de la KAN** en el rango de los datos.
- **Contrasta** con los coeficientes de la **cabeza lineal**.
- **Ajuste simbólico** de **al menos una** función de activación, con su **R²**.
- Lectura climatológica coherente (**bimodal, orografía, ENSO**) + análisis de los **peores casos** con sus curvas.

*Trampa:* interpretación que **contradice tus propias gráficas** → tope en 2. Gráficas sin interpretación → 1.

---

## 5. Diagramas

### (1) Pipeline general

```
 DATOS CRUDOS            CURVAS FUNCIONALES        ANÁLISIS                 INFORME
 ───────────            ──────────────────        ────────                 ───────
 Estaciones (diaria) ┐  Preproc 4.1 (depurar,  ┐  Fase 2: EDA funcional ┐
 Satélite (péntada)  ┼─▶ cobertura, trazas)    ┼─▶Fase 3: comparar      ┼─▶ PDF ≤20 pág
 ONI + altitud       ┘  Fase 1: base B-spline  │  fuentes               │   + 2 scripts
                        a mano → curva/año      ┘  Fase 4: KAN vs MLP    │   + CSV reservadas
                                                   vs lineal vs satélite ┘   + sustentación 40'
```

### (2) Forma / dimensiones de los datos

```
  Cada AÑO en una LOCALIZACIÓN = UNA CURVA funcional (no escalares sueltos)

        eje t (días/péntadas del año) ───────────────────▶
   1981 ┤   ╱‾‾╲      ╱‾‾╲         curva de lluvia del año (bimodal)
   1982 ┤  ╱    ╲    ╱    ╲
    ...                                   ── una curva por (fuente × estación/píxel × año) ──
   2013 ┤
        └ años 1981…2013   ×   estaciones (puntual)  /  satélite (píxel malla)

   Estación: resolución DIARIA, puntual     Satélite: resolución PÉNTADA, píxel regular
   → ambas llevadas al MISMO dominio anual antes de comparar
```

### (3) Reparto de la sustentación (40 min + preguntas)

```
  0'       5'            13'           21'                      38'   40'
  ├─intro──┼─Fase1+prep──┼─Fase2 EDA──┼─Fase3 comparación──────┼KAN─┼  → PREGUNTAS
  │ (todos)│  ~8'        │  ~8'        │  ~8'                    │ 4  │    del docente
  └ contexto, objetivo   └ construcción y exploración           └17' └  (a cualquiera,
    y datos                                   Fase 4 (45%) = más tiempo    sobre cualquier fase)
```
> La Fase 4 pesa 45 %: dale el mayor bloque. Todos deben poder responder de cualquier fase. ⚠️ El orden/tiempos exactos los decides tú; el único dato duro es **40 min**.

### (4) Reparto del trabajo entre integrantes (roles por fase)

```
  INTEGRANTE A ─▶ Fase 1 (construcción) + Preproc 4.1      ┐
  INTEGRANTE B ─▶ Fase 2 (EDA funcional)                   │  TODOS revisan
  INTEGRANTE C ─▶ Fase 3 (comparación fuentes)             ┼─ TODO y pueden
  INTEGRANTE D ─▶ Fase 4 (KAN/MLP/lineal, evaluación)      │  sustentar
  ROTA / César ─▶ integración, informe PDF, scripts, CSV   ┘  cualquier fase
```
> ⚠️ El nº de integrantes y nombres los pones tú. Regla 15/16: aunque te repartas, **cada uno estudia las cuatro fases** — el profesor pregunta cruzado y una mala respuesta resta −0,2 individual.

---

## 6. Checklist final antes de entregar

**Contenido técnico**
- [ ] Fase 1: base B-spline y mínimos cuadrados **a mano** (sin librerías); K/λ elegido por fuente con gráfica ECM/VC/VCG; verificación de S (idempotente, tr(S)=K); bandas punto a punto.
- [ ] Preproc 4.1: reglas de imposibles/digitación, totales mensuales, ceros falsos; criterio de cobertura justificado; **tabla de trazabilidad** (estaciones y pares estación–año antes/después de cada paso).
- [ ] Fase 2: media funcional a mano **verificada vs librería**; varianza/covarianza/correlación; derivadas; profundidad (Fraiman–Muniz, MBD) + boxplot funcional + atípicas; interpretación climatológica.
- [ ] Fase 3: emparejamiento estación–píxel documentado; curva d(t) con banda ±2EE; correlación cruzada; comparación por altitud y fase ENSO.
- [ ] 4.2: codificador con Gram por cuadratura; decodificador FPCA solo con train + varianza explicada; pérdida L² con W + rugosidad; **demostración y verificación numérica (≤1e-4)** lineal ≡ regresión funcional; reconstrucción no negativa + % recortado.
- [ ] 4.3: 3 cabezas misma arquitectura; params KAN/MLP ≤10 % con `sum(p.numel())`; 3 semillas (media±std); mismo presupuesto; KAN verificada con función sintética; conclusiones solo si el IC95% de la diferencia excluye 0.
- [ ] 4.4: partición 1981–2005/2006–2009/2010–2013 + estaciones reservadas, sin fuga; comparación vs satélite sin corregir; 4 métricas; bootstrap ≥1000 + Wilcoxon; resultados por fase ENSO (ONI citado).
- [ ] 4.5: w_k(t) y activaciones KAN graficadas; contraste con cabeza lineal; ajuste simbólico con R²; peores casos.

**Entrega y formato**
- [ ] CSV de estaciones reservadas con columnas exactas `station_id, year, day, value_mm`.
- [ ] **Todos los integrantes** aparecen dentro del documento.
- [ ] PDF a **una columna**, **≤ 20 páginas** incluidas referencias; tablas/figuras numeradas, tituladas, citadas e interpretadas.
- [ ] Nombre del archivo exactamente **`Proyecto final AnalíticaDeDatos`**.
- [ ] **Un** script R + **un** script Python, reproducibles de principio a fin (rutas relativas, semillas fijas, versiones de paquetes), comprimidos en **zip/rar**.
- [ ] Referencias para métodos y fuentes de datos (estaciones, satélite, ONI).
- [ ] Si se usó IA: **Word aparte** con herramienta + prompts.
- [ ] **Subir al campus** (plataforma institucional, no correo ni Drive) **antes del domingo 22-nov 19:00**.
- [ ] Ensayar la sustentación (40 min) y que **cada integrante** pueda defender **cualquier** fase.
