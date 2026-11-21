# GUION DE SUSTENTACIÓN — miércoles (presencial, 40 min + preguntas)

**Reglas del profesor:** prohibido leer (será "súper estricto"); ir a lo puntual; no profundizar en el histórico (no contar versiones anteriores); mostrar el aporte de cada covariable; todos preparados para preguntas de cualquier fase.
**Ensayo:** lunes y martes, cronometrado, con los cambios de turno practicados (que nadie se pise).
**Diapositivas:** una idea por lámina, solo cifra + figura. Este guion es para estudiar, **no** para llevar en la mano.

Reparto (cambiar nombres si el grupo decide otro orden):

| Bloque | Quién | Tiempo | Láminas |
|---|---|---|---|
| A. Problema, datos y preprocesamiento | **Integrante 1** | 0-13 min | 1-7 |
| B. Datos funcionales, EDA y comparación de fuentes | **Integrante 2** | 13-24 min | 8-13 |
| C. Red funcional, resultados y covariables | **Integrante 3** | 24-38 min | 14-21 |
| Cierre (los tres, 1 frase cada uno) | todos | 38-40 min | 22 |

---

## Bloque A — Integrante 1 (13 min)

**Lámina 1. Pregunta (1 min).** ¿Cuánto se equivoca el satélite frente a las estaciones del Valle y se puede corregir con una red funcional? Cada año de lluvia de una estación es una curva.

**Lámina 2. Datos (1,5 min).** 323 estaciones CVC con lluvia diaria 1981-2003; 178 tienen coordenadas en la malla. Satélite tipo CHIRPS, 0,05°, péntadas. **Dato clave:** verificamos que la péntada es un **total (mm/péntada), no un promedio**: el satélite da 25,4 mm frente a 24,4 mm de suma de la estación (razón 1,04). Por eso se divide entre los días de la péntada. Tope de 200 mm, sin −9999.

**Lámina 3. Inventario (2 min)** (Fig. 1). Tabla de % de faltantes por estación y año. Cero es dato, no faltante.

**Lámina 4. Reglas (3 min)** (Tabla 1). OMM deja 77 estaciones y solo **6 de 28 del Pacífico**; nacional deja 141 y **22 del Pacífico**. Elegimos la nacional por cobertura de todo el Valle. Separamos la regla **anual** (≤25 % de faltantes; si no, el año se descarta) de la del **periodo** (>45 % de datos). Decirlo con honestidad si preguntan: la fuente del 45 % no la encontramos; nos apoyamos en el 25 % del profesor y en los umbrales del IDEAM (15 % por mes).

**Lámina 5. Cero ≠ faltante y rachas (2 min)** (Fig. 2). Prueba en el código: 0 ceros pasaron a NaN sin regla. Rachas ≥30 días en cero: 2 vecinas a ≤25 km. Si las vecinas llovieron, es falla del sensor (49 casos); si estaban secas, es sequía real (840, se conservan). La racha seca más larga conservada es de 96 días, en 1982 (año Niño).

**Lámina 6. Altura (1,5 min)** (Fig. 3). SRTM 90 m para 87 estaciones sin altitud; con las 126 validables, mediana del error 36 m y r = 0,93. La lluvia frente a la altura **no es lineal**: en el Pacífico baja con la altura, en la zona andina sube un poco.

**Lámina 7. Imputación (2 min)** (Tabla 2, Fig. 4). Primero la banda de altura (|Δz| ≤ 400 m), luego el radio (20 km). IDW frente a CSW frente a kriging: el kriging gana día a día y el CSW en totales de 30 días con sesgo de 0,1 %. Elegimos CSW, con kriging de radio doble de respaldo. Solo 1,1-1,2 % de los días imputados. **Pase:** "Con 141 estaciones y 2945 curvas, [Integrante 2] muestra qué dicen las curvas."

## Bloque B — Integrante 2 (11 min)

**Lámina 8. Base a mano (2 min).** B-splines por Cox-de Boor, mínimos cuadrados con Cholesky propio. K por VCG: 6 para estación y 11 para satélite; tr(S) = K. R reproduce K y los coeficientes (diferencia 7·10⁻¹⁴).

**Lámina 9. EDA (2 min).** Régimen bimodal: picos el 22 de abril y el 12 de noviembre, valle el 10 de agosto. Mayor variabilidad en enero (efecto ENSO). La anomalía anual va contra el ONI (ρ = −0,55).

**Lámina 10. FPCA por régimen (1,5 min)** (Fig. 5). ψ₁ es igual en los tres regímenes; ψ₂ y ψ₃ cambian (coseno de 0,08 en La Niña). Por eso **no se mezclan** Niño, Niña y neutro en el FPCA.

**Lámina 11. Diferencia estación − satélite (2 min)** (Fig. 6). Enero-marzo: la estación supera al satélite; mayo-junio: el satélite supera a la estación. Sesgo mediano del total: +10,4 %. Aquí la prueba pareada **sí** aplica (la estación y su píxel son el mismo sitio y año).

**Lámina 12. ENSO (1 min).** Con El Niño el satélite sobreestima (+22,5 %); con La Niña subestima (−5,9 %). El satélite atenúa la señal ENSO.

**Lámina 13. ANOVA funcional (2,5 min)** (Fig. 7). Cuatro subregiones con 22, 48, 56 y 15 estaciones: ninguna con una sola. F(t) con permutaciones: difieren todo el año (p = 0,0005). **Por qué Welch y no la t pareada:** son estaciones distintas, poblaciones independientes, sin pares naturales (ni el mismo n), y con varianzas hasta 100 veces distintas. **Pase:** "¿Se puede corregir ese sesgo? [Integrante 3]."

## Bloque C — Integrante 3 (14 min)

**Lámina 14. Arquitectura (2 min).** Codificador: 6 neuronas z_k = ∫w_k(t)x(t)dt con la Gram W. + altitud + ONI. Cabeza lineal / MLP / KAN (395 frente a 397 parámetros). Decodificador FPCA **del régimen del año**, M = 5 (≥96 % de varianza). Lineal = regresión funcional (diferencia 0).

**Lámina 15. Partición (1,5 min).** Entrenamiento 1981-1996 con validación cruzada temporal de ventana creciente (sin ver el futuro). Prueba 1997-2003 en tres tramos: Niño 97-98, Niña 99-00, neutro 01-03. 14 estaciones reservadas.

**Lámina 16. Resultado global (2 min)** (Tabla 5). Satélite 0,461 → MLP 0,431 → lineal 0,444 → KAN 0,495. El sesgo del total baja de +17 % a +4,6 % (lineal). **Bootstrap:** MLP frente a satélite IC [−0,057; −0,000]; **Wilcoxon p = 0,30**. Mejora marginal: decirlo así.

**Lámina 17. Por tramo (2 min)** (Fig. 8). Niño: todas mejoran y la KAN es la mejor (−0,11 frente al satélite). Niña: todas empeoran; la KAN amplifica el pico de noviembre y no ve el inicio húmedo de 1999-2000. **Atenuación** de 0,83 en el Niño y **amplificación** de 1,33 en la Niña (KAN).

**Lámina 18. Aporte de cada covariable (2,5 min)** (Tabla 6, Fig. 28). Esta lámina la pidió el profesor.
- Satélite solo: 0,414 (KAN).
- Altitud sola: la peor.
- **Satélite + altitud: la mejor (0,400; 13 % menos error que el satélite).**
- Agregar el ONI **empeora** (KAN 0,495): solo hay 16 años distintos de ONI en entrenamiento y la red lo memoriza.

**Lámina 19. KAN: activaciones y poda (1,5 min)** (Fig. 11). Activaciones tipo tanh (R² = 1,00 en la principal). La poda L1 + entropía quita 8 de 39 aristas pero empeora el CV score (0,278 → 0,301): no estaba sobreparametrizada. La KAN casi solo "aporta" lluvia (6811 días de aporte frente a 1140 de atenuación) y por eso sobreestima en la Niña.

**Lámina 20. MLP: activaciones y SAE (1 min)** (Fig. 12). Neuronas "de nivel de lluvia" (r ≈ 0,8) y neuronas "de ONI" (|r| ≈ 0,9). El SAE no encontró rasgos que separen sequía de lluvia.

**Lámina 21. Limitaciones (0,5 min).** Pacífico con más imputación y satélite censurado en 200; Niña extrema en la prueba; fuente del 45 % no encontrada.

**Lámina 22. Cierre (los tres, 2 min).** Una frase cada uno:
- Integrante 1: "El dato bien construido (inventario, ceros, altura) es la base."
- Integrante 2: "El satélite capta el ciclo, pero atenúa la variación entre años y el ENSO."
- Integrante 3: "La corrección sirve en el Niño y en el total anual; la altitud aporta, el ONI sobreajusta."

---

## Preguntas probables (todos deben poder responderlas)

1. **¿Por qué no la regla OMM?** Deja 6 de 28 estaciones del Pacífico, aplica 10 días por mes (no por año) y está pensada para normales de 30 años; nosotros tenemos 23.
2. **¿Cero o faltante?** 0 mm = no llovió. Solo se anula con regla (racha ≥30 días con vecinas lluviosas, bloque copiado, total mensual en un día). Hay un `assert` en el código.
3. **¿Por qué CSW y no kriging?** El kriging gana a escala diaria (RMSE 8,94 frente a 9,12); el CSW en el total de 30 días (67,2 frente a 69,7), que es la escala de una curva con K = 6. El kriging se usa de respaldo con radio de 40 km.
4. **¿El satélite es promedio o suma?** Suma (mm/péntada): razón 1,04 con la suma de la estación y cabecera oficial de CHIRPS.
5. **¿Hay fuga de información?** No: umbrales de control de calidad con años ≤1995; correlogramas, FPCA y estandarización con datos de entrenamiento de cada fold; hiperparámetros por CV temporal; modelo para las estaciones reservadas elegido por CV, no por la prueba.
6. **¿Por qué Wilcoxon y bootstrap dan distinto (MLP frente a satélite)?** El bootstrap prueba la media de la diferencia, dominada por pocas estaciones muy mejoradas; Wilcoxon prueba los rangos (si mejora en la mayoría), y solo mejora en el 47 % de las estaciones.
7. **¿Por qué la KAN falla en la Niña?** Extrapola el ONI fuera del rango que vio (solo 16 valores) y sus splines casi solo aportan lluvia; amplifica el segundo pico. Sin el ONI su error baja 0,094.
8. **¿Por qué la t pareada no sirve entre subregiones?** Son estaciones distintas (poblaciones independientes); no hay pares naturales. Welch admite varianzas y tamaños distintos.
9. **¿Qué es `value_mm`?** La lluvia diaria reconstruida por el modelo (mm/día) para ese día de ese año y esa estación.
