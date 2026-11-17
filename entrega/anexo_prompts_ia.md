# Anexo — Uso de inteligencia artificial en el proyecto

Proyecto Final de Analítica de Datos · UAO 2026-2S

Este anexo documenta, como exige el reglamento del curso, el uso de asistentes de IA
generativa en el desarrollo del proyecto: para qué se usaron, con qué instrucciones y **cómo
se verificó cada resultado** para no depender de la salida del modelo.

## 1. Herramienta y alcance

- **Herramienta:** asistente de código basado en modelos Claude (Anthropic), usado desde
  terminal con acceso al sistema de archivos del proyecto.
- **Para qué se usó:** redacción de los scripts de las cinco fases, diagnóstico de errores
  de ejecución, búsqueda de fuentes de datos auxiliares (Catálogo Nacional de Estaciones del
  IDEAM e índice Niño 3.4 de NOAA/PSL) y redacción del informe.
- **Para qué NO se usó:** no se generó ningún dato sintético, ninguna cifra del informe
  proviene del modelo. Todas las cifras citadas salen de la ejecución real de los scripts y
  están en `entrega/log_fase*.txt` y en los CSV de `entrega/tablas/`.

## 2. Instrucciones principales (resumen literal de lo pedido)

1. *«Construye los datos funcionales de las dos fuentes sobre el mismo dominio anual: base
   B-spline por Cox-de Boor y base de Fourier, ambas construidas explícitamente; mínimos
   cuadrados penalizados resueltos con una factorización de Cholesky propia, sin `lm()` ni
   `lstsq`; selecciona K y λ con ECM, validación cruzada y VCG; verifica la matriz de
   suavizado (simetría, idempotencia, tr(S)=K) y calcula bandas punto a punto.»*
2. *«Haz el EDA funcional calculando todo a mano: media con acumulador explícito, varianza,
   covarianza y correlación funcional, derivadas analíticas de la base, profundidades
   Fraiman–Muniz y MBD, boxplot funcional y FPCA; cada estadístico debe verificarse contra
   una referencia independiente (identidad analítica, `numpy`, `skfda`) y el error de
   verificación debe quedar impreso en el log.»*
3. *«Empareja estaciones con píxeles y compara las dos fuentes: curva de diferencia con banda
   bootstrap, correlación cruzada funcional, desglose por altitud y subregión.»*
4. *«Monta la red funcional: codificador con la matriz de Gram por cuadratura, decodificador
   FPCA estimado solo con entrenamiento, pérdida L² con W más penalización de rugosidad, y
   tres cabezas (lineal, MLP, KAN) con el número de parámetros igualado a menos del 10 %.
   Demuestra numéricamente que la cabeza lineal coincide con la regresión funcional cerrada
   con tolerancia 1e-4, valida la KAN con una función sintética, usa tres semillas, evalúa
   con las cuatro métricas, bootstrap de 1000 remuestreos y Wilcoxon pareada por estación, y
   no concluyas que una cabeza gana si el IC 95 % de la diferencia incluye el cero.»*
5. *«No inventes números: si algo falla, dilo y deja la evidencia en el log.»*

## 3. Problemas que el asistente detectó y cómo se resolvieron

Los dos hallazgos metodológicos más importantes del proyecto salieron de revisar críticamente
resultados que *parecían* correctos:

1. **Los `STATION_ID` no son códigos del CNE del IDEAM.** El primer intento de obtener
   coordenadas emparejaba 134 de 186 estaciones con una regla de reescritura de dígitos. La
   verificación geográfica (distribución por departamento y pertenencia a la malla satelital)
   mostró que el emparejamiento era **espurio**, y se descartó en lugar de usarlo. Evidencia:
   `aux/NOTA_CODIGOS_ESTACIONES.md`, `aux/diag_codigos.py`, `aux/estaciones_cne_match.csv`.
2. **Las dos fuentes no estaban en la misma escala.** Al comparar estaciones con satélite
   aparecía una razón de 3,32, implausible como sesgo físico. El diagnóstico fue que la
   Fase 1 ajustó las estaciones sobre √P diario y el satélite sobre √P pentadal, de modo que
   la desigualdad de Jensen explicaba un factor 2,14. Se reajustaron las estaciones con el
   mismo preprocesamiento del satélite (`fase3_0_pentadas.py`) y la razón bajó a 1,62.

Un tercer caso es de higiene numérica: la verificación «cabeza lineal ≡ regresión funcional»
fallaba (2,8e-02) porque el entrenamiento se detenía por *early stopping* sobre validación
mientras la solución cerrada minimiza el error de entrenamiento; y luego se quedaba en
2,5e-04 por precisión de `float32`. Con LBFGS a convergencia en `float64` la diferencia bajó
a **6,7e-09**. El fallo se documentó en lugar de relajar la tolerancia.

## 4. Verificaciones independientes del trabajo asistido

Ninguna afirmación del informe depende de la palabra del modelo:

| Qué se verificó | Contra qué | Resultado |
|---|---|---|
| Media funcional | identidad analítica `Φ(t)c̄`, `numpy`, `skfda` | 6,2e-15 |
| Profundidades FM y MBD propias | `skfda` (`IntegratedDepth`, `ModifiedBandDepth`) | 1,1e-4 / 8,1e-5 |
| FPCA propio (autovalores generalizados con Gram) | `skfda.FPCA` | correlación de scores 1,000000 |
| Cabeza lineal entrenada | regresión funcional cerrada con Cholesky propia | 6,7e-09 |
| KAN | función sintética `sin(πx₁)+x₂²` | RMSE 0,0026 (sd 0,764) |
| Matriz de suavizado | simetría, idempotencia, tr(S)=K | 3,5e-17 / 5,000000 |
| Toda la Fase 1 | **reimplementación independiente en R** (`proyecto_final_R.R`) | ver `tablas/R_verificacion_python.csv` |
| Emparejamiento estación–píxel | catálogo oficial de coordenadas de la CVC | el criterio de máxima correlación falló en 38/38 casos |
| Integral del codificador funcional | cuadratura de Simpson de 400.003 nodos | 1,1e-13 |
| Activaciones de la KAN | ajuste simbólico contra 8 formas funcionales | R² = 0,9871 (cúbica) |

La reimplementación en R de la Fase 1 es deliberadamente redundante: es la forma de
comprobar que el resultado no depende de una librería ni de una implementación concreta.

## 5. Declaración

El diseño de las fases, las decisiones metodológicas (base de Fourier frente a B-spline,
reajuste pentadal, descartar el emparejamiento espurio de códigos, rehacer la
georreferenciación con el catálogo oficial publicado el 9 de noviembre, reportar que el
codificador fijo generaliza mejor que el aprendible en lugar de ocultarlo, y tratar las
atípicas del Pacífico como régimen y no como error) y la interpretación climatológica fueron
discutidas y asumidas por el equipo. El asistente se usó como herramienta de implementación y de revisión
crítica; todos los resultados reportados son reproducibles ejecutando
`proyecto_final_python.py` y `proyecto_final_R.R`.
