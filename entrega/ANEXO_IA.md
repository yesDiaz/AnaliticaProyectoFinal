---
title: "Anexo — Declaración de uso de inteligencia artificial generativa"
subtitle: "Proyecto final Analítica de Datos · UAO 2026-2S"
---

# 1. Declaración

En cumplimiento de la condición de entrega del proyecto, se declara el uso de una herramienta
de inteligencia artificial generativa como **asistente de programación y de redacción** en la
elaboración de este trabajo.

- **Herramienta:** Claude (Anthropic), familia Claude 5 (Opus), operada como agente de código
  en terminal con acceso a la carpeta del proyecto.
- **Fechas:** 14-15 de noviembre (primera versión), 19 de noviembre (migración del pipeline a
  base B-spline) y 20-21 de noviembre de 2026 (incorporación de las correcciones de las
  asesorías).
- **Alcance:** implementación de código, ejecución de las corridas y borradores de texto. **No**
  se usó para producir resultados numéricos: toda cifra de este informe proviene de una
  ejecución real registrada en log.

# 2. Qué define el grupo y qué no se delegó

La metodología de este trabajo **no la propuso la herramienta**. Proviene de tres fuentes, en
este orden:

1. **El enunciado del docente**, que fija la base B-spline construida a mano, la selección de
   K por ECM/VC/VCG, la verificación de la matriz de suavizado, el emparejamiento
   estación-píxel, la arquitectura del codificador funcional con matriz de Gram y la ablación
   KAN/MLP/lineal con parámetros igualados y tres semillas.
2. **Las asesorías del 19 y 20 de noviembre**, de donde salen las nueve correcciones de
   preprocesamiento (umbral de faltantes, cero ≠ faltante, rachas contra estaciones vecinas,
   comparación de métodos de imputación, discriminación por altura, altitud por SRTM,
   unidades del satélite) y los ajustes de la fase 4 (partición temporal con tramos ENSO,
   FPCA por régimen, bootstrap, poda de la KAN, aporte por covariable).
3. **Decisiones argumentadas del grupo**, que están sustentadas en el informe y son
   defendibles una por una:
   - adoptar el criterio nacional de faltantes en vez del de la OMM, porque este último dejaba
     la vertiente Pacífica con 6 de 28 estaciones y el régimen del Valle no se explica sin
     esa zona;
   - tomar la **estación** como unidad de análisis en la comparación por fuente, para no
     inflar la precisión con 23 años de la misma estación;
   - reportar como modelo principal el que quedó fijado por validación cruzada **antes** de
     mirar el conjunto de prueba, aun sabiendo que otra configuración obtuvo mejor error en
     prueba;
   - declarar como no verificada la fuente del criterio del 45 % en lugar de atribuirla a una
     cita que no se pudo abrir.

# 3. Controles aplicados al trabajo asistido

Para que el uso de la herramienta no comprometiera la validez de los resultados, se fijaron
cinco controles, y todos quedan evidenciados en los entregables:

| Control | Evidencia en la entrega |
|---|---|
| Ninguna cifra escrita a mano: toda tabla y figura se genera por código | `tablas/` (120 archivos), `figuras/` (37), logs en `cache/log_*.txt` |
| Doble implementación independiente en Python y R, con comparación explícita | `tablas_R/TabR_comparacion_*.csv` (mismas 141 estaciones, mismos 2945 pares, mismo K) |
| Lo construido a mano se contrasta contra librería especializada, que **no** se usa para construir | base B-spline vs `fda`/`scikit-fda`: diferencia ~1e-13; media funcional: 1,8·10⁻¹⁵ |
| Cada referencia bibliográfica se abrió y se marcó verificada o no verificada | `REFERENCIAS_VERIFICADAS.md` |
| Lo que no se pudo calcular o verificar se reporta como limitación, no se completa | §limitaciones del informe; criterio del 45 % declarado sin fuente |

# 4. Errores de la herramienta detectados y corregidos

Se documentan porque muestran que el resultado fue revisado y no aceptado tal como salió:

- **Escala de modelación (desigualdad de Jensen).** La primera versión suavizaba la raíz de
  los datos diarios y elevaba al cuadrado la curva suavizada, lo que destruía la varianza
  diaria y hundía el total anual. Se detectó por inconsistencia entre fases: el sesgo del
  satélite salía en +281 % cuando la fase 3 daba +10 %. Se corrigió modelando la raíz de la
  curva suavizada (informe, §5.1).
- **Malla de la KAN.** La verificación sintética no alcanzaba el RMSE ≤ 10⁻² exigido con malla
  G = 10 (0,022). Se aumentó a G = 20 (0,0019).
- **Cita no localizable.** El criterio "nacional > 45 %" mencionado en clase no pudo ser
  respaldado con fuente; en vez de atribuirlo, el informe lo declara no verificado y lo
  sustenta con IDEAM (2018) y Cruz-Roa y Barrios (2018).

# 5. Apéndice — instrucciones dadas a la herramienta

Se transcriben las instrucciones de trabajo, como pide la condición de entrega. Nótese que su
contenido técnico es el del enunciado y el de las asesorías.

**Instrucción 1 (14-nov, versión inicial).** "Construye los datos funcionales de las dos
fuentes sobre el mismo dominio anual con base B-spline construida a mano, mínimos cuadrados
con Cholesky propia, selección de K con ECM, VC y VCG, verificación de la matriz de suavizado
y bandas punto a punto; haz el AED funcional verificando la media contra la librería; empareja
estación-píxel y compara las fuentes; monta la red funcional con cabezas lineal, KAN y MLP con
parámetros igualados, tres semillas, bootstrap y Wilcoxon."

**Instrucción 2 (19-nov, reanudación).** "Pipeline único `entrega/proyecto_final.py` con fases
0 a 3 y `proyecto_final.R` que lo reproduzca; ejecutar con el entorno virtual del proyecto."

**Instrucción 3 (20-nov, correcciones de las asesorías; resumen fiel).** "Aplica las
correcciones de las asesorías del 19 y 20 de noviembre, leyendo completas las notas y
transcripciones: (1) inventario de porcentaje de faltantes por estación y por año con umbral
justificado (~25 %); (2) comparar las reglas OMM, NOAA y nacional y elegir una, separando la
regla anual de la del periodo; (3) distinguir cero de faltante con prueba explícita; (4)
rachas de 30 días en cero comparadas con dos estaciones vecinas a ~20 km; (5) imputación
comparando IDW y kriging ordinario, opcionalmente ponderación por correlación; (6)
discriminación por altura antes de imputar y relación lluvia-altura; (7) altitud faltante con
SRTM 90 m validada contra el catálogo; (8) documentar resolución temporal y unidades del
satélite; (9) lluvia puntual por año. Fase 4: partición 1981-1996 con validación cruzada
temporal y prueba 1997-2003 en tres tramos ENSO; FPCA por régimen; bootstrap además de
Wilcoxon; poda de la KAN con L1 más entropía y CV score; activaciones y autoencoder disperso
para la MLP; aporte de cada covariable; ANOVA funcional sin grupos de una sola estación y
justificación de Welch frente a la t pareada. No inventar resultados: todo número debe salir
de una corrida con log."

**Instrucción 4 (21-nov, búsqueda bibliográfica; resumen fiel).** "Busca y abre referencias
reales: artículos de modelación de lluvia en Colombia y el Valle del Cauca con su sección de
datos y preprocesamiento; el estudio colombiano del criterio > 45 %; reglas OMM (WMO-No. 1203
y No. 100) y NOAA; unidades de CHIRPS pentadal; SRTM; IDW, kriging y ponderación por
correlación; KAN, autoencoders dispersos, bootstrap, validación cruzada temporal, Welch, ANOVA
funcional e índice ONI. Marca cada una como verificada o no verificada."
