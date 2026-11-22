---
title: "Anexo — Declaración de uso de inteligencia artificial generativa"
subtitle: "Proyecto final Analítica de Datos · UAO 2026-2S"
---

# 1. Herramienta

- **Herramienta:** Claude (Anthropic), modelo de la familia Claude 5 (Opus), usado como agente de código en terminal (Claude Code) con acceso a la carpeta del proyecto. Un subagente de la misma herramienta buscó y abrió las referencias bibliográficas.
- **Fechas de uso:** 14-15 de noviembre (primera versión), 19 de noviembre (reanudación del pipeline en B-splines) y 20-21 de noviembre de 2026 (correcciones de las asesorías).

# 2. Para qué se usó y para qué no

- **Sí:** escribir y depurar el código de `proyecto_final.py` y `proyecto_final.R`, correr las fases, redactar borradores del informe, del inventario de datos y del guion, y buscar referencias (cada una se abrió y se marcó como verificada o no en `REFERENCIAS_VERIFICADAS.md`).
- **No:** ninguna cifra del informe la escribió el modelo. Todas salen de corridas reales con log guardado (`cache/log_*.txt`) y de las tablas CSV de `tablas/`. Lo que no se pudo calcular aparece como limitación.
- **Revisión humana:** el grupo debe revisar y poder explicar cada decisión (regla de estaciones, imputación, partición, modelos) antes de la sustentación.

# 3. Prompts usados (texto literal o resumen fiel)

**Prompt 1 (14-nov, versión inicial).** "Construye los datos funcionales de las dos fuentes sobre el mismo dominio anual con base B-spline construida a mano, mínimos cuadrados con Cholesky propia, selección de K con ECM, VC y VCG, verificación de la matriz de suavizado y bandas punto a punto; haz el EDA funcional verificando la media contra la librería; empareja estación-píxel y compara las fuentes; monta la red funcional con cabezas lineal, KAN y MLP con parámetros igualados, tres semillas, bootstrap y Wilcoxon."

**Prompt 2 (19-nov, reanudación).** "Reanuda el proyecto: pipeline único `entrega/proyecto_final.py` con fases 0 a 3 y `proyecto_final.R` que lo reproduzca; ejecutar con el venv; dejar el estado en PROGRESO.md."

**Prompt 3 (20-nov, correcciones; resumen fiel del texto entregado al agente).** "Aplica las correcciones de las asesorías del 19 y 20 de noviembre (lee completas las notas y transcripciones): (1) inventario de % de faltantes por estación y por año con umbral justificado (~25 %); (2) comparar las reglas OMM, NOAA y nacional y elegir una, separando la regla anual de la del periodo; (3) cero ≠ faltante con prueba; (4) rachas de 30 días en cero comparadas con 2 estaciones a ~20 km; (5) imputación comparando IDW y kriging ordinario (opcional CSW); (6) discriminación por altura antes de imputar y relación lluvia-altura; (7) altitud faltante con SRTM 90 m validada; (8) documentar resolución temporal y unidades del satélite; (9) lluvia puntual por año. Fase 4: partición 1981-1996 con validación cruzada temporal y prueba 1997-2003 en tres tramos ENSO; FPCA por régimen; bootstrap además de Wilcoxon; poda de la KAN con L1 + entropía y CV score; activaciones y sparse autoencoder para la MLP; aporte de cada covariable; ANOVA funcional sin grupos de una estación y justificar Welch frente a la t pareada. Entregables: informe PDF ≤20 páginas con referencias reales, este anexo, inventario de datos, guion de sustentación para 3 personas y ZIP de scripts. No inventar resultados: todo número sale de una corrida con log."

**Prompt 4 (subagente de referencias, 21-nov; resumen fiel).** "Busca y abre referencias reales: 1-3 artículos de modelación de lluvia en Colombia/Valle del Cauca con su sección de datos y preprocesamiento; el estudio colombiano del criterio >45 % de datos; reglas OMM (WMO-No. 1203 y No. 100) y NOAA; unidades de CHIRPS pentadal; SRTM; IDW, kriging y ponderación por correlación; KAN, sparse autoencoders, bootstrap, validación cruzada temporal, Welch, ANOVA funcional y ONI. Marca cada una como verificada o no."

# 4. Errores del asistente detectados y corregidos

- La primera métrica del satélite sin corregir comparaba curvas elevadas al cuadrado después de suavizar la raíz de los datos diarios; eso bajaba artificialmente el total observado (desigualdad de Jensen). Se detectó porque el sesgo del satélite salió en +281 % cuando la fase 3 daba +10 %. Se corrigió modelando la raíz de la curva suavizada (ver informe, §5.1).
- La verificación sintética de la KAN no llegó a RMSE ≤ 10⁻² con malla G = 10 (0,022); se aumentó a G = 20 (0,0019).
- El criterio "nacional >45 %" se citó en clase, pero el subagente no encontró la fuente; el informe lo declara como no verificado.
