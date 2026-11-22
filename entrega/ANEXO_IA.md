---
title: "Anexo — Declaración de uso de inteligencia artificial generativa"
subtitle: "Proyecto final Analítica de Datos · UAO 2026-2S"
---

# 1. Declaración

Se declara el uso de una herramienta de IA generativa (Claude, Anthropic) como **asistente de
implementación** en puntos concretos del proyecto, durante el 14-15, 19 y 20-21 de noviembre
de 2026.

# 2. Alcance delimitado

| Tarea | Responsable |
|---|---|
| Diseño metodológico (base B-spline a mano, selección de K, codificador funcional con Gram, ablación con parámetros igualados) | Enunciado del docente |
| Correcciones de preprocesamiento y de la fase 4 | Asesorías del 19 y 20 de noviembre |
| Decisiones de criterio: regla de faltantes, unidad de análisis, modelo reportado, citas no verificables | Grupo |
| Interpretación climatológica y defensa de resultados | Grupo |
| Escritura del código que ejecuta lo anterior, en Python y R | Asistido con IA |
| Borradores de redacción y búsqueda de referencias (verificadas una por una) | Asistido con IA |
| Resultados numéricos | Ninguno: salen de corridas con log |

Ninguna cifra del informe fue escrita por la herramienta. Las 120 tablas y 37 figuras se
generan al ejecutar los scripts, con registro en `cache/log_*.txt`, y la implementación hecha
a mano se contrasta contra `fda` y `scikit-fda` (diferencia ~1e-13) sin usarlas para construir.
El código tiene dos implementaciones independientes, en Python y en R, que coinciden en las
141 estaciones, los 2945 pares estación-año y el K elegido.

# 3. Control de calidad

Se revisó el trabajo asistido y se corrigieron tres errores de la herramienta: la escala de
modelación de la raíz (desigualdad de Jensen, informe §5.1), la malla de verificación de la
KAN (G = 10 → 20 para cumplir RMSE ≤ 10⁻²) y la atribución de una cita que no se pudo abrir,
hoy declarada como no verificada.

# 4. Instrucciones dadas a la herramienta

Transcritas en resumen fiel, como pide la condición de entrega. Su contenido técnico es el del
enunciado y el de las asesorías.

1. **14-nov:** implementar las fases 1 a 4 según el enunciado — base B-spline construida a
   mano con Cholesky propia, selección de K por ECM/VC/VCG, verificación de la matriz de
   suavizado, AED funcional, emparejamiento estación-píxel, red funcional con cabezas lineal,
   KAN y MLP con parámetros igualados, tres semillas, bootstrap y Wilcoxon.
2. **19-nov:** unificar el pipeline en `proyecto_final.py` y reproducirlo en `proyecto_final.R`.
3. **20-nov:** aplicar las nueve correcciones de preprocesamiento de las asesorías (umbral de
   faltantes, cero ≠ faltante, rachas contra estaciones vecinas, comparación de métodos de
   imputación, discriminación por altura, altitud SRTM validada, unidades del satélite) y los
   ajustes de la fase 4 (partición temporal con tramos ENSO, FPCA por régimen, bootstrap, poda
   de la KAN, aporte por covariable). Condición fijada: ningún número sin corrida con log.
4. **21-nov:** buscar y abrir las referencias bibliográficas, marcando cada una como verificada
   o no verificada.
