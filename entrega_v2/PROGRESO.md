# PROGRESO — proyecto final Analítica (mark02, reanudación 19-nov-2026 22:50)

Ejecutar desde `entrega/`: `../venv/bin/python proyecto_final.py [fase0 fase1 fase2 fase3]`

- [x] Enunciado leído (datos/Proyecto_Final_Analitica.docx) y scripts del docente revisados.
- [ ] INVENTARIO_DATOS.md
- [x] Fase 0 preprocesamiento (Python) → tablas/Tab01..Tab06 (+Tab02b, Tab04b, Tab04c), figuras/Fig01, Fig02,
  cache/fase0_matriz_diaria.npz. 323→274 estaciones, 5514→4783 pares (fase 3: 161 est / 2936 pares).
- [x] Fase 1 (Python) → Tab07..Tab12, Fig03..Fig07, cache/fase1_coef_*.npz. K_est=6, K_sat=11 (VCG=VC).
- [x] Fase 2 (Python) → Tab13..Tab23, Fig08..Fig14, cache/fase2_profundidades_estaciones.csv.
- [x] Fase 3 (Python) → Tab24..Tab32 (+Tab26b sensibilidad K), Fig15..Fig20, cache/fase3_pares.csv.
- [x] proyecto_final.R (fases 0-3) → tablas_R/, figuras/R_*.png. Reproduce a Python: trazabilidad idéntica en 11 pasos,
  mismo K y VCG, coef. = fda::smooth.basis (1e-13), MBD/atípicas idénticas, fase 3 idéntica. Tiempo ≈ 5 min.
- [ ] INFORME.md, GUION_SUSTENTACION.md, ANEXO_IA.md, PENDIENTE_FASE4.md

Notas:
- Entorno: usar `venv/bin/python` (python3 del sistema tiene pandas roto por ABI numpy).
- R 4.3.3: se instalaron fda 6.3.0 y fda.usc 2.2.0 en ~/R/x86_64-pc-linux-gnu-library/4.3 (binarios Posit PPM).
- ONI descargado de NOAA-CPC a entrega/datos_externos/oni.ascii.txt (aux/oni.txt era Niño 3.4 de PSL, no ONI).
- Existe `datos_nuevos/Proyecto_Final_Analitica_FeErratas.docx` (procedencia no verificada) que
  cambia 1981-2013 → 1981-2003 y la partición a 1981-1995/1996-2000/2001-2003.

## Reanudación 20/21-nov-2026 (mark02) — correcciones de las asesorías 19 y 20-nov
- Respaldo de lo del 19-nov (copia): respaldo_19nov/ (script, tablas, figuras, cache npz, logs).
- [x] Altitud SRTM 90 m: datos_externos/srtm90m_estaciones.csv (OpenTopoData srtm90m, 238 estaciones CVC).
- [x] Fase 0 v2 (P1 #1-9) → Tab02, Tab02b, Tab02c (prueba cero≠faltante), Tab05 (rachas 30 d + 2 vecinas),
  Tab06 (% faltantes estación×año), Tab06b-i, Fig01, Fig02, Fig02b-d. Log: cache/log_fase0_v2.txt.
  Regla elegida: periodo = nacional (>45 % datos) + coordenadas → 141 est. (22 Pacífico); año = ≤25 % faltantes → 2945 pares.
  OMM daría 77 est. (6 Pacífico). Imputación: CSW elegido (menor RMSE 30 d); IDW y kriging comparados (Tab06g).
  Satélite: razón sat/suma-estación = 1,04 → péntada = TOTAL mm/péntada (no promedio). Tope 200 mm, sin -9999.
- [x] Fases 1, 2, 3 re-corridas con la matriz imputada (log cache/log_fase123_v2.txt): 2945 curvas, K_est=6, K_sat=11, 141 est./123 píxeles.
- [x] Fase 2: FPCA por régimen ENSO (Tab23b, Fig14b). Fase 3b nueva: ANOVA funcional 4 subregiones (22/48/56/15 est.)
  + Welch funcional por pares con permutación (Tab33, Tab34, Fig21). Log cache/log_fase2_3b_v2.txt.
- [x] Fase 4 (proyecto_final.py fase4; log cache/log_fase4_v2.txt, ~6 min): CV temporal 4 folds en 1981-1996, prueba 1997-2003 en
  3 tramos, FPCA por régimen (M=5) vs global, 5 conjuntos de covariables × 3 cabezas × 3 semillas, bootstrap 2000 + Wilcoxon,
  poda KAN, activaciones MLP + SAE → Tab35-Tab46, Fig22-Fig28, predicciones_estaciones_reservadas.csv.
  Corrección: el objetivo en mm era (curva de sqrt)^2 → sesgo de Jensen; ahora se modela sqrt de la curva suavizada.
- [x] R: fase0b nueva (reproduce la regla elegida desde Tab06 y carga la matriz imputada); fases 1-3 en R re-corridas
  (cache/log_R_fase123_v2.txt). INVENTARIO_DATOS.md, ANEXO_IA.md, REFERENCIAS_VERIFICADAS.md escritos.
- [x] INFORME_FINAL_v2.md → "Proyecto final AnalíticaDeDatos.pdf" (15 págs; pandoc→HTML→chrome). GUION_SUSTENTACION.md,
  ANEXO_IA.docx, "Proyecto final AnalíticaDeDatos - scripts v2.zip".
- [x] Corrida completa Python de punta a punta (cache/log_completo_v2.txt, 8,5 min): tablas idénticas bit a bit a las del informe.
  Corrida completa R (cache/log_R_completo_v2.txt, 3 min): P0-P3c y regla elegida idénticos a Python.
- PENDIENTE César: nombres completos de los integrantes en el informe (portada) y reparto real del guion; decidir si se
  avisa al profesor de: regla nacional sin fuente verificada, satélite = total (no promedio), partición nueva.
