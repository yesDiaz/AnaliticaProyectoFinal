# Precipitación en el Valle del Cauca: estaciones en tierra frente a satélite como datos funcionales

Proyecto final · **Analítica de Datos** · Universidad Autónoma de Occidente · 2026-2S

Se modela la lluvia diaria de **141 estaciones** de la red CVC (1981-2003) y el producto
satelital pentadal de 0,05° como **curvas anuales** sobre una base B-spline construida a
mano, y se corrige el sesgo del satélite con **redes funcionales** (cabeza lineal, MLP y KAN).

**Entregable principal:** [`entrega/Proyecto final AnalíticaDeDatos.pdf`](entrega/Proyecto%20final%20Anal%C3%ADticaDeDatos.pdf) (15 páginas)

---
| Nombre completo | Código | Correo institucional |
|---|---|---|
| César Armando Reyes Oliveros | 2236379 | cesar_armando.reyes@uao.edu.co |
| Yesenia Díaz | 2231783 |yesenia.diaz@uao.edu.co |
| Juan Pablo Maya | 2236377 |juan_pablo.maya@uao.edu.co |
## Resultados principales

| Hallazgo | Cifra |
|---|---|
| Sesgo mediano del total anual del satélite | **+10,4 %** (sobreestima abr-jun, subestima ene-mar) |
| Error L² relativo en prueba, sin corregir → MLP | **0,461 → 0,431** (IC 95 % [−0,057; −0,000]; Wilcoxon p = 0,30) |
| Sesgo del total anual tras corrección (cabeza lineal) | **+17,3 % → +4,6 %** |
| Mejor cabeza por régimen | KAN gana en el tramo **Niño**, pierde en **Niña** |
| Resolución del satélite | 39 × 37 píxeles de 0,05°, valor pentadal = **total mm/péntada** (verificado, razón 1,04) |

Ninguna cifra del informe se escribió a mano: todas salen de las tablas que generan los
scripts, con log guardado.

## Estructura

```
entrega/            Entrega final
  Proyecto final AnalíticaDeDatos.pdf   informe (15 págs)
  INFORME_FINAL_v2.md                   fuente del informe
  proyecto_final.py                     fases 0, 1, 2, 3, 3b, 4   (~9 min)
  proyecto_final.R                      fases 0, 0b, 1, 2, 3      (~3 min)
  tablas/ tablas_R/ figuras/            salidas citadas en el informe
  datos_externos/                       ONI (NOAA-CPC) y altitud SRTM 90 m
  INVENTARIO_DATOS.md                   inventario de faltantes y reglas de selección
  REFERENCIAS_VERIFICADAS.md            bibliografía verificada
  ANEXO_IA.md                           declaración de uso de herramientas de IA
Datos/              Datos del curso (estaciones 1930-2003)
ImagenesSatelitePentadal/   GeoTIFF pentadales
Resultados/         Salidas del script inicial del grupo
historico/          Avances previos: fase 1 y ScriptInicial.R
Proyecto_Final_Analitica.docx   Enunciado del docente
```

## Cómo reproducir

Desde `entrega/`:

```bash
python proyecto_final.py      # fases 0 a 4, semilla 2026, ~9 min
Rscript proyecto_final.R      # fases 0 a 3 en R, ~3 min
```

Espera los datos del curso en `datos/` (o ajusta la ruta al directorio `Datos/` de este
repo). Salidas: `tablas/`, `tablas_R/`, `figuras/`, `cache/`.

**Versiones:** Python 3.12 · numpy 2.5.4 · pandas 3.1.1 · scipy 1.18.1 · matplotlib 3.11.2 ·
torch 2.15 (CPU) · scikit-fda 0.10.1 (solo verificación) · R 4.3.3 · data.table · readxl ·
fda 6.3.0 · fda.usc 2.2.0 (solo verificación).

La base B-spline y los coeficientes se calculan **desde cero**, como exige el enunciado;
`fda` y `scikit-fda` se usan únicamente para contrastar el resultado (coinciden a 1e-13).

## Metodología, en corto

1. **Fase 0 — preprocesamiento.** Inventario de faltantes por estación y año, comparación de
   las reglas OMM / NOAA / nacional, prueba de que 0 mm nunca se trata como faltante, rachas
   de 30 días en cero contrastadas con dos vecinas a ≤25 km, altitud del SRTM 90 m, e
   imputación comparando IDW, ponderación por correlación y kriging ordinario.
2. **Fase 1 — datos funcionales.** Base B-spline propia; `K_est = 6`, `K_sat = 11` por
   validación cruzada generalizada.
3. **Fase 2 — EDA funcional.** Media y covarianza, derivadas, profundidad FM y MBD, boxplot
   funcional, FPCA global y por régimen ENSO.
4. **Fase 3 — comparación por fuente.** Emparejamiento estación-píxel documentado, ANOVA
   funcional de 4 subregiones y Welch funcional por pares con permutación.
5. **Fase 4 — red funcional.** Codificador B-spline con matriz de Gram, decodificador FPCA
   congelado, ablación de 5 conjuntos de covariables × 3 cabezas × 3 semillas, bootstrap de
   2000 réplicas y Wilcoxon, partición 1981-1996 (CV temporal) / 1997-2003 (prueba por tramo
   Niño, Niña y neutro).

## Integrantes

- César Armando Reyes Oliveros ([@caesar-dat-com](https://github.com/caesar-dat-com))
- [@yesDiaz](https://github.com/yesDiaz)
- [@Juan-maya935](https://github.com/Juan-maya935)
