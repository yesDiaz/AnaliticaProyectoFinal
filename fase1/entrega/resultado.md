# Resultado — Fases 4.1 y 1 (14-nov-2026)

## Qué quedó hecho

- `preproc_4_1.py` — depuración completa con tabla de trazabilidad por paso.
- `fase1_datos_funcionales.py` — datos funcionales construidos a mano (Cox-de Boor, Fourier y
  Cholesky propia), comparación de dos bases, selección de K y λ con ECM/VC/VCG, verificación de S,
  bandas punto a punto, control de bimodalidad y las dos fuentes en el mismo dominio anual.
- `entrega/informe_fase1.md` — redacción lista para pegar al informe PDF.

## Números principales

| Concepto | Valor |
|---|---|
| Estaciones finales | 186 (de 323 en 1981-2013) |
| Curvas estación-año | 3.567 |
| Periodo efectivo | 1981-2003 |
| Curvas satelitales | 46.176 (1.443 píxeles × 32 años completos) |
| **Base elegida (ambas fuentes)** | **Fourier K = 5** (constante + 2 armónicos) |
| VCG estaciones | 3,0401 Fourier K=5 vs 3,0418 B-spline K=6 |
| VCG satélite | 0,2749 Fourier K=5 vs 0,2804 B-spline K=8 |
| λ (VCG, mínimo interior) | 10⁵ → ν = tr(S_λ) = 4,736 |
| S idempotente | max\|S·S − S\| = 3,47e−17 |
| tr(S) = K | 5,000000 = 5 (diferencia exactamente 0) |
| σ̂² residual | 2,0951 (escala √mm) |
| Empalme 31-dic/1-ene | **0,0000** con Fourier vs 0,4136 con B-spline |
| Picos del perfil medio | días 112 (22-abr) y 304 (31-oct) |

Depuración: 50 valores > 300 mm anulados (el máximo era 859.375 mm), 80 estación-mes con el total
mensual en un solo día (27 detectados solo por la regla de cierre de mes), 321 rachas de ≥ 45 días
en cero, 1.033 pares estación-año descartados por cobertura, solo 52 valores interpolados.

## El hallazgo que cambió la Fase 1

La primera versión usaba solo B-splines y el resultado parecía correcto (VCG con mínimo interior,
S idempotente, tr(S) = K). Al revisar la figura del perfil climatológico se vio que **la curva se
desploma al cerrar el año**: 0,61 el 31 de diciembre frente a 1,03 el 1 de enero. La curva anual de
lluvia es periódica y la base B-spline no impone esa condición, así que el último tramo polinómico
se desvía sin datos a la derecha que lo sujeten.

Se añadió una base de **Fourier** (periódica por construcción) y se dejó que el VCG decidiera. Gana
en las dos fuentes, por tres razones acumuladas:

1. Mejor VCG (3,0401 vs 3,0418 en estaciones; 0,2749 vs 0,2804 en satélite).
2. Empalme de fin de año exacto (0,0000 vs 0,4136).
3. Interpretación física: los dos armónicos **son** los dos pasos anuales de la ZCIT sobre el valle,
   así que cada coeficiente tiene lectura climatológica. El B-spline además desplazaba el segundo
   pico dos semanas (día 317 en vez de 304) y aplanaba el primero.

**Consecuencia para la Fase 3:** las dos fuentes, seleccionadas por separado, convergen en la misma
base y la misma dimensión. Los coeficientes de estación y de píxel son comparables componente a
componente, sin reproyección: la comparación puede hacerse sobre 5 números en vez de curvas día a día.

## Reproducir

```bash
cd /srv/mark02/analitica-final
/usr/bin/python3 preproc_4_1.py              # ~1 min  -> entrega/estaciones_depuradas.csv
/usr/bin/python3 fase1_datos_funcionales.py  # ~7 min  -> coef_estaciones.csv, coef_satelite.csv
```
Requiere numpy, pandas, matplotlib, Pillow (los GeoTIFF se leen con Pillow: no hace falta rasterio).

## Decisiones que César debe confirmar con el docente o el grupo

1. **La partición 1981-2005 / 2006-2009 / 2010-2013 no es aplicable**: las estaciones terminan en
   2003 y el satélite en abril de 2013. Hay que conseguir los datos 2004-2013 o redefinir la
   partición dentro de 1981-2003 (propuesta: 1981-1995 / 1996-1999 / 2000-2003).
2. **Faltan las coordenadas de las estaciones** (lat, lon, altitud). Sin el catálogo CNE del IDEAM
   no hay emparejamiento estación↔píxel (Fase 3) ni altitud como covariable (Fase 4).
3. 22 de las 186 estaciones son de las zonas hidrográficas 53 y 54, no de la 26 (Cauca): hay que
   verificar si caen dentro de la malla satelital.
4. Confirmar la transformación: se trabajó con **√P**. Si el grupo prefiere log(1+P), es cambiar
   una línea, pero debe ser la misma en todas las fases.

## Lo que falta del proyecto

- **Versión en R** de la Fase 1 (si el grupo entrega en R; el `ScriptInicial.R` del compañero es R).
- Fase 2: EDA funcional (media, desviación, FPCA, variabilidad interanual).
- Fase 3: comparación estación vs. píxel (sesgo, correlación funcional) — bloqueada por las coordenadas.
- Fase 4.2-4.5: modelo funcional, comparación con MLP/KAN/lineal, métricas.
- Informe PDF ≤ 20 páginas y anexo obligatorio con los prompts de IA usados.
