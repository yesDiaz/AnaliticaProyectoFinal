# Referencias verificadas (consulta: 2026-11-20/21)

Leyenda: **[VERIFICADO]** = abrí el texto completo (PDF/HTML) y el dato aparece ahí. **[VERIFICADO-resumen]** = solo vi resumen/página oficial del editor. **[METADATOS]** = solo confirmé la cita (Crossref/DOI), no el contenido.

## 1. Modelación de lluvia en Colombia (datos / preprocesamiento)

1. Duarte, Y. E., Suárez Barón, M. J., García Cabrejo, O. J., Jaramillo Acevedo, C. A., & Meneses Escobar, C. A. (2025). Deep learning and time series for the prediction of monthly precipitation. A case study in the Department of Boyacá, Colombia. *Ingeniería, 30*(1), e21930. https://doi.org/10.14483/23448393.21930 — [VERIFICADO] (HTML del artículo, https://revistas.udistrital.edu.co/index.php/reving/article/view/21930)
   - CHIRPS 2.0 (netCDF); datos diarios "from 1981 to May 2023", agregados a mensual "through the sum of daily data"; LSTM/ConvLSTM; 753 puntos/estaciones en Boyacá.
   - No reporta criterio de % de faltantes ni imputación; no eliminó atípicos. Ojo: el texto dice 0.05° en Métodos y 0.5° en Resumen (inconsistencia del artículo).
   - Es el artículo de Boyacá que mencionó el profesor, pero es de la revista *Ingeniería* (U. Distrital), **no** de la UTP.

2. Russi Peña, J., & Larrahondo, E. (2017). *Comparación de métodos de estimación de datos faltantes en series de precipitación diaria en el Valle del Cauca* [Trabajo de grado, Estadística]. Universidad del Valle. https://bibliotecadigital.univalle.edu.co/server/api/core/bitstreams/1b64315c-e000-4b50-9601-74faf66c447b/content — [VERIFICADO]
   - Lluvia diaria (mm) 1993-2011, ~106 estaciones IDEAM/CVC, 97 con registros en todos los años; zonas valle (≤1239 m) y montaña.
   - Compara KNN, IDW, regresión lineal y kriging ordinario con escenarios de 10, 15 y 20 % de faltantes; ningún método domina (KO mejor en valle con 10 %, IDW en valle con 20 %, etc.).
   - Peor año: 2011 con 18 % de faltantes; mejor: 1997 con 4 %. No define un umbral de exclusión de estaciones.

3. Gómez Sepúlveda, J. I., Lafaurie Suárez, J. A., & Mendoza García, M. C. (2024). *Modelo de predicción de precipitación acumulada para un departamento de Colombia por medio de la implementación de redes neuronales recurrentes (LSTM) e integración de datos satelitales* [Trabajo de maestría]. Pontificia Universidad Javeriana Cali. https://vitela.javerianacali.edu.co/bitstreams/b047822a-acc5-46dc-a7c2-bc01d75b61ba/download — [VERIFICADO]
   - Estaciones IDEAM + CHIRPS diario a 0.05° (netCDF), Valle del Cauca, agregación semanal, LSTM, 80/20 temporal.
   - Valle del Cauca: 79 estaciones, 57 812 datos faltantes. Se excluyen las 20 estaciones con más faltantes y se elige "una estación por municipio" con la mayor completitud (no hay un umbral en %). Los NaN del rezago se reemplazan por 0.

4. Canchala-Nastar, T., Carvajal-Escobar, Y., Alfonso-Morales, W., Loaiza Cerón, W., & Caicedo, E. (2019). Estimation of missing data of monthly rainfall in southwestern Colombia using artificial neural networks. *Data in Brief, 26*, 104517. https://doi.org/10.1016/j.dib.2019.104517 — [VERIFICADO] (texto completo vía Europe PMC, PMC6811873)
   - 45 estaciones IDEAM del suroccidente, lluvia mensual 1983-2016 (34 años); imputación con NLPCA (red [45-44-45]); error ≈ 9.8 mm/mes. No da un umbral de % de faltantes para seleccionar estaciones.

5. Valencia-González, J. N., Arteaga-Ramírez, R., Vázquez-Peña, M. A., & Quevedo-Nolasco, A. (2022). Relleno de datos diarios faltantes en registros de series climatológicas temporales. *Revista Mexicana de Ciencias Agrícolas, 13*(4), 617-629. https://doi.org/10.29312/remexca.v13i4.2514 — [VERIFICADO] (HTML de SciELO México)
   - 9 estaciones IDEAM/CVC de la zona arrocera del sur del Valle del Cauca; lluvia diaria 1969-2018 con 16 % de faltantes; criterio: "más de 20 años de información".
   - Compara razón normal, serie de Fourier e IDW (cuadrado del inverso de la distancia). Para precipitación, el mejor fue IDW (el resumen y las conclusiones no coinciden del todo en las demás variables).

6. Cruz-Roa, A. F., & Barrios, M. I. (2018). Estimación de datos faltantes de lluvia mensual a través de la asimilación de información satelital y pluviométrica en una cuenca andina tropical. *Idesia, 36*(3), 107-117. https://doi.org/10.4067/S0718-34292018005001601 — [VERIFICADO] (PDF de Dialnet)
   - 16 estaciones IDEAM, cuenca del río Coello (Tolima), lluvia mensual 1998-2005. Los escenarios usan estaciones con "menos de un 20% de datos faltantes"; faltantes simulados del 10 al 40 %. El mejor método fue cokriging con elevación (RMSE 15.54).

## 2. Umbrales de selección de estaciones en Colombia (búsqueda del "45 %")

7. Martínez Pedraza, A., & Serna Cuenca, J. (2018). *Validación de las estimaciones de precipitación con CHIRPS e IRE/IDEAM* (Nota técnica IDEAM-METEO/002-2018). IDEAM. https://bart.ideam.gov.co/meteorologia/NOTAS_TECNICAS/NT_IDEAM-2018-002_CHIRPS_IRE.pdf — [VERIFICADO]
   - Criterio 1: "mínimo de 25 datos en el periodo 1981 – 2010 (15% de datos faltantes)" por mes calendario. Criterio 2: homogeneidad SNHT. Resultado: de 1404 series quedan 1052 estaciones.
8. Ruiz Murcia, J. F., Melo Franco, J. Y., Herrera Aponte, S. M., Fonseca Gutiérrez, E. K., & Vega Burgos, A. (2023). *Cálculo de indicadores (evidencias) de cambio climático para Colombia 1981-2020 con el uso de inqc, climatol y climpact* (Nota técnica IDEAM-METEO/004-2023). IDEAM. https://bart.ideam.gov.co/meteorologia/NOTAS_TECNICAS/NT_IDEAM-2023-004_INDICADORES_CLIMPACT_CLIMATOL.pdf — [VERIFICADO]
   - 228 estaciones diarias; desde ~1995 hay en promedio "alrededor de un 63% de datos diarios disponibles de precipitación". Relleno y homogeneización con `climatol` (promedio ponderado de vecinas). No da un umbral explícito en %.
9. IDEAM. (2020). *Instructivo para la elaboración del mapa del índice de la precipitación* (M-GDI-M-I002, v01). https://www.ideam.gov.co/sites/default/files/mapa-de-procesos/m-gdi-m-i002_instructivo_para_la_elaboracion_del_mapa_del_indice_de_la_precipitacion_v1.pdf — [VERIFICADO]
   - Descarta meses con menos del 80 % de los días ("24 si el mes tiene 30 días o 25 si el mes es de 31"), citando a la OMM.

## 3. OMM / WMO

10. World Meteorological Organization. (2017). *WMO guidelines on the calculation of climate normals* (WMO-No. 1203). ISBN 978-92-63-11203-3. Copia: https://dgm-meteo.github.io/wmo-clino/Docs/WMO2017_1203_en_WMO%20Guidelines%20on%20the%20Calculation%20of%20Climate%20Normals.pdf (ficha oficial: https://library.wmo.int/records/item/55797) — [VERIFICADO]
    - §4.4.1 (variables de media): no calcular el valor mensual si faltan "11 or more days" o hay "5 or more consecutive days" faltantes. Los países que usan la regla 5/3 (WMO/TD-341, 1989) pueden mantenerla.
    - Variables de suma (precipitación total): solo con el mes completo. Excepción: datos estimados o acumulados, si no cubren 11 o más días ni 5 o más consecutivos.
    - §4.4.2: normal mensual solo con valores válidos en "at least 80% of the years" (24 de 30), sin criterio adicional de años consecutivos.
11. World Meteorological Organization. (2018). *Guide to climatological practices* (WMO-No. 100, 2018 ed.). ISBN 978-92-63-10100-6. Copia: https://biotech.law.lsu.edu/blog/100-2018_en.pdf — [VERIFICADO]
    - §4.8.4: no calcular el valor mensual si faltan "more than ten daily values or five or more consecutive daily values". Para lluvia (suma), solo con todos los días o con acumulados. WMO/TD-341 es más estricta: "more than five missing days in total, or more than three consecutive". Normales con ≥80 % de los años y "no more than three consecutive missing years".
12. Trewin, B. (2007). *The role of climatological normals in a changing climate* (WCDMP-No. 61, WMO/TD-No. 1377). WMO. https://www.uncclearn.org/wp-content/uploads/library/wmo46.pdf — [VERIFICADO] Resume la regla 10/5 (Guía 1983) y la regla 5/3 (WMO 1989), y la exigencia de 25 de 30 años (WMO 1989).

## 4. NOAA

13. Arguez, A., Durre, I., Applequist, S., Vose, R. S., Squires, M. F., Yin, X., Heim, R. R., & Owen, T. W. (2012). NOAA's 1981–2010 U.S. climate normals: An overview. *BAMS, 93*(11), 1687-1697. https://doi.org/10.1175/BAMS-D-11-00197.1 — [VERIFICADO-resumen] (resumen vía OpenAlex; el PDF de AMS dio 403). Calcula normales para ~9800 estaciones NWS y "quasi normals" para ~2000.
14. NOAA NCEI. (2021). *Normals calculation methodology 2020*. https://www.ncei.noaa.gov/data/normals-daily/1991-2020/doc/Normals_Calculation_Methodology_2020.pdf — [VERIFICADO]
    - "According to the WMO guidelines, a minimum of 10 years is required for calculating any normal". Banderas: S ≥24 años (80 %), R/P ≥10 años, E ≥2 años.
    - En su Apéndice (Durre et al., 2011, precipitación 1981-2010): cada mes debe estar "complete in at least 10 of the 30 years". **Sí: el criterio es ≥10 años.**

## 5. CHIRPS

15. Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data, 2*, 150066. https://doi.org/10.1038/sdata.2015.66 — [VERIFICADO] (PMC4672685): "quasi-global (50°S-50°N), high resolution (0.05°), daily, pentadal, and monthly"; 1981-presente.
16. Climate Hazards Center, UCSB. *README-CHIRPS.txt* y *README-CHIRPS2-Daily.txt*. https://data.chc.ucsb.edu/products/CHIRPS/v2.0/ (la ruta antigua `/products/CHIRPS-2.0/` devuelve 404) — [VERIFICADO]
    - "CHIRPS is fundamentally a pentad and monthly product". El diario se obtiene desagregando "the CHIRPS2 pentadal rainfall total" con CFSv2. Se recomienda trabajar con totales pentadales o mensuales.
    - Cabecera del netCDF `global_pentad/netcdf/chirps-v2.0.1981.pentads.nc` (leída por mí): `units = "mm/pentad"`, `missing_value = _FillValue = -9999`; "time variable denotes the first day of the given pentad".
    - Aviso: CHIRPS v2 deja de producirse después de diciembre de 2026 (existe CHIRPS v3).

## 6. SRTM

17. Farr, T. G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S., ... Alsdorf, D. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics, 45*(2), RG2004. https://doi.org/10.1029/2005RG000183 — [VERIFICADO] (preprint JPL de 2007: https://webarchive.library.unt.edu/web/20130215012607id_/http://www2.jpl.nasa.gov/srtm/SRTM_paper.pdf)
    - Meta de diseño: error vertical absoluto < 16 m (90 %). Resultado: "better than 9 m". Tabla 1, Sudamérica: error absoluto de altura 6.2 m y relativo 5.5 m (90 %).
18. Jarvis, A., Reuter, H. I., Nelson, A., & Guevara, E. (2008). *Hole-filled SRTM for the globe, Version 4*. CGIAR-CSI. https://srtm.csi.cgiar.org — [VERIFICADO parcial] La página confirma 3″ (~90 m en el ecuador), relleno de vacíos y "vertical error ... less than 16m". El título exacto de la cita y el año 2008 NO aparecen en la página (forma de cita usual, no verificada).

## 7. Interpolación e imputación

19. Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. En *Proc. 23rd ACM National Conference* (pp. 517-524). https://doi.org/10.1145/800186.810616 — [VERIFICADO-resumen] Origen del IDW; el resumen menciona como ejemplo las estaciones de lluvia.
20. Goovaerts, P. (2000). Geostatistical approaches for incorporating elevation into the spatial interpolation of rainfall. *Journal of Hydrology, 228*(1-2), 113-129. https://doi.org/10.1016/S0022-1694(00)00144-X — [VERIFICADO] (PDF: https://people.geog.ucsb.edu/~kclarke/Geography232/Goovaerts_2000.pdf)
    - 36 estaciones en Portugal. Los métodos multivariados con DEM superan a Thiessen, al inverso del cuadrado de la distancia, a OK y a la regresión. OK supera a la regresión cuando la correlación lluvia-elevación es menor de 0.75.
21. Teegavarapu, R. S. V., & Chandramouli, V. (2005). Improved weighting methods, deterministic and stochastic data-driven models for estimation of missing precipitation records. *Journal of Hydrology, 312*(1-4), 191-206. https://doi.org/10.1016/j.jhydrol.2005.02.015 — [VERIFICADO-resumen] (resumen en copia de terceros; ScienceDirect dio 403)
    - Propone mejoras del IDW con "surrogate measures for distances", 20 pluviómetros de Kentucky. No pude confirmar en la fuente primaria la fórmula del CCWM (peso = coeficiente de correlación).

## 8. Métodos estadísticos y de ML

22. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold networks* (arXiv:2404.19756v5). https://doi.org/10.48550/arXiv.2404.19756 — [VERIFICADO] (HTML v5)
    - Activaciones entrenables (splines) en las aristas.
    - §2.5.1: pérdida = predicción + λ(μ1·L1 + μ2·entropía), con μ1 = μ2 = 1; "L1 alone is insufficient", por eso se añade la entropía.
    - Poda: se conserva un nodo si sus puntajes de entrada y de salida superan θ = 10⁻².
23. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., ... Olah, C. (2023). *Towards monosemanticity: Decomposing language models with dictionary learning*. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features/index.html — [VERIFICADO] "Using a sparse autoencoder, we extract a large number of interpretable features from a one-layer transformer."
24. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse autoencoders find highly interpretable features in language models* (arXiv:2309.08600). https://arxiv.org/abs/2309.08600 — [VERIFICADO-resumen] Los SAE aprenden características "more interpretable and monosemantic".
25. Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593 — [METADATOS] (Crossref registra el DOI de la reedición digital con fecha 1994; la edición original es de 1993).
26. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192-213. https://doi.org/10.1016/j.ins.2011.12.028 — [VERIFICADO-resumen] (página de Monash): proponen que la validación cruzada por bloques ("blocked form of cross-validation") sea el procedimiento estándar.
27. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/tscv.html — [VERIFICADO] §5.10: "evaluation on a rolling forecasting origin"; el entrenamiento usa "only observations that occurred prior to" el conjunto de prueba. (La web no muestra un año; 2021 es la fecha usual de la 3.ª edición.)
28. Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika, 34*(1-2), 28-35. https://doi.org/10.1093/biomet/34.1-2.28 — [METADATOS]
29. Cuevas, A., Febrero, M., & Fraiman, R. (2004). An anova test for functional data. *Computational Statistics & Data Analysis, 47*(1), 111-122. https://doi.org/10.1016/j.csda.2003.10.021 — [METADATOS] Además, la documentación de `fda.usc::fanova.onefactor` (https://rdrr.io/cran/fda.usc/man/fanova.onefactor.html) la cita como "an asymptotic version of the anova F–test" con bootstrap (`nboot`).
30. NOAA Climate Prediction Center. *Oceanic Niño Index (ONI)*. https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/ (también /v6/) — [VERIFICADO]
    - "3 month running mean of ERSST.v5 SST anomalies in the Niño 3.4 region (5oN-5oS, 120o-170oW)", umbral ±0.5 °C, "minimum of 5 consecutive overlapping seasons", bases de 30 años centradas que se actualizan cada 5 años.
    - Nota de 2026: la tabla principal ya usa ERSST.v6, y la NWS usa el RONI (Relative ONI) para el monitoreo oficial.

## No encontrado / dudoso

- **Criterio ">45 % de datos / <55 % de faltantes" con 163 estaciones:** NO encontrado tras varias búsquedas (español e inglés, IDEAM, CVC, "163 estaciones"). No lo citen sin la fuente original; pidan al estudiante la referencia. Umbrales colombianos verificados: 15 % de faltantes por mes calendario (IDEAM 2018), mes válido con ≥80 % de días (IDEAM 2020), estaciones con <20 % de faltantes (Cruz-Roa & Barrios 2018), >20 años de registro (Valencia-González et al. 2022). La nota IDEAM 2023 informa ~63 % de disponibilidad media, sin umbral.
- **Artículo de Boyacá "de la UTP":** el que coincide (CHIRPS diario 1981-2023 sumado a mensual) es Duarte et al. (2025), revista *Ingeniería* de la U. Distrital. Posible confusión UTP/UPTC; la financiación por UPTC no la verifiqué en el texto.
- **WMO-No. 1203:** el texto vino de una copia en GitHub (DGM Marruecos), porque la e-Library de la OMM no se pudo descargar. El ISBN y la edición coinciden con la ficha oficial.
- **Arguez et al. 2012, Welch 1947, Cuevas et al. 2004, Efron & Tibshirani 1993:** no pude abrir el texto completo (paywall/403). Solo metadatos o resumen.
- **Teegavarapu & Chandramouli 2005 (CCWM):** la definición "peso = coeficiente de correlación" solo aparece en fuentes secundarias. Verificar en el PDF institucional.
- **Jarvis et al. 2008:** la forma exacta de la cita no aparece en srtm.csi.cgiar.org.
