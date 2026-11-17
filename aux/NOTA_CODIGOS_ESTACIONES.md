# Nota: los STATION_ID del dataset NO son codigos del CNE del IDEAM

Fecha: 2026-11-15. Autor: verificacion propia (Fase 2/3).

## Que se intento
Conseguir latitud/longitud/altitud de las 186 estaciones para poder hacer la Fase 3
(covarianza espacial, variograma funcional, kriging) y usar altitud como covariable.

Fuente probada: **Catalogo Nacional de Estaciones del IDEAM (CNE)**, datos.gov.co
dataset `hp9r-jxuu` (29.320 registros, con Altitud/LONGITUD/LATITUD).
Copia local: `aux/cne.csv`.

## Resultado: no empareja
- Coincidencia literal de los 186 `STATION_ID` (10 digitos) contra `Codigo` del CNE
  (con y sin ceros a la izquierda): **0 de 186**.
- Se probaron 6 reglas de reescritura (tomar subzona + consecutivo, mover el par final,
  rellenar con ceros, etc.). La mejor fue `id[:4] + id[6:9] + "0"` con 134/186
  "coincidencias", pero es **espuria**: las estaciones resultantes caen en Antioquia (49),
  Quindio (32), Risaralda (30), Caldas (11) y solo 12 en Valle del Cauca, con latitudes
  hasta 7.86 N. Solo 21 de 134 caen dentro de la malla satelital del proyecto
  (lon -77.52 a -75.72, lat 3.07 a 4.97). Incompatible con el area de estudio.
- Los prefijos de 4 digitos del dataset (2620: 14 estaciones, 2622: 32, 2624: 18,
  5300: 4, 5400: 1, 5420: 8, 5421: 2) **no existen** entre los codigos del CNE del
  Valle del Cauca (cuyos prefijos son 2609, 2610, 2607, 2608, 2611, 5407, 5311, ...).

## Conclusion
El dataset entregado usa un esquema de codificacion propio (muy probablemente la red
de la **CVC - Corporacion Autonoma Regional del Valle del Cauca**, no el CNE del IDEAM).
Sin el catalogo de ese esquema no hay coordenadas para las estaciones.

## Implicaciones
1. **Fase 3** (geoestadistica funcional) se hace con la fuente **satelital**, que si trae
   LON/LAT por pixel (1.443 pixeles, malla 0.05 grados). Es una base valida y mas densa.
2. Para las estaciones se mantiene el analisis no espacial (Fases 2 y 4) y, si aparece el
   catalogo, se añade el componente espacial.
3. **Pedir al profesor** el catalogo de estaciones del dataset (codigo, nombre, lat, lon,
   altitud) o la fuente original de descarga. Es el unico dato que falta para cerrar la
   Fase 3 con las dos fuentes.
