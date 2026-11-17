#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integra las fases nuevas (3b, 3c, 4b, 4.5) en el archivo unico entregable.

La guia exige un unico script reproducible. Los scripts por fase se envuelven
en funciones _bloque_N() respetando el orden logico: las coordenadas reales
deben calcularse ANTES de la fase 4, porque sobrescriben el dataset pareado.

Uso: ./venv/bin/python integra_script_unico.py
"""
import os
import re
import textwrap

RAIZ = os.path.dirname(os.path.abspath(__file__))
DESTINO = os.path.join(RAIZ, "proyecto_final_python.py")
FASES = [
    ("fase3b_coords_reales.py", "_bloque_5b",
     "Fase 3b - emparejamiento con coordenadas oficiales"),
    ("fase3c_control_espacial.py", "_bloque_5c",
     "Fase 3c - control espacial del emparejamiento"),
    ("fase3d_comparacion_real.py", "_bloque_5d",
     "Fase 3d - comparacion pareada georreferenciada"),
    ("fase4b_red_funcional.py", "_bloque_7",
     "Fase 4b - codificador funcional aprendible (4.2) y particion oficial"),
    ("fase45_interpretacion.py", "_bloque_8",
     "Fase 4.5 - interpretacion climatologica (4.5)"),
]
SALTAR = re.compile(r"^\s*(import |from |#!|# -\*-|matplotlib\.use)")


def cuerpo(ruta):
    """Devuelve el codigo del script sin cabecera, docstring ni imports."""
    txt = open(os.path.join(RAIZ, ruta)).read()
    lineas = txt.split("\n")
    # salta el docstring del modulo
    i = 0
    while i < len(lineas) and not lineas[i].startswith('"""'):
        i += 1
    if i < len(lineas):
        i += 1
        while i < len(lineas) and not lineas[i].rstrip().endswith('"""'):
            i += 1
        i += 1
    out = []
    for ln in lineas[i:]:
        if SALTAR.match(ln):
            continue
        # SystemExit dentro de una funcion rompe la ejecucion de los demas bloques
        ln = ln.replace("raise SystemExit(0 if ok else 1)",
                        "P('    verificacion de la integral: '"
                        " + ('OK' if ok else 'FALLA'))")
        ln = re.sub(r"^import fase4b_red_funcional as F4.*$", "", ln)
        out.append(ln)
    return "\n".join(out).strip("\n")


def envuelve(nombre, desc, codigo):
    cab = (f'def {nombre}():\n'
           f'    """{desc}."""\n'
           f'    global log\n'
           f'    log = []\n'
           f'    import numpy as np\n'
           f'    import pandas as pd\n'
           f'    import matplotlib\n'
           f'    matplotlib.use("Agg")\n'
           f'    import matplotlib.pyplot as plt\n'
           f'    import torch\n'
           f'    import torch.nn as nn\n')
    return cab + textwrap.indent(codigo, "    ") + "\n"


src = open(DESTINO).read()
if "_bloque_5b" in src:
    raise SystemExit("el archivo unico ya contiene las fases nuevas; nada que hacer")

nuevas = []
for ruta, nombre, desc in FASES:
    nuevas.append(envuelve(nombre, desc, cuerpo(ruta)))
bloque_nuevo = "\n\n".join(nuevas)

marca = "_BLOQUES = ["
i = src.index(marca)
src = src[:i] + bloque_nuevo + "\n\n" + src[i:]
src = src.replace(
    '    (_bloque_5, "Fase 3 - comparacion estaciones vs satelital"),\n',
    '    (_bloque_5, "Fase 3 - comparacion estaciones vs satelital"),\n'
    '    (_bloque_5b, "Fase 3b - emparejamiento con coordenadas oficiales"),\n'
    '    (_bloque_5c, "Fase 3c - control espacial del emparejamiento"),\n'
    '    (_bloque_5d, "Fase 3d - comparacion pareada georreferenciada"),\n')
src = src.replace(
    '    (_bloque_6, "Fase 4 - red funcional, ablacion y evaluacion"),\n',
    '    (_bloque_6, "Fase 4 - red funcional, ablacion y evaluacion"),\n'
    '    (_bloque_7, "Fase 4b - codificador funcional aprendible (4.2)"),\n'
    '    (_bloque_8, "Fase 4.5 - interpretacion climatologica (4.5)"),\n')
open(DESTINO, "w").write(src)
n = len(open(DESTINO).read().split("\n"))
print(f"archivo unico actualizado: {n} lineas, {len(FASES)} bloques nuevos")
print("verificacion de sintaxis:")
import py_compile
py_compile.compile(DESTINO, doraise=True)
print("    OK")
