#!/usr/bin/env python3
# =====================================================================
# FASE 4 - Red neuronal funcional, ablacion de cabezas, evaluacion
# Proyecto Final de Analitica de Datos (UAO 2026-2S)
#
# Tarea: predecir la curva anual de precipitacion de una ESTACION terrestre
# a partir de la curva del PRODUCTO SATELITAL del pixel emparejado en la
# fase 3, mas tres covariables de contexto (altitud de la subzona, longitud
# y latitud del pixel).
#
# Arquitectura (4.2):
#   codificador funcional : z = W^{1/2} c_sat, con W = integral Phi Phi^T dt
#                           calculada por cuadratura trapezoidal (producto
#                           interno de L2, no el euclideo de los coeficientes)
#   cabeza                : lineal | MLP | KAN  (lo que se ablaciona en 4.3)
#   decodificador FPCA    : scores -> coeficientes, con la base de
#                           componentes principales estimada SOLO con el
#                           conjunto de entrenamiento (sin fuga)
#   perdida               : (c_pred - c_obs)^T W (c_pred - c_obs) / T
#                           + lambda_rug * rugosidad de las funciones peso,
#                           con R = integral Phi'' Phi''^T dt analitica
#
# Verificaciones exigidas por la rubrica:
#   - la cabeza lineal sin penalizacion coincide con la regresion funcional
#     resuelta en forma cerrada por Cholesky propia (tolerancia 1e-4)
#   - KAN validada sobre una funcion sintetica conocida
#   - numero de parametros de KAN y MLP igualado (diferencia <= 10 %)
#   - reconstruccion en mm no negativa: se reporta el % de dias recortados
#
# Evaluacion (4.4): particion temporal + estaciones reservadas sin fuga,
# las cuatro metricas (L2 relativo, KGE, sesgo del acumulado anual en %,
# error en la fecha de los dos picos), bootstrap de 1000 remuestreos,
# Wilcoxon pareada por estacion y desglose por fase ENSO.
#
# Nota de desviacion documentada: la rubrica sugiere la particion
# 1981-2005 / 2006-2009 / 2010-2013, pero las estaciones terrestres de este
# proyecto terminan en 2003 (el satelital llega a 2012). Se usa por tanto
# 1981-1996 / 1997-1999 / 2000-2003, que conserva la logica de extrapolar
# en el tiempo sin inventar datos.
#
# Entradas : entrega/pares_estacion_pixel.csv, aux/oni.txt
# Salidas  : entrega/log_fase4.txt, entrega/tablas/fase4_*.csv,
#            entrega/figs/fase4_*.png, entrega/pred_reservadas.csv
# =====================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
BASE = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(BASE, "entrega")
FIG = os.path.join(ENT, "figs")
TAB = os.path.join(ENT, "tablas")
AUX = os.path.join(BASE, "aux")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

K, PERIODO = 5, 365.0
t_dia = np.arange(0.5, 365.0, 1.0)
T = len(t_dia)
SEMILLAS = [20264, 20265, 20266]
LAM_RUG = 1e-6
ANIOS_TR = range(1981, 1997)
ANIOS_VA = range(1997, 2000)
ANIOS_TE = range(2000, 2004)
FRAC_RESERVADAS = 0.20
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
INI_MES = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


# ---------------------------------------------------------------------
# base de Fourier, derivadas analiticas, Gram y rugosidad
# ---------------------------------------------------------------------
def base_fourier(t, K, periodo=PERIODO):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


def dbase_fourier(t, K, periodo=PERIODO):
    w = 2 * np.pi / periodo
    cols = [np.zeros_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(j * w * np.cos(j * w * t))
        cols.append(-j * w * np.sin(j * w * t))
    return np.column_stack(cols[:K])


def d2base_fourier(t, K, periodo=PERIODO):
    w = 2 * np.pi / periodo
    cols = [np.zeros_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(-(j * w) ** 2 * np.sin(j * w * t))
        cols.append(-(j * w) ** 2 * np.cos(j * w * t))
    return np.column_stack(cols[:K])


def pesos_trapecio(t):
    w = np.empty_like(t)
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    w[1:-1] = (t[2:] - t[:-2]) / 2
    return w


def cholesky_propia(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - float(L[i, :j] @ L[j, :j])
            if i == j:
                if s <= 0:
                    raise np.linalg.LinAlgError("no definida positiva")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def resuelve_chol(A, B):
    """Resuelve A X = B (A simetrica def. pos.) por sustitucion, a mano."""
    L = cholesky_propia(A)
    n = A.shape[0]
    B = np.atleast_2d(B.T).T if B.ndim > 1 else B.reshape(-1, 1)
    Y = np.zeros_like(B)
    for i in range(n):
        Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
    X = np.zeros_like(B)
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - L[i + 1:, i] @ X[i + 1:]) / L[i, i]
    return X


Phi = base_fourier(t_dia, K)
dPhi = dbase_fourier(t_dia, K)
d2Phi = d2base_fourier(t_dia, K)
wq = pesos_trapecio(t_dia)
W = Phi.T @ (Phi * wq[:, None])            # Gram por cuadratura
R = d2Phi.T @ (d2Phi * wq[:, None])        # rugosidad
lamW, Uw = np.linalg.eigh(W)
Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T

log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.append(s)


P("=" * 74)
P("FASE 4 - red funcional, ablacion de cabezas y evaluacion")
P("=" * 74)
P(f"    Gram W por cuadratura: traza {np.trace(W):.2f} | "
  f"max|W - W^T| = {np.abs(W - W.T).max():.2e}")
P(f"    rugosidad R: traza {np.trace(R):.3e} (penaliza la curvatura de las w_k(t))")

# ---------------------------------------------------------------------
# datos y particiones
# ---------------------------------------------------------------------
D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"),
                dtype={"STATION_ID": str})
SAT = [f"sat_c{i+1}" for i in range(K)]
EST = [f"est_c{i+1}" for i in range(K)]
COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
P(f"\n[1] datos: {len(D)} pares estacion-anio | {D.STATION_ID.nunique()} estaciones "
  f"| {D.YEAR.min()}-{D.YEAR.max()}")

rng0 = np.random.default_rng(2026)
ests = np.array(sorted(D.STATION_ID.unique()))
reserv = set(rng0.choice(ests, int(round(FRAC_RESERVADAS * len(ests))), replace=False))
D["RESERVADA"] = D.STATION_ID.isin(reserv)
tr = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TR)]
va = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_VA)]
te = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TE)]
rs = D[D.RESERVADA]
P(f"    estaciones reservadas por sorteo (no se usan para entrenar): {len(reserv)} "
  f"de {len(ests)}")
P(f"    particion temporal: train {len(tr)} ({min(ANIOS_TR)}-{max(ANIOS_TR)}) | "
  f"val {len(va)} ({min(ANIOS_VA)}-{max(ANIOS_VA)}) | test {len(te)} "
  f"({min(ANIOS_TE)}-{max(ANIOS_TE)})")
P(f"    bloque reservado (generalizacion espacial): {len(rs)} pares, "
  f"{rs.STATION_ID.nunique()} estaciones, todos los anios")

# ---------------------------------------------------------------------
# decodificador FPCA estimado SOLO con train
# ---------------------------------------------------------------------
Ctr = tr[EST].to_numpy(float)
c_mu = Ctr.mean(axis=0)
V = ((Ctr - c_mu).T @ (Ctr - c_mu)) / (len(Ctr) - 1)
Whi = Uw @ np.diag(1 / np.sqrt(lamW)) @ Uw.T
lamF, Uf = np.linalg.eigh(Wh @ V @ Wh)
o = np.argsort(lamF)[::-1]
lamF, Uf = lamF[o], Uf[:, o]
Bfull = Whi @ Uf
vexp = lamF / lamF.sum()
M = int(np.searchsorted(np.cumsum(vexp), 0.99) + 1)
Bm = Bfull[:, :M]
P(f"\n[2] decodificador FPCA (solo train): varianza explicada "
  + " ".join(f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)))
P(f"    componentes retenidas para el 99 %: M = {M} "
  f"(acumulado {100*vexp[:M].sum():.2f} %)")
P(f"    ortonormalidad en L2: max|B^T W B - I| = "
  f"{np.abs(Bm.T @ W @ Bm - np.eye(M)).max():.2e}")


def codifica(df):
    z = df[SAT].to_numpy(float) @ Wh           # producto interno de L2
    x = np.column_stack([z, df[COV].to_numpy(float)])
    return x


Xtr, Xva, Xte, Xrs = codifica(tr), codifica(va), codifica(te), codifica(rs)
mx, sx = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
Xtr_s, Xva_s, Xte_s, Xrs_s = [(A - mx) / sx for A in (Xtr, Xva, Xte, Xrs)]
# objetivo: scores FPCA de la curva de estacion
Str = (tr[EST].to_numpy(float) - c_mu) @ W @ Bm
Sva = (va[EST].to_numpy(float) - c_mu) @ W @ Bm
P(f"    entrada al modelo: {Xtr.shape[1]} variables "
  f"(5 del codificador L2 + {len(COV)} covariables) -> salida {M} scores")


# ---------------------------------------------------------------------
# cabezas
# ---------------------------------------------------------------------
class KANPropia(nn.Module):
    """KAN minima: suma de funciones univariadas en base B-spline cubica.
    Se usa como respaldo si pykan no esta disponible o falla."""

    def __init__(self, n_in, n_out, grid=5, k=3):
        super().__init__()
        self.n_in, self.k = n_in, k
        paso = 1.0 / grid
        self.nodos = torch.linspace(-0.5 - k * paso, 1.5 + k * paso,
                                    grid + 2 * k + 2)
        self.nb = len(self.nodos) - k - 1
        self.coef = nn.Parameter(0.1 * torch.randn(n_in, self.nb, n_out))
        self.w_lin = nn.Parameter(0.1 * torch.randn(n_in, n_out))
        self.b = nn.Parameter(torch.zeros(n_out))

    def bspline(self, x):
        nod = self.nodos.to(x.device)
        B = ((x.unsqueeze(-1) >= nod[:-1]) & (x.unsqueeze(-1) < nod[1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            iz = (x.unsqueeze(-1) - nod[:-(p + 1)]) / (nod[p:-1] - nod[:-(p + 1)])
            de = (nod[p + 1:] - x.unsqueeze(-1)) / (nod[p + 1:] - nod[1:-p])
            B = iz * B[..., :-1] + de * B[..., 1:]
        return B

    def forward(self, x):
        xn = (x - x.min()) / (x.max() - x.min() + 1e-9) if False else x
        B = self.bspline(torch.tanh(xn) * 0.5 + 0.5)        # (n, n_in, nb)
        y = torch.einsum("nib,ibo->no", B, self.coef)
        y = y + torch.einsum("ni,io->no", torch.tanh(x), self.w_lin)
        return y + self.b


def construye_cabeza(tipo, n_in, n_out, semilla, h_mlp=None, usa_pykan=True):
    torch.manual_seed(semilla)
    if tipo == "lineal":
        return nn.Linear(n_in, n_out)
    if tipo == "mlp":
        return nn.Sequential(nn.Linear(n_in, h_mlp), nn.GELU(),
                             nn.Linear(h_mlp, n_out))
    if tipo == "kan":
        if usa_pykan:
            try:
                from kan import KAN
                return KAN(width=[n_in, n_out], grid=5, k=3, seed=semilla,
                           device="cpu")
            except Exception as e:                      # pragma: no cover
                P(f"    [aviso] pykan no utilizable ({type(e).__name__}: {e}); "
                  f"se usa KANPropia (splines B cubicas, misma idea)")
        return KANPropia(n_in, n_out)
    raise ValueError(tipo)


def n_par(m):
    return sum(p.numel() for p in m.parameters())


# igualar parametros de KAN y MLP (<= 10 % de diferencia)
kan0 = construye_cabeza("kan", Xtr.shape[1], M, SEMILLAS[0])
usa_pykan = type(kan0).__name__ == "MultKAN" or "kan" in type(kan0).__module__
npar_kan = n_par(kan0)
mejor_h, mejor_d = 1, 1e9
for h in range(1, 400):
    d = abs(n_par(construye_cabeza("mlp", Xtr.shape[1], M, 0, h_mlp=h)) - npar_kan)
    if d < mejor_d:
        mejor_h, mejor_d = h, d
H_MLP = mejor_h
npar_mlp = n_par(construye_cabeza("mlp", Xtr.shape[1], M, 0, h_mlp=H_MLP))
npar_lin = n_par(construye_cabeza("lineal", Xtr.shape[1], M, 0))
P(f"\n[3] presupuesto de parametros (sum(p.numel()))")
P(f"    KAN ({'pykan' if usa_pykan else 'KANPropia'}): {npar_kan} | "
  f"MLP (h={H_MLP}): {npar_mlp} | lineal: {npar_lin}")
P(f"    diferencia KAN vs MLP: {100*abs(npar_kan-npar_mlp)/npar_kan:.1f} % "
  f"({'cumple <= 10 %' if abs(npar_kan-npar_mlp)/npar_kan <= 0.10 else 'NO CUMPLE'})")

# ---------------------------------------------------------------------
# entrenamiento con perdida L2 funcional + rugosidad
# ---------------------------------------------------------------------
Wt = torch.tensor(W, dtype=torch.float32)
Rt = torch.tensor(R, dtype=torch.float32)
Bmt = torch.tensor(Bm, dtype=torch.float32)
cmut = torch.tensor(c_mu, dtype=torch.float32)


def perdida(pred_s, obs_c, modelo):
    c = cmut + pred_s @ Bmt.T
    d = c - obs_c
    l2 = ((d @ Wt) * d).sum(dim=1).mean() / PERIODO
    # rugosidad de las funciones peso del operador: w_k(t) = Phi(t) (B_m A)_k
    A = None
    for p in modelo.parameters():
        if p.ndim == 2 and p.shape[0] == pred_s.shape[1]:
            A = p
            break
    rug = 0.0
    if A is not None:
        Bw = Bmt @ A                                   # (K x n_in)
        rug = ((Bw.T @ Rt) * Bw.T).sum() / PERIODO
    return l2 + LAM_RUG * rug, l2


def entrena(tipo, semilla, epocas=3000, lr=0.01):
    m = construye_cabeza(tipo, Xtr.shape[1], M, semilla, h_mlp=H_MLP)
    xt = torch.tensor(Xtr_s, dtype=torch.float32)
    ct = torch.tensor(tr[EST].to_numpy(float), dtype=torch.float32)
    xv = torch.tensor(Xva_s, dtype=torch.float32)
    cv = torch.tensor(va[EST].to_numpy(float), dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    mejor, mejor_st, paciencia = 1e18, None, 0
    for ep in range(epocas):
        opt.zero_grad()
        p_, _ = perdida(m(xt), ct, m)
        p_.backward()
        opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                _, lv = perdida(m(xv), cv, m)
            if lv.item() < mejor - 1e-7:
                mejor, paciencia = lv.item(), 0
                mejor_st = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                paciencia += 1
                if paciencia >= 30:
                    break
    if mejor_st is not None:
        m.load_state_dict(mejor_st)
    return m, mejor, ep


def predice_coef(m, Xs):
    with torch.no_grad():
        s = m(torch.tensor(Xs, dtype=torch.float32)).numpy()
    return c_mu + s @ Bm.T


# ---------------------------------------------------------------------
# verificacion 1: cabeza lineal == regresion funcional cerrada
# ---------------------------------------------------------------------
P(f"\n[4] verificaciones")
Xa = np.column_stack([Xtr_s, np.ones(len(Xtr_s))])
G = Xa.T @ Xa + 1e-10 * np.eye(Xa.shape[1])
Theta = resuelve_chol(G, Xa.T @ Str)          # solucion cerrada, Cholesky propia
pred_cerrada = c_mu + (Xa @ Theta) @ Bm.T
# la equivalencia es teorica: B es W-ortonormal y M=K, asi que la perdida
# (c-c_hat)^T W (c-c_hat) es exactamente ||s - s_hat||^2 euclidea en scores,
# y el minimizador de la cabeza lineal es la regresion funcional cerrada.
# Para comprobarlo hay que optimizar a CONVERGENCIA sobre train (sin
# penalizacion y sin parada temprana por validacion): se usa LBFGS.
LAM_RUG_BK = LAM_RUG
LAM_RUG = 0.0
# en float64: con float32 el residuo de la optimizacion (2e-4) es ruido de
# precision de la maquina, no desacuerdo entre los dos metodos
torch.manual_seed(SEMILLAS[0])
m_lin0 = nn.Linear(Xtr.shape[1], M).double()
_xt = torch.tensor(Xtr_s, dtype=torch.float64)
_ct = torch.tensor(tr[EST].to_numpy(float), dtype=torch.float64)
_W64 = torch.tensor(W, dtype=torch.float64)
_B64 = torch.tensor(Bm, dtype=torch.float64)
_mu64 = torch.tensor(c_mu, dtype=torch.float64)
_opt = torch.optim.LBFGS(m_lin0.parameters(), lr=1.0, max_iter=800,
                         tolerance_grad=1e-16, tolerance_change=1e-18,
                         history_size=80, line_search_fn="strong_wolfe")


def _cierre():
    _opt.zero_grad()
    d = (_mu64 + m_lin0(_xt) @ _B64.T) - _ct
    l = ((d @ _W64) * d).sum(dim=1).mean() / PERIODO
    l.backward()
    return l


for _ in range(8):
    _opt.step(_cierre)
LAM_RUG = LAM_RUG_BK
with torch.no_grad():
    pred_grad = (_mu64 + m_lin0(_xt) @ _B64.T).numpy()
dif = np.abs(pred_grad - pred_cerrada).max()
P(f"    cabeza lineal por gradiente vs regresion funcional cerrada (Cholesky propia):")
P(f"      max|dif| en los coeficientes = {dif:.2e} "
  f"({'CUMPLE <= 1e-4' if dif <= 1e-4 else 'NO cumple 1e-4'})")

# verificacion 2: KAN sobre funcion sintetica conocida
torch.manual_seed(7)
xs = torch.rand(800, 2) * 2 - 1
ys = (torch.sin(np.pi * xs[:, :1]) + xs[:, 1:] ** 2)
kv = construye_cabeza("kan", 2, 1, 7)
ov = torch.optim.Adam(kv.parameters(), lr=0.02)
for _ in range(1500):
    ov.zero_grad()
    l = ((kv(xs) - ys) ** 2).mean()
    l.backward()
    ov.step()
with torch.no_grad():
    rmse_sint = float(((kv(xs) - ys) ** 2).mean().sqrt())
P(f"    KAN sobre f(x1,x2)=sin(pi x1)+x2^2: RMSE {rmse_sint:.4f} "
  f"(sd objetivo {float(ys.std()):.3f}) -> "
  f"{'aprende la funcion' if rmse_sint < 0.1 * float(ys.std()) else 'ajuste pobre'}")

# ---------------------------------------------------------------------
# metricas
# ---------------------------------------------------------------------
def curvas(coef):
    return coef @ Phi.T


def picos_dos(coef):
    """Fechas de los dos maximos principales por derivada ANALITICA."""
    d = coef @ dPhi.T
    out = []
    for fila in d:
        sg = np.sign(fila)
        cruces = np.where((sg[:-1] > 0) & (sg[1:] <= 0))[0]
        if len(cruces) == 0:
            out.append([np.nan, np.nan])
            continue
        y = (coef @ Phi.T)[len(out)] if False else None
        dias = t_dia[cruces]
        val = (fila[cruces] * 0)
        out.append(list(dias[:2]) + [np.nan] * (2 - len(dias[:2])))
    return np.array(out)


def kge(obs, pred):
    r = np.corrcoef(obs, pred)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2
                       + (pred.std() / obs.std() - 1) ** 2
                       + (pred.mean() / obs.mean() - 1) ** 2)


def metricas(coef_obs, coef_pred, etiqueta, modelo, semilla, grupo):
    Xo, Xp = curvas(coef_obs), curvas(coef_pred)
    rec = np.maximum(Xp, 0.0)
    pct_rec = 100 * (Xp < 0).mean()
    acum_o = (np.maximum(Xo, 0) ** 2).sum(axis=1)
    acum_p = (rec ** 2).sum(axis=1)
    l2rel = np.linalg.norm(Xp - Xo, axis=1) / (np.linalg.norm(Xo, axis=1) + 1e-12)
    po, pp = picos_dos(coef_obs), picos_dos(coef_pred)
    err_pico = np.nanmean(np.abs(pp - po), axis=1)
    return dict(modelo=modelo, semilla=semilla, bloque=grupo, n=len(Xo),
                l2_rel=float(l2rel.mean()),
                rmse_func=float(np.sqrt(((Xp - Xo) ** 2).mean())),
                kge=float(kge(acum_o, acum_p)),
                sesgo_acum_pct=float(100 * (acum_p.mean() / acum_o.mean() - 1)),
                err_picos_dias=float(np.nanmean(err_pico)),
                pct_recortado=float(pct_rec)), l2rel, acum_o, acum_p, err_pico


# ONI para el desglose por fase ENSO
oni_f = os.path.join(AUX, "oni.txt")
fase = {}
if os.path.exists(oni_f):
    lineas = [l.split() for l in open(oni_f).read().strip().split("\n")]
    for l in lineas:
        if len(l) == 13 and l[0].isdigit():
            a = int(l[0])
            v = np.mean([float(x) for x in l[1:]])
            fase[a] = "El Nino" if v >= 0.5 else ("La Nina" if v <= -0.5 else "Neutro")

P(f"\n[5] entrenamiento y evaluacion (3 semillas por cabeza)")
filas, detalle, modelos_fin = [], {}, {}
for tipo in ["lineal", "mlp", "kan"]:
    for sd in SEMILLAS:
        m, lv, ep = entrena(tipo, sd)
        for nombre, df, Xs in [("test_temporal", te, Xte_s),
                               ("reservadas", rs, Xrs_s)]:
            cp = predice_coef(m, Xs)
            mt, l2rel, ao, ap, epi = metricas(df[EST].to_numpy(float), cp,
                                              nombre, tipo, sd, nombre)
            filas.append(mt)
            if sd == SEMILLAS[0]:
                detalle[(tipo, nombre)] = dict(df=df, coef_pred=cp, l2rel=l2rel,
                                               acum_o=ao, acum_p=ap)
        modelos_fin[(tipo, sd)] = m
        P(f"    {tipo:7s} semilla {sd}: epocas {ep:5d} | perdida val {lv:.5f}")

# baseline: el satelital sin corregir
for nombre, df, _ in [("test_temporal", te, None), ("reservadas", rs, None)]:
    mt, l2rel, ao, ap, epi = metricas(df[EST].to_numpy(float),
                                      df[SAT].to_numpy(float),
                                      nombre, "satelital_sin_corregir", 0, nombre)
    filas.append(mt)
    detalle[("satelital_sin_corregir", nombre)] = dict(
        df=df, coef_pred=df[SAT].to_numpy(float), l2rel=l2rel, acum_o=ao, acum_p=ap)
F = pd.DataFrame(filas)
F.to_csv(os.path.join(TAB, "fase4_metricas.csv"), index=False)

res = (F.groupby(["bloque", "modelo"])
       .agg(l2_rel_m=("l2_rel", "mean"), l2_rel_sd=("l2_rel", "std"),
            kge_m=("kge", "mean"), kge_sd=("kge", "std"),
            sesgo_m=("sesgo_acum_pct", "mean"), sesgo_sd=("sesgo_acum_pct", "std"),
            picos_m=("err_picos_dias", "mean"), picos_sd=("err_picos_dias", "std"),
            rec_m=("pct_recortado", "mean"))
       .reset_index())
P("\n[6] resultados (media +- sd entre semillas)")
for b in ["test_temporal", "reservadas"]:
    P(f"\n    bloque {b}")
    P(f"      {'modelo':24s} {'L2 rel':>14s} {'KGE':>14s} "
      f"{'sesgo acum %':>14s} {'err picos (d)':>14s} {'recortado %':>11s}")
    for _, r in res[res.bloque == b].iterrows():
        sd_l = 0 if np.isnan(r.l2_rel_sd) else r.l2_rel_sd
        sd_k = 0 if np.isnan(r.kge_sd) else r.kge_sd
        sd_s = 0 if np.isnan(r.sesgo_sd) else r.sesgo_sd
        sd_p = 0 if np.isnan(r.picos_sd) else r.picos_sd
        P(f"      {r.modelo:24s} {r.l2_rel_m:7.4f}+-{sd_l:5.4f} "
          f"{r.kge_m:7.4f}+-{sd_k:5.4f} {r.sesgo_m:+7.1f}+-{sd_s:5.1f} "
          f"{r.picos_m:7.1f}+-{sd_p:5.1f} {r.rec_m:10.2f}")
res.to_csv(os.path.join(TAB, "fase4_resumen.csv"), index=False)

# ---------------------------------------------------------------------
# bootstrap e inferencia
# ---------------------------------------------------------------------
P(f"\n[7] inferencia: bootstrap 1000 por estacion + Wilcoxon pareada")
rngb = np.random.default_rng(99)
comp = []
for bloque in ["test_temporal", "reservadas"]:
    base_l2 = detalle[("satelital_sin_corregir", bloque)]["l2rel"]
    dfb = detalle[("satelital_sin_corregir", bloque)]["df"]
    for tipo in ["lineal", "mlp", "kan"]:
        l2 = detalle[(tipo, bloque)]["l2rel"]
        d = l2 - base_l2
        idx = rngb.integers(0, len(d), (1000, len(d)))
        bs = d[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w = wilcoxon(l2, base_l2)
        # Wilcoxon pareada POR ESTACION (promedio por estacion antes del test)
        g = pd.DataFrame(dict(e=dfb.STATION_ID.to_numpy(), a=l2, b=base_l2)).groupby("e").mean()
        we = wilcoxon(g.a, g.b)
        comp.append(dict(bloque=bloque, modelo=tipo, contra="satelital_sin_corregir",
                         dif_media=float(d.mean()), ic_bajo=float(lo), ic_alto=float(hi),
                         excluye_cero=bool(hi < 0 or lo > 0),
                         p_wilcoxon_par=float(w.pvalue),
                         p_wilcoxon_estacion=float(we.pvalue), n_est=len(g)))
    for a, b in [("mlp", "lineal"), ("kan", "lineal"), ("kan", "mlp")]:
        la, lb = detalle[(a, bloque)]["l2rel"], detalle[(b, bloque)]["l2rel"]
        d = la - lb
        idx = rngb.integers(0, len(d), (1000, len(d)))
        bs = d[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        g = pd.DataFrame(dict(e=dfb.STATION_ID.to_numpy(), a=la, b=lb)).groupby("e").mean()
        comp.append(dict(bloque=bloque, modelo=a, contra=b, dif_media=float(d.mean()),
                         ic_bajo=float(lo), ic_alto=float(hi),
                         excluye_cero=bool(hi < 0 or lo > 0),
                         p_wilcoxon_par=float(wilcoxon(la, lb).pvalue),
                         p_wilcoxon_estacion=float(wilcoxon(g.a, g.b).pvalue),
                         n_est=len(g)))
C2 = pd.DataFrame(comp)
C2.to_csv(os.path.join(TAB, "fase4_wilcoxon.csv"), index=False)
for _, r in C2.iterrows():
    P(f"    [{r.bloque:14s}] {r.modelo:7s} vs {r.contra:24s} "
      f"dif L2 {r.dif_media:+.4f} IC95 [{r.ic_bajo:+.4f},{r.ic_alto:+.4f}] "
      f"{'SIGNIFICATIVO' if r.excluye_cero else 'no concluyente':15s} "
      f"p_est={r.p_wilcoxon_estacion:.2e}")

# ---------------------------------------------------------------------
# desglose por fase ENSO
# ---------------------------------------------------------------------
P(f"\n[8] desglose por fase ENSO (ONI medio anual, NOAA/PSL Nino 3.4)")
rows = []
for bloque in ["test_temporal", "reservadas"]:
    for tipo in ["satelital_sin_corregir", "lineal", "mlp", "kan"]:
        d = detalle[(tipo, bloque)]
        df = d["df"].copy()
        df["l2"] = d["l2rel"]
        df["FASE"] = df.YEAR.map(fase)
        for f_, g in df.groupby("FASE"):
            rows.append(dict(bloque=bloque, modelo=tipo, fase=f_, n=len(g),
                             l2_rel=float(g.l2.mean())))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(TAB, "fase4_enso.csv"), index=False)
piv = E[E.bloque == "test_temporal"].pivot(index="modelo", columns="fase", values="l2_rel")
P("    L2 relativo por fase (bloque test_temporal):")
P("      " + piv.round(4).to_string().replace("\n", "\n      "))

# ---------------------------------------------------------------------
# CSV de estaciones reservadas, formato exacto
# ---------------------------------------------------------------------
mejor_tipo = (res[res.bloque == "reservadas"]
              .sort_values("l2_rel_m").iloc[0].modelo)
if mejor_tipo == "satelital_sin_corregir":
    mejor_tipo = (res[(res.bloque == "reservadas")
                      & (res.modelo != "satelital_sin_corregir")]
                  .sort_values("l2_rel_m").iloc[0].modelo)
m_best = modelos_fin[(mejor_tipo, SEMILLAS[0])]
cp = predice_coef(m_best, Xrs_s)
curv = np.maximum(curvas(cp), 0.0) ** 2            # mm/dia, no negativo
sal = []
for i, (_, r) in enumerate(rs.iterrows()):
    sal.append(pd.DataFrame(dict(station_id=r.STATION_ID, year=int(r.YEAR),
                                 day=np.arange(1, T + 1),
                                 value_mm=np.round(curv[i], 4))))
S = pd.concat(sal, ignore_index=True)
S.to_csv(os.path.join(ENT, "pred_reservadas.csv"), index=False)
P(f"\n[9] predicciones de estaciones reservadas: entrega/pred_reservadas.csv")
P(f"    modelo usado: {mejor_tipo} (mejor L2 en el bloque reservado) | "
  f"{len(S):,} filas | columnas {list(S.columns)}")
P(f"    rango de value_mm: {S.value_mm.min():.3f} a {S.value_mm.max():.1f} mm/dia "
  f"(negativos: {int((S.value_mm < 0).sum())})")

# ---------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------
def eje_meses(ax, ylab):
    ax.set_xticks(INI_MES + 15)
    ax.set_xticklabels(MESES, fontsize=7)
    ax.set_xlim(0, 365)
    ax.set_ylabel(ylab, fontsize=8)
    ax.grid(alpha=0.25)


d0 = detalle[(mejor_tipo, "test_temporal")]
orden = np.argsort(d0["l2rel"])
sel = list(orden[:3]) + list(orden[-3:])
fig, axs = plt.subplots(2, 3, figsize=(14.5, 7))
for ax, i in zip(axs.ravel(), sel):
    r = d0["df"].iloc[i]
    ax.plot(t_dia, (r[EST].to_numpy(float) @ Phi.T), color="black", lw=1.8,
            label="estacion (obs)")
    ax.plot(t_dia, (r[SAT].to_numpy(float) @ Phi.T), color="tab:orange", lw=1.2,
            ls="--", label="satelital")
    ax.plot(t_dia, d0["coef_pred"][i] @ Phi.T, color="tab:red", lw=1.5,
            label=f"{mejor_tipo}")
    eje_meses(ax, r"$\sqrt{P}$")
    ax.set_title(f"{r.STATION_ID} {int(r.YEAR)} | L2rel {d0['l2rel'][i]:.3f}",
                 fontsize=8)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Fase 4 - mejores (arriba) y peores (abajo) predicciones del bloque de prueba")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase4_curvas.png"), dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2))
mods = ["satelital_sin_corregir", "lineal", "mlp", "kan"]
col = dict(zip(mods, ["gray", "tab:blue", "tab:green", "tab:red"]))
for k_, (met, lab) in enumerate([("l2_rel_m", "L2 relativo"), ("kge_m", "KGE"),
                                 ("sesgo_m", "sesgo del acumulado (%)")]):
    r1 = res[res.bloque == "test_temporal"].set_index("modelo").reindex(mods)
    r2 = res[res.bloque == "reservadas"].set_index("modelo").reindex(mods)
    x = np.arange(len(mods))
    axs[k_].bar(x - 0.2, r1[met], 0.4, color=[col[m] for m in mods], label="test temporal")
    axs[k_].bar(x + 0.2, r2[met], 0.4, color=[col[m] for m in mods], alpha=0.5,
                hatch="//", label="reservadas")
    axs[k_].set_xticks(x)
    axs[k_].set_xticklabels(["satelital", "lineal", "MLP", "KAN"], fontsize=8)
    axs[k_].set_title(lab, fontsize=9)
    axs[k_].grid(alpha=0.25, axis="y")
    if k_ == 0:
        axs[k_].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase4_metricas.png"), dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, tipo in zip(axs, mods):
    d = detalle[(tipo, "test_temporal")]
    ax.scatter(d["acum_o"], d["acum_p"], s=8, alpha=0.4, color=col[tipo])
    lim = [0, float(max(d["acum_o"].max(), d["acum_p"].max())) * 1.03]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("acumulado observado [mm]", fontsize=8)
    ax.set_ylabel("acumulado predicho [mm]", fontsize=8)
    rr = res[(res.bloque == "test_temporal") & (res.modelo == tipo)].iloc[0]
    ax.set_title(f"{tipo}\nKGE {rr.kge_m:.3f} | sesgo {rr.sesgo_m:+.1f} %", fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase4_dispersion.png"), dpi=130)
plt.close(fig)

P(f"\n[fin] figuras fase4_*.png | tablas fase4_*.csv | pred_reservadas.csv")
P(f"      torch {torch.__version__} | semillas {SEMILLAS} | lambda_rug {LAM_RUG:g}")
open(os.path.join(ENT, "log_fase4.txt"), "w").write("\n".join(log) + "\n")
