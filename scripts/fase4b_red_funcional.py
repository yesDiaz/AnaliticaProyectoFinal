#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 4b: red funcional con CODIFICADOR APRENDIBLE y particion oficial.

Cambios respecto a la fase 4 original, exigidos por la guia corregida del
proyecto (fe de erratas del 9-nov-2026):

  1. Codificador. Antes:  z = c_sat W^(1/2)   (operador FIJO).
     Ahora:  z_k = <w_k, X> = a_k^T Wx c_sat, con w_k(t) = sum_b a_kb B_b(t)
     en base B-spline cubica y a_k APRENDIBLES por descenso de gradiente.
     Wx = integral de B(t) Phi(t)^T dt (nb x K) se calcula por cuadratura y la
     integral se verifica contra cuadratura fina independiente.

  2. Particion temporal oficial: 1981-1995 / 1996-2000 / 2001-2003.

  3. Datos: pares estacion-pixel con COORDENADAS REALES (fase 3b) y altitud
     real de la estacion como covariable.

Ablacion incluida: codificador fijo (a_k congelados en W^(1/2)) contra
codificador aprendible, con las tres cabezas (lineal, MLP, KAN).

Salidas:
  entrega/tablas/fase4b_*.csv
  entrega/figs/fase4b_*.png
  entrega/modelos/  (modelos entrenados para la fase 4.5)
  entrega/log_fase4b.txt
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
RAIZ = os.path.dirname(os.path.abspath(__file__))
ENT = os.path.join(RAIZ, "entrega")
TAB = os.path.join(ENT, "tablas")
FIG = os.path.join(ENT, "figs")
MOD = os.path.join(ENT, "modelos")
os.makedirs(MOD, exist_ok=True)

K, PERIODO = 5, 365.0
NB_NODOS = 8                 # nodos interiores de la base B-spline de las w_k
N_NEU = 5                    # neuronas funcionales del codificador
LAM_RUG = 1e-6
FRAC_RESERVADAS = 0.13
SEMILLAS = [11, 23, 37]
ANIOS_TR = list(range(1981, 1996))      # oficial: 1981-1995
ANIOS_VA = list(range(1996, 2001))      # oficial: 1996-2000
ANIOS_TE = list(range(2001, 2004))      # oficial: 2001-2003

log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------------
# bases y matrices de Gram (cuadratura trapezoidal, a mano)
# ---------------------------------------------------------------------
def base_fourier(t, K, periodo=PERIODO):
    cols = [np.ones_like(t)]
    for j in range(1, K // 2 + 1):
        cols.append(np.sin(2 * np.pi * j * t / periodo))
        cols.append(np.cos(2 * np.pi * j * t / periodo))
    return np.column_stack(cols[:K])


def nodos_bspline(n_int, k=3, a=0.0, b=PERIODO):
    interior = np.linspace(a, b, n_int + 2)[1:-1]
    return np.concatenate([[a] * (k + 1), interior, [b] * (k + 1)])


def base_bspline(t, nodos, k=3):
    """B-splines por recurrencia de Cox-de Boor (sin scipy)."""
    n = len(nodos) - k - 1
    B = np.zeros((len(t), len(nodos) - 1))
    for i in range(len(nodos) - 1):
        B[:, i] = ((t >= nodos[i]) & (t < nodos[i + 1])).astype(float)
    B[t >= nodos[-1], -1] = 0.0
    B[np.isclose(t, nodos[-1]), np.where(nodos < nodos[-1])[0][-1]] = 1.0
    for p in range(1, k + 1):
        Bn = np.zeros((len(t), len(nodos) - p - 1))
        for i in range(len(nodos) - p - 1):
            d1 = nodos[i + p] - nodos[i]
            d2 = nodos[i + p + 1] - nodos[i + 1]
            a1 = (t - nodos[i]) / d1 if d1 > 0 else np.zeros_like(t)
            a2 = (nodos[i + p + 1] - t) / d2 if d2 > 0 else np.zeros_like(t)
            Bn[:, i] = a1 * B[:, i] + a2 * B[:, i + 1]
        B = Bn
    return B[:, :n]


def pesos_trapecio(t):
    w = np.zeros_like(t)
    w[1:-1] = (t[2:] - t[:-2]) / 2
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    return w


t_dia = np.arange(365) + 0.5
Phi = base_fourier(t_dia, K)
NODOS = nodos_bspline(NB_NODOS)
Bsp = base_bspline(t_dia, NODOS)
NB = Bsp.shape[1]
wq = pesos_trapecio(t_dia)

W = Phi.T @ (Phi * wq[:, None])              # Gram de Fourier (K x K)

# Wx y RB se calculan en malla FINA con regla de Simpson: con 365 nodos y
# trapecio el error relativo de <w_k, X> es 8e-3, insuficiente para que la
# integral del codificador sea exacta a 1e-4.
NF = 36501                                   # impar: Simpson compuesto
t_f = np.linspace(0.0, PERIODO, NF)
hf = t_f[1] - t_f[0]
ws = np.ones(NF)
ws[1:-1:2] = 4.0
ws[2:-1:2] = 2.0
ws *= hf / 3.0
Phi_f = base_fourier(t_f, K)
Bsp_f = base_bspline(t_f, NODOS)
Wx = Bsp_f.T @ (Phi_f * ws[:, None])         # integral B Phi^T  (NB x K)
d2B_f = np.gradient(np.gradient(Bsp_f, t_f, axis=0), t_f, axis=0)
RB = d2B_f.T @ (d2B_f * ws[:, None])         # (NB x NB)
lamW, Uw = np.linalg.eigh(W)
Wh = Uw @ np.diag(np.sqrt(lamW)) @ Uw.T
Whi = Uw @ np.diag(1 / np.sqrt(lamW)) @ Uw.T

P("=" * 74)
P("FASE 4b - codificador funcional APRENDIBLE, particion oficial")
P("=" * 74)
P(f"    base de las w_k: B-spline cubica, {NB_NODOS} nodos interiores -> NB = {NB}")
P(f"    particion de la base: suma de filas = {Bsp.sum(axis=1).min():.6f} "
  f"a {Bsp.sum(axis=1).max():.6f}")
P(f"    Wx = int B Phi^T: {Wx.shape} | ||Wx|| = {np.linalg.norm(Wx):.2f}")
P(f"    Gram W de Fourier: traza {np.trace(W):.2f}")

# ---------------------------------------------------------------------
# verificacion de la integral del codificador
# ---------------------------------------------------------------------
rng_v = np.random.default_rng(1)
a_test = rng_v.normal(size=NB)
c_test = rng_v.normal(size=K)
z_matriz = float(a_test @ Wx @ c_test)
t_fino = np.linspace(0, PERIODO, 400003)   # malla independiente de la de Wx
w_fino = base_bspline(t_fino, NODOS) @ a_test
x_fino = base_fourier(t_fino, K) @ c_test
z_fino = float(np.trapezoid(w_fino * x_fino, t_fino))
err_rel = abs(z_matriz - z_fino) / (abs(z_fino) + 1e-12)
P(f"\n[0] verificacion del producto interno del codificador")
P(f"    a^T Wx c            = {z_matriz:.10f}")
P(f"    int w(t) X(t) dt    = {z_fino:.10f}   (cuadratura fina, 2e5 nodos)")
P(f"    error relativo      = {err_rel:.2e} "
  f"({'CUMPLE <= 1e-4' if err_rel <= 1e-4 else 'NO CUMPLE'})")

# ---------------------------------------------------------------------
# datos y particiones oficiales
# ---------------------------------------------------------------------
D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
SAT = [f"sat_c{i+1}" for i in range(K)]
EST = [f"est_c{i+1}" for i in range(K)]
COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
P(f"\n[1] datos (coordenadas reales de la fase 3b): {len(D)} pares | "
  f"{D.STATION_ID.nunique()} estaciones | {D.YEAR.min()}-{D.YEAR.max()}")

rng0 = np.random.default_rng(2026)
ests = np.array(sorted(D.STATION_ID.unique()))
reserv = set(rng0.choice(ests, int(round(FRAC_RESERVADAS * len(ests))), replace=False))
D["RESERVADA"] = D.STATION_ID.isin(reserv)
tr = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TR)].reset_index(drop=True)
va = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_VA)].reset_index(drop=True)
te = D[(~D.RESERVADA) & D.YEAR.isin(ANIOS_TE)].reset_index(drop=True)
rs = D[D.RESERVADA].reset_index(drop=True)
P(f"    particion OFICIAL: train {len(tr)} (1981-1995) | val {len(va)} (1996-2000) | "
  f"test {len(te)} (2001-2003)")
P(f"    estaciones reservadas por sorteo propio (el docente no publico lista): "
  f"{len(reserv)} de {len(ests)} -> {len(rs)} pares")

# ---------------------------------------------------------------------
# decodificador FPCA (solo train, a mano por autovalores generalizados)
# ---------------------------------------------------------------------
Ctr = tr[EST].to_numpy(float)
c_mu = Ctr.mean(axis=0)
V = ((Ctr - c_mu).T @ (Ctr - c_mu)) / (len(Ctr) - 1)
lamF, Uf = np.linalg.eigh(Wh @ V @ Wh)
o = np.argsort(lamF)[::-1]
lamF, Uf = lamF[o], Uf[:, o]
Bfull = Whi @ Uf
vexp = lamF / lamF.sum()
M = int(np.searchsorted(np.cumsum(vexp), 0.99) + 1)
Bm = Bfull[:, :M]
P(f"\n[2] decodificador FPCA (solo train): " +
  " ".join(f"PC{j+1} {100*vexp[j]:.1f}%" for j in range(K)))
P(f"    M = {M} componentes para el 99 % | ortonormalidad max|B^T W B - I| = "
  f"{np.abs(Bm.T @ W @ Bm - np.eye(M)).max():.2e}")

mcov = tr[COV].to_numpy(float).mean(axis=0)
scov = tr[COV].to_numpy(float).std(axis=0) + 1e-12


def tensores(df):
    return (torch.tensor(df[SAT].to_numpy(float), dtype=torch.float32),
            torch.tensor((df[COV].to_numpy(float) - mcov) / scov, dtype=torch.float32),
            torch.tensor(df[EST].to_numpy(float), dtype=torch.float32))


Ctr_s, Vtr, Ytr = tensores(tr)
Cva_s, Vva, Yva = tensores(va)
Cte_s, Vte, Yte = tensores(te)
Crs_s, Vrs, Yrs = tensores(rs)

Wxt = torch.tensor(Wx, dtype=torch.float32)
RBt = torch.tensor(RB, dtype=torch.float32)
Wt = torch.tensor(W, dtype=torch.float32)
Bmt = torch.tensor(Bm, dtype=torch.float32)
cmut = torch.tensor(c_mu, dtype=torch.float32)


# ---------------------------------------------------------------------
# codificador funcional: z_k = a_k^T Wx c   con a_k aprendibles
# ---------------------------------------------------------------------
class CodificadorFuncional(nn.Module):
    """Capa de neuronas funcionales con pesos w_k(t) en base B-spline.

    modo = "fijo"       : a_k congelados en el operador W^(1/2) (fase 4 original)
    modo = "aleatorio"  : a_k aprendibles desde una inicializacion aleatoria
    modo = "informado"  : a_k aprendibles pero INICIALIZADOS en W^(1/2), de modo
                          que la red parte del operador teorico y lo ajusta
    """

    def __init__(self, n_neu=N_NEU, modo="informado", semilla=0):
        super().__init__()
        g = torch.Generator().manual_seed(semilla)
        if modo == "aleatorio":
            A0 = torch.randn(NB, n_neu, generator=g) * 0.1
        else:
            # a_k que reproduce z = c W^(1/2): se resuelve Wx^T a_k = (W^(1/2))_k
            objetivo = torch.tensor(Wh, dtype=torch.float32)[:, :n_neu]
            A0 = torch.linalg.lstsq(Wxt.T, objetivo).solution       # (NB x n_neu)
        self.A = nn.Parameter(A0, requires_grad=(modo != "fijo"))
        self.modo = modo
        self.escala = None

    def forward(self, c):
        z = c @ Wxt.T @ self.A
        if self.escala is not None:
            z = z / self.escala
        return z

    def pesos_wk(self):
        """w_k(t) evaluadas en la malla diaria (365 x n_neu)."""
        return torch.tensor(Bsp, dtype=torch.float32) @ self.A

    def rugosidad(self):
        return ((self.A.T @ RBt) * self.A.T).sum() / PERIODO


class KANPropia(nn.Module):
    """KAN minima: funciones univariadas en B-spline cubica (respaldo)."""

    def __init__(self, n_in, n_out, grid=5, k=3):
        super().__init__()
        self.n_in, self.k = n_in, k
        paso = 1.0 / grid
        self.nodos = torch.linspace(-0.5 - k * paso, 1.5 + k * paso, grid + 2 * k + 2)
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
        B = self.bspline(torch.tanh(x) * 0.5 + 0.5)
        y = torch.einsum("nib,ibo->no", B, self.coef)
        return y + torch.einsum("ni,io->no", torch.tanh(x), self.w_lin) + self.b


def construye_cabeza(tipo, n_in, n_out, semilla, h_mlp=None):
    torch.manual_seed(semilla)
    if tipo == "lineal":
        return nn.Linear(n_in, n_out)
    if tipo == "mlp":
        return nn.Sequential(nn.Linear(n_in, h_mlp), nn.GELU(), nn.Linear(h_mlp, n_out))
    if tipo == "kan":
        try:
            from kan import KAN
            return KAN(width=[n_in, n_out], grid=5, k=3, seed=semilla, device="cpu")
        except Exception as e:
            P(f"    [aviso] pykan no utilizable ({type(e).__name__}); KANPropia")
        return KANPropia(n_in, n_out)
    raise ValueError(tipo)


class RedFuncional(nn.Module):
    def __init__(self, tipo, semilla, h_mlp=None, modo="informado", n_neu=N_NEU):
        super().__init__()
        self.cod = CodificadorFuncional(n_neu, modo=modo, semilla=semilla)
        self.cabeza = construye_cabeza(tipo, n_neu + len(COV), M, semilla, h_mlp)
        self.tipo, self.n_neu, self.modo = tipo, n_neu, modo

    def forward(self, c, v):
        z = self.cod(c)
        return self.cabeza(torch.cat([z, v], dim=1))


def n_par(m, solo_entrenables=True):
    ps = [p for p in m.parameters() if (p.requires_grad or not solo_entrenables)]
    return sum(p.numel() for p in ps)


# h del MLP que iguala los parametros de la KAN (<= 10 %)
_k0 = construye_cabeza("kan", N_NEU + len(COV), M, SEMILLAS[0])
npar_kan = sum(p.numel() for p in _k0.parameters())
usa_pykan = "kan" in type(_k0).__module__
mejor_h, mejor_d = 1, 1e9
for h in range(1, 400):
    d = abs(sum(p.numel() for p in construye_cabeza("mlp", N_NEU + len(COV), M, 0, h).parameters())
            - npar_kan)
    if d < mejor_d:
        mejor_h, mejor_d = h, d
H_MLP = mejor_h
npar_mlp = sum(p.numel() for p in construye_cabeza("mlp", N_NEU + len(COV), M, 0, H_MLP).parameters())
P(f"\n[3] parametros de las cabezas: KAN ({'pykan' if usa_pykan else 'propia'}) {npar_kan} | "
  f"MLP (h={H_MLP}) {npar_mlp} | diferencia "
  f"{100*abs(npar_kan-npar_mlp)/npar_kan:.1f} %")
P(f"    codificador aprendible: {NB} x {N_NEU} = {NB*N_NEU} parametros adicionales")


def perdida(pred_s, obs_c, red):
    c = cmut + pred_s @ Bmt.T
    d = c - obs_c
    l2 = ((d @ Wt) * d).sum(dim=1).mean() / PERIODO
    rug = red.cod.rugosidad() if red.cod.A.requires_grad else 0.0
    return l2 + LAM_RUG * rug, l2


def entrena(tipo, semilla, modo="informado", epocas=4000, lr=0.01, n_neu=N_NEU):
    red = RedFuncional(tipo, semilla, H_MLP, modo, n_neu)
    with torch.no_grad():                      # escala inicial de z (sd ~ 1 en train)
        z0 = red.cod(Ctr_s)
        red.cod.escala = z0.std(dim=0) + 1e-9
    opt = torch.optim.Adam([p for p in red.parameters() if p.requires_grad], lr=lr)
    mejor, st, pac = 1e18, None, 0
    for ep in range(epocas):
        opt.zero_grad()
        p_, _ = perdida(red(Ctr_s, Vtr), Ytr, red)
        p_.backward()
        opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                _, lv = perdida(red(Cva_s, Vva), Yva, red)
            if lv.item() < mejor - 1e-7:
                mejor, pac = lv.item(), 0
                st = {k: v.clone() for k, v in red.state_dict().items()}
            else:
                pac += 1
                if pac >= 40:
                    break
    if st is not None:
        red.load_state_dict(st)
    return red, mejor, ep


# ---------------------------------------------------------------------
# metricas
# ---------------------------------------------------------------------
def metricas(red, C, Vv, Y):
    with torch.no_grad():
        s = red(C, Vv)
        coef = (cmut + s @ Bmt.T).numpy()
    obs = Y.numpy()
    cur_p, cur_o = coef @ Phi.T, obs @ Phi.T
    dif = coef - obs
    l2 = np.sqrt(((dif @ W) * dif).sum(axis=1) / PERIODO)
    nor = np.sqrt(((obs @ W) * obs).sum(axis=1) / PERIODO) + 1e-12
    acu_p, acu_o = cur_p.sum(axis=1), cur_o.sum(axis=1)
    r = np.corrcoef(acu_p, acu_o)[0, 1]
    alfa = acu_p.std() / (acu_o.std() + 1e-12)
    beta = acu_p.mean() / (acu_o.mean() + 1e-12)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alfa - 1) ** 2 + (beta - 1) ** 2)
    return dict(l2_rel=float(np.mean(l2 / nor)), rmse=float(np.sqrt((dif ** 2).mean())),
                sesgo_acum_pct=float(100 * (acu_p.mean() - acu_o.mean()) / acu_o.mean()),
                kge=float(kge), r_acum=float(r))


MODOS = [("fijo", "fijo W^1/2"), ("aleatorio", "aprend. aleatorio"),
         ("informado", "aprend. informado")]
P(f"\n[4] entrenamiento: 3 cabezas x {len(MODOS)} codificadores x {len(SEMILLAS)} semillas")
filas = []
modelos = {}
for modo, etq in MODOS:
    for tipo in ("lineal", "mlp", "kan"):
        for sem in SEMILLAS:
            red, lv, ep = entrena(tipo, sem, modo)
            mt = metricas(red, Cte_s, Vte, Yte)
            mr = metricas(red, Crs_s, Vrs, Yrs)
            filas.append(dict(codificador=etq, cabeza=tipo, semilla=sem, epocas=ep,
                              val=lv, **{f"test_{k}": v for k, v in mt.items()},
                              **{f"resv_{k}": v for k, v in mr.items()}))
            modelos[(modo, tipo, sem)] = red
            P(f"    {etq:18s} {tipo:7s} s{sem}: val {lv:.4f} | "
              f"test L2rel {mt['l2_rel']:.3f} KGE {mt['kge']:+.3f} | "
              f"resv L2rel {mr['l2_rel']:.3f} ({ep} ep)")
F = pd.DataFrame(filas)
F.to_csv(os.path.join(TAB, "fase4b_ablacion.csv"), index=False)

res = (F.groupby(["codificador", "cabeza"])
       [["test_l2_rel", "test_kge", "test_sesgo_acum_pct", "resv_l2_rel", "resv_kge"]]
       .agg(["mean", "std"]).round(4))
P("\n[5] resumen por codificador y cabeza (media +- sd sobre semillas)")
P(res.to_string())
res.to_csv(os.path.join(TAB, "fase4b_resumen.csv"))

P("\n    L2 relativo medio en test por codificador:")
for _, etq in MODOS:
    P(f"      {etq:18s} {F[F.codificador == etq].test_l2_rel.mean():.4f}")
mf = F[F.codificador == "fijo W^1/2"].test_l2_rel.mean()
mi = F[F.codificador == "aprend. informado"].test_l2_rel.mean()
P(f"    aprendible informado vs fijo: {100*(mf-mi)/mf:+.1f} % de mejora")

# ---------------------------------------------------------------------
# por que el codificador aprendible no gana con 5 neuronas: con n_neu = K el
# operador fijo W^(1/2) ya genera TODO el espacio de X (dim K = 5), asi que
# aprender a_k solo aplica un cambio de base invertible. La ventaja debe
# aparecer cuando el codificador COMPRIME (n_neu < K).
P("\n[5b] barrido del numero de neuronas funcionales (cabeza MLP, 2 semillas)")
f2 = []
for n_neu in (1, 2, 3, 5):
    for modo, etq in MODOS:
        ms = []
        for sem in SEMILLAS[:2]:
            red, lv, ep = entrena("mlp", sem, modo, n_neu=n_neu)
            ms.append(metricas(red, Cte_s, Vte, Yte)["l2_rel"])
        f2.append(dict(n_neuronas=n_neu, codificador=etq, test_l2_rel=float(np.mean(ms))))
        P(f"    n_neu {n_neu} | {etq:18s} | test L2rel {np.mean(ms):.4f}")
Fc = pd.DataFrame(f2)
Fc.to_csv(os.path.join(TAB, "fase4b_barrido_neuronas.csv"), index=False)
piv = Fc.pivot(index="n_neuronas", columns="codificador", values="test_l2_rel")
piv["ganancia_informado_pct"] = (100 * (piv["fijo W^1/2"] - piv["aprend. informado"])
                                 / piv["fijo W^1/2"])
P("\n" + piv.round(4).to_string())

# modelo elegido para la fase 4.5: el codificador aprendible informado (cumple
# la especificacion 4.2 de la guia) con la mejor perdida de validacion
MODO_FINAL = "aprend. informado"
sub = F[(F.codificador == MODO_FINAL) & (F.cabeza == "kan")]
sem_kan = int(sub.loc[sub.val.idxmin(), "semilla"])
sub_l = F[(F.codificador == MODO_FINAL) & (F.cabeza == "lineal")]
sem_lin = int(sub_l.loc[sub_l.val.idxmin(), "semilla"])
for nom, llave in [("kan", ("informado", "kan", sem_kan)),
                   ("lineal", ("informado", "lineal", sem_lin))]:
    r = modelos[llave]
    torch.save(dict(state_dict=r.state_dict(), tipo=r.tipo, n_neu=r.n_neu, modo=r.modo,
                    h_mlp=H_MLP, escala=r.cod.escala, M=M),
               os.path.join(MOD, f"red_{nom}_aprendible.pt"))
np.savez(os.path.join(MOD, "contexto_fase4b.npz"), Phi=Phi, Bsp=Bsp, NODOS=NODOS, W=W,
         Wx=Wx, RB=RB, Bm=Bm, c_mu=c_mu, mcov=mcov, scov=scov, t_dia=t_dia, M=M,
         vexp=vexp, sem_kan=sem_kan, sem_lin=sem_lin, usa_pykan=usa_pykan)
P(f"\n[6] modelos guardados en entrega/modelos/ (KAN semilla {sem_kan}, "
  f"lineal semilla {sem_lin}, codificador {MODO_FINAL})")

# figura: funciones peso aprendidas vs fijas
red_a = modelos[("informado", "kan", sem_kan)]
red_f = modelos[("fijo", "kan", SEMILLAS[0])]
Wk_a = red_a.cod.pesos_wk().detach().numpy()
Wk_f = red_f.cod.pesos_wk().detach().numpy()
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for k in range(N_NEU):
    axs[0].plot(t_dia, Wk_f[:, k], label=f"w{k+1}")
    axs[1].plot(t_dia, Wk_a[:, k], label=f"w{k+1}")
axs[0].set_title("codificador fijo: $w_k$ que reproduce $W^{1/2}$")
axs[1].set_title("codificador aprendible (inicio informado): $w_k$ ajustadas")
for ax in axs:
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("dia del anio")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
axs[0].set_ylabel("$w_k(t)$")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase4b_pesos_codificador.png"), dpi=150)

with open(os.path.join(ENT, "log_fase4b.txt"), "w") as f:
    f.write("\n".join(log))
ok = (err_rel <= 1e-4)
P(f"\nFASE 4b {'OK' if ok else 'CON FALLOS'} (verificacion de la integral: {err_rel:.1e})")
raise SystemExit(0 if ok else 1)
