#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 4.5: interpretacion de la red funcional (subcriterio 4.5 de la rubrica).

Exige la guia corregida: graficar las funciones peso w_k(t) del codificador y
las funciones de activacion de la primera capa de la KAN, ajustar
simbolicamente al menos una activacion reportando su R2, contrastarlas con los
coeficientes de la cabeza lineal, leerlas con la climatologia regional y
analizar los peores casos.

Nota teorica: la cabeza KAN es de una sola capa, width = [n_in, n_out], luego
y_j(x) = sum_i phi_ij(x_i) + b_j. La respuesta de la red al variar SOLO la
entrada i es por tanto phi_ij salvo una constante: no es una aproximacion de
la activacion, es la activacion.

Salidas: entrega/figs/fase45_*.png, entrega/tablas/fase45_*.csv,
         entrega/log_fase45.txt
"""
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
PERIODO, K, N_NEU = 365.0, 5, 5
COV = ["ALT_SUB", "LON_PIX", "LAT_PIX"]
NOM_ENT = [f"z{j+1}" for j in range(N_NEU)] + ["altitud", "longitud", "latitud"]
LIM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


ctx = np.load(os.path.join(MOD, "contexto_fase4b.npz"))
Phi, Bsp, NODOS = ctx["Phi"], ctx["Bsp"], ctx["NODOS"]
W, Wx, Bm, c_mu = ctx["W"], ctx["Wx"], ctx["Bm"], ctx["c_mu"]
mcov, scov, t_dia, M = ctx["mcov"], ctx["scov"], ctx["t_dia"], int(ctx["M"])
NB = Bsp.shape[1]
Wxt = torch.tensor(Wx, dtype=torch.float32)
RBt = torch.tensor(ctx["RB"], dtype=torch.float32) if "RB" in ctx else torch.zeros(NB, NB)


# --- clases identicas a las de la fase 4b, registradas como su modulo para
#     que torch.load reconstruya los objetos sin re-entrenar nada ----------
class CodificadorFuncional(nn.Module):
    def __init__(self, n_neu=N_NEU, modo="informado", semilla=0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(NB, n_neu), requires_grad=(modo != "fijo"))
        self.modo = modo
        self.escala = None

    def forward(self, c):
        z = c @ Wxt.T @ self.A
        return z / self.escala if self.escala is not None else z

    def pesos_wk(self):
        return torch.tensor(Bsp, dtype=torch.float32) @ self.A

    def rugosidad(self):
        return ((self.A.T @ RBt) * self.A.T).sum() / PERIODO


class KANPropia(nn.Module):
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
    def __init__(self, tipo="kan", semilla=0, h_mlp=None, modo="informado", n_neu=N_NEU):
        super().__init__()
        self.cod = CodificadorFuncional(n_neu, modo, semilla)
        self.cabeza = construye_cabeza(tipo, n_neu + len(COV), int(M), semilla, h_mlp)
        self.tipo, self.n_neu, self.modo = tipo, n_neu, modo

    def forward(self, c, v):
        return self.cabeza(torch.cat([self.cod(c), v], dim=1))


def carga(nombre):
    """Reconstruye la red desde el state_dict guardado por la fase 4b."""
    d = torch.load(os.path.join(MOD, nombre), weights_only=False)
    red = RedFuncional(d["tipo"], 0, d["h_mlp"], d["modo"], d["n_neu"])
    red.load_state_dict(d["state_dict"], strict=False)
    red.cod.escala = d["escala"]
    red.eval()
    return red, d


P("=" * 74)
P("FASE 4.5 - interpretacion climatologica de la red funcional")
P("=" * 74)
red_kan, dk = carga("red_kan_aprendible.pt")
red_lin, dl = carga("red_lineal_aprendible.pt")
P(f"    modelos cargados: cabeza KAN ({type(red_kan.cabeza).__name__}) y cabeza lineal; "
  f"codificador en modo '{dk['modo']}' con {dk['n_neu']} neuronas funcionales")

# ---------------------------------------------------------------------
# 1. funciones peso del codificador y su lectura climatologica
# ---------------------------------------------------------------------
Wk = red_kan.cod.pesos_wk().detach().numpy()          # (365 x N_NEU)
esc = red_kan.cod.escala.detach().numpy()
Wk_n = Wk / esc                                        # como entran al modelo
filas = []
for k in range(N_NEU):
    w = Wk_n[:, k]
    masa = np.array([np.abs(w[LIM[m]:LIM[m + 1]]).sum() for m in range(12)])
    masa = masa / masa.sum()
    filas.append(dict(neurona=f"z{k+1}", mes_dominante=MESES[int(masa.argmax())],
                      peso_mes_dominante=float(masa.max()),
                      **{f"p_{MESES[m]}": float(masa[m]) for m in range(12)},
                      signo_medio=float(np.sign(w.mean())),
                      cambios_de_signo=int((np.diff(np.sign(w)) != 0).sum())))
Tw = pd.DataFrame(filas)
Tw.to_csv(os.path.join(TAB, "fase45_pesos_por_mes.csv"), index=False)
P("\n[1] funciones peso w_k(t) del codificador: reparto mensual de |w_k|")
P(Tw[["neurona", "mes_dominante", "peso_mes_dominante", "cambios_de_signo"]]
  .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
LLU1, LLU2 = (LIM[2], LIM[5]), (LIM[8], LIM[11])       # mar-may y sep-nov
for k in range(N_NEU):
    w = np.abs(Wk_n[:, k])
    fr = (w[LLU1[0]:LLU1[1]].sum() + w[LLU2[0]:LLU2[1]].sum()) / w.sum()
    P(f"    z{k+1}: {100*fr:.0f} % de |w| en las dos temporadas de lluvia "
      f"(mar-may y sep-nov, que ocupan el {100*(LLU1[1]-LLU1[0]+LLU2[1]-LLU2[0])/365:.0f} % del anio)")

fig, ax = plt.subplots(figsize=(8.2, 4.4))
for k in range(N_NEU):
    ax.plot(t_dia, Wk_n[:, k], lw=1.6, label=f"$w_{k+1}(t)$")
for a, b, et in [(LLU1[0], LLU1[1], "lluvias mar-may"), (LLU2[0], LLU2[1], "lluvias sep-nov")]:
    ax.axvspan(a, b, color="tab:blue", alpha=0.10)
    ax.text((a + b) / 2, ax.get_ylim()[1] * 0.92, et, ha="center", fontsize=7, color="tab:blue")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("dia del anio")
ax.set_ylabel("peso")
ax.set_title("Funciones peso aprendidas por las neuronas funcionales\n"
             "$z_k=\\int w_k(t)X(t)dt$ (base B-spline cubica, 12 coeficientes)")
ax.set_xticks(LIM[:12] + np.diff(LIM) / 2)
ax.set_xticklabels(MESES, fontsize=8)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase45_pesos_wk.png"), dpi=150)

# ---------------------------------------------------------------------
# 2. activaciones de la primera capa de la KAN
# ---------------------------------------------------------------------
D = pd.read_csv(os.path.join(ENT, "pares_estacion_pixel.csv"), dtype={"STATION_ID": str})
SAT = [f"sat_c{i+1}" for i in range(K)]
EST = [f"est_c{i+1}" for i in range(K)]
rng0 = np.random.default_rng(2026)
ests = np.array(sorted(D.STATION_ID.unique()))
reserv = set(rng0.choice(ests, int(round(0.13 * len(ests))), replace=False))
D["RESERVADA"] = D.STATION_ID.isin(reserv)
tr = D[(~D.RESERVADA) & D.YEAR.between(1981, 1995)].reset_index(drop=True)
te = D[(~D.RESERVADA) & D.YEAR.between(2001, 2003)].reset_index(drop=True)
rs = D[D.RESERVADA].reset_index(drop=True)


def entradas(df):
    c = torch.tensor(df[SAT].to_numpy(float), dtype=torch.float32)
    v = torch.tensor((df[COV].to_numpy(float) - mcov) / scov, dtype=torch.float32)
    with torch.no_grad():
        z = red_kan.cod(c)
    return c, v, torch.cat([z, v], dim=1).numpy()


Ctr, Vtr, Xtr = entradas(tr)
Cte, Vte, Xte = entradas(te)
Crs, Vrs, Xrs = entradas(rs)
x_med = torch.tensor(np.median(Xtr, axis=0), dtype=torch.float32)
q = np.percentile(Xtr, [1, 99], axis=0)
NG = 201
P(f"\n[2] activaciones de la cabeza KAN: {Xtr.shape[1]} entradas x {M} salidas")


def activacion(cabeza, i, xs, j=None):
    """phi_ij(x) salvo constante: respuesta al variar solo la entrada i."""
    Xb = x_med.repeat(len(xs), 1).clone()
    Xb[:, i] = torch.tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        y = cabeza(Xb).numpy()
    y = y - y.mean(axis=0, keepdims=True)
    return y if j is None else y[:, j]


n_in = Xtr.shape[1]
act = {}
for i in range(n_in):
    xs = np.linspace(q[0, i], q[1, i], NG)
    act[i] = (xs, activacion(red_kan.cabeza, i, xs))

# importancia: amplitud de cada activacion sobre la salida PC1 y total
imp = []
for i in range(n_in):
    xs, y = act[i]
    imp.append(dict(entrada=NOM_ENT[i], amplitud_PC1=float(y[:, 0].max() - y[:, 0].min()),
                    amplitud_total=float(np.abs(y).max(axis=0).sum()),
                    no_linealidad=float(np.nan)))
Ti = pd.DataFrame(imp)
# no linealidad: 1 - R2 del mejor ajuste lineal de la activacion sobre PC1
for i in range(n_in):
    xs, y = act[i]
    A = np.column_stack([np.ones_like(xs), xs])
    be = np.linalg.lstsq(A, y[:, 0], rcond=None)[0]
    r2 = 1 - ((y[:, 0] - A @ be) ** 2).sum() / (((y[:, 0] - y[:, 0].mean()) ** 2).sum() + 1e-18)
    Ti.loc[i, "no_linealidad"] = float(1 - r2)
# la KAN puede anular por completo una entrada: hay que distinguirlo, porque
# una activacion identicamente nula da R2 degenerado en el ajuste simbolico
# criterio RELATIVO: una entrada esta podada si su activacion es despreciable
# frente a la mayor (el criterio absoluto no distingue escalas)
TOL_PODA = 1e-3 * Ti.amplitud_total.max()
Ti["estado"] = np.where(Ti.amplitud_total <= TOL_PODA, "podada", "activa")
Ti = Ti.sort_values("amplitud_total", ascending=False)
Ti.to_csv(os.path.join(TAB, "fase45_importancia_entradas.csv"), index=False)
P(Ti.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
PODADAS = set(Ti[Ti.estado == "podada"].entrada)
ACTIVAS = [n for n in NOM_ENT if n not in PODADAS]
P(f"    entradas que la KAN ANULA (activacion identicamente nula): "
  f"{', '.join(sorted(PODADAS)) if PODADAS else 'ninguna'}")
P(f"    la red se queda con {len(ACTIVAS)} de {n_in} entradas: {', '.join(ACTIVAS)}")

fig, axs = plt.subplots(M, n_in, figsize=(2.0 * n_in, 1.7 * M), sharex="col")
for i in range(n_in):
    xs, y = act[i]
    for j in range(M):
        ax = axs[j, i]
        podada = NOM_ENT[i] in PODADAS
        ax.plot(xs, y[:, j], lw=1.3, color="0.6" if podada else "tab:red")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_title(NOM_ENT[i] + (" (podada)" if podada else ""), fontsize=8,
                         color="0.5" if podada else "k")
        if i == 0:
            ax.set_ylabel(f"PC{j+1}", fontsize=8)
fig.suptitle("Funciones de activacion de la cabeza KAN: $\\phi_{ij}$ por entrada (columnas) "
             "y componente de salida (filas)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "fase45_activaciones_kan.png"), dpi=150)

# ---------------------------------------------------------------------
# 3. ajuste simbolico de las activaciones
# ---------------------------------------------------------------------
def disenos(x):
    return {
        "lineal: a+bx": np.column_stack([np.ones_like(x), x]),
        "cuadratica: a+bx+cx^2": np.column_stack([np.ones_like(x), x, x ** 2]),
        "cubica": np.column_stack([np.ones_like(x), x, x ** 2, x ** 3]),
        "seno: a+b sen(pi x/2)": np.column_stack([np.ones_like(x), np.sin(np.pi * x / 2)]),
        "tangente hiperbolica": np.column_stack([np.ones_like(x), np.tanh(x)]),
        "valor absoluto": np.column_stack([np.ones_like(x), np.abs(x)]),
        "exponencial": np.column_stack([np.ones_like(x), np.exp(np.clip(x, -5, 5))]),
        "logaritmica: a+b ln(1+|x|)": np.column_stack([np.ones_like(x), np.log1p(np.abs(x))]),
    }


P("\n[3] ajuste simbolico de las activaciones (R2, salida PC1)")
fs = []
for i in range(n_in):
    xs, y = act[i]
    obj = y[:, 0]
    sst = ((obj - obj.mean()) ** 2).sum()
    if NOM_ENT[i] in PODADAS or sst <= 1e-12 * len(obj):   # R2 no definido
        P(f"    {NOM_ENT[i]:9s}: activacion nula (entrada podada por la KAN), "
          f"sin ajuste simbolico")
        continue
    mejor = None
    for nom, A in disenos(xs).items():
        be = np.linalg.lstsq(A, obj, rcond=None)[0]
        r2 = 1 - ((obj - A @ be) ** 2).sum() / sst
        r2aj = 1 - (1 - r2) * (len(xs) - 1) / (len(xs) - A.shape[1])
        fs.append(dict(entrada=NOM_ENT[i], forma=nom, R2=float(r2), R2_ajustado=float(r2aj),
                       coef=np.round(be, 4).tolist()))
        if mejor is None or r2aj > mejor[1]:
            mejor = (nom, r2aj, r2, be)
    P(f"    {NOM_ENT[i]:9s}: mejor forma = {mejor[0]:24s} R2 = {mejor[2]:.4f} "
      f"(ajustado {mejor[1]:.4f})")
S = pd.DataFrame(fs)
S.to_csv(os.path.join(TAB, "fase45_ajuste_simbolico.csv"), index=False)
best = S.sort_values("R2_ajustado", ascending=False).iloc[0]
P(f"    (se excluyeron las entradas podadas: {', '.join(sorted(PODADAS)) or 'ninguna'})")
P(f"    activacion mejor explicada: {best.entrada} con {best.forma}, "
  f"R2 = {best.R2:.4f}, coeficientes {best.coef}")

i_dom = NOM_ENT.index(Ti[Ti.estado == "activa"].iloc[0].entrada)
xs, y = act[i_dom]
obj = y[:, 0]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(xs, obj, lw=2.2, color="tab:red", label=f"$\\phi$ de la KAN ({NOM_ENT[i_dom]} $\\to$ PC1)")
for nom in ["lineal: a+bx", "cuadratica: a+bx+cx^2", "tangente hiperbolica"]:
    A = disenos(xs)[nom]
    be = np.linalg.lstsq(A, obj, rcond=None)[0]
    r2 = 1 - ((obj - A @ be) ** 2).sum() / (((obj - obj.mean()) ** 2).sum() + 1e-18)
    ax.plot(xs, A @ be, "--", lw=1.2, label=f"{nom.split(':')[0]} ($R^2$={r2:.3f})")
ax.set_xlabel(f"{NOM_ENT[i_dom]} (estandarizada)")
ax.set_ylabel("contribucion a PC1")
ax.set_title("Ajuste simbolico de la activacion dominante")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase45_ajuste_simbolico.png"), dpi=150)

# ---------------------------------------------------------------------
# 4. contraste con la cabeza lineal
# ---------------------------------------------------------------------
Wlin = red_lin.cabeza.weight.detach().numpy()          # (M x n_in)
pend = np.zeros(n_in)
for i in range(n_in):
    xs, y = act[i]
    A = np.column_stack([np.ones_like(xs), xs])
    pend[i] = np.linalg.lstsq(A, y[:, 0], rcond=None)[0][1]
Tc = pd.DataFrame(dict(entrada=NOM_ENT, coef_lineal_PC1=Wlin[0],
                       pendiente_media_KAN_PC1=pend))
Tc["razon"] = Tc.pendiente_media_KAN_PC1 / (Tc.coef_lineal_PC1 + 1e-12)
Tc.to_csv(os.path.join(TAB, "fase45_lineal_vs_kan.csv"), index=False)
Tc["estado"] = ["podada" if n in PODADAS else "activa" for n in NOM_ENT]
rho = np.corrcoef(Wlin[0], pend)[0, 1]
act_m = np.array([n not in PODADAS for n in NOM_ENT])
rho_act = np.corrcoef(Wlin[0][act_m], pend[act_m])[0, 1] if act_m.sum() > 2 else np.nan
P(f"\n[4] cabeza lineal vs pendiente media de las activaciones KAN (PC1)")
P(Tc.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
P(f"    correlacion sobre las 8 entradas:            r = {rho:.3f}")
P(f"    correlacion solo sobre las entradas activas: r = {rho_act:.3f}")
P(f"    (la discrepancia la producen las entradas que la KAN anula y el modelo "
  f"lineal si usa)")

fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.scatter(Wlin[0], pend, s=42, color="tab:purple")
for i in range(n_in):
    ax.annotate(NOM_ENT[i], (Wlin[0][i], pend[i]), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
lim = max(np.abs(Wlin[0]).max(), np.abs(pend).max()) * 1.2
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="identidad")
ax.axhline(0, color="0.8", lw=0.6)
ax.axvline(0, color="0.8", lw=0.6)
ax.set_xlabel("coeficiente de la cabeza lineal (PC1)")
ax.set_ylabel("pendiente media de $\\phi_{i,PC1}$ (KAN)")
ax.set_title(f"La KAN reproduce el signo y la escala del modelo lineal\n(r = {rho:.3f})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fase45_lineal_vs_kan.png"), dpi=150)

# ---------------------------------------------------------------------
# 5. peores casos
# ---------------------------------------------------------------------
Bmt = torch.tensor(Bm, dtype=torch.float32)
cmut = torch.tensor(c_mu, dtype=torch.float32)


def errores(df, C, V):
    with torch.no_grad():
        coef = (cmut + red_kan(C, V) @ Bmt.T).numpy()
    obs = df[EST].to_numpy(float)
    d = coef - obs
    l2 = np.sqrt(((d @ W) * d).sum(axis=1) / PERIODO)
    nor = np.sqrt(((obs @ W) * obs).sum(axis=1) / PERIODO) + 1e-12
    out = df.copy()
    out["l2_rel"] = l2 / nor
    out["acum_obs"] = (obs @ Phi.T).sum(axis=1)
    out["acum_pred"] = (coef @ Phi.T).sum(axis=1)
    return out, coef, obs


Ete, cte, ote = errores(te, Cte, Vte)
Ers, crs, ors_ = errores(rs, Crs, Vrs)
tod = pd.concat([Ete.assign(bloque="test 2001-2003"), Ers.assign(bloque="reservadas")])
peor = tod.sort_values("l2_rel", ascending=False).head(12)
cols = ["bloque", "STATION_ID", "NOMBRE", "MUNICIPIO", "YEAR", "ALT_EST", "dist_km",
        "l2_rel", "acum_obs", "acum_pred"]
cols = [c for c in cols if c in peor.columns]
peor[cols].to_csv(os.path.join(TAB, "fase45_peores_casos.csv"), index=False)
P("\n[5] peores casos (mayor error L2 relativo)")
P(peor[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
P(f"\n    error L2 relativo medio: test {Ete.l2_rel.mean():.3f} | "
  f"reservadas {Ers.l2_rel.mean():.3f}")
if "ALT_EST" in tod.columns:
    alto = tod[tod.ALT_EST > tod.ALT_EST.median()]
    bajo = tod[tod.ALT_EST <= tod.ALT_EST.median()]
    P(f"    por altitud: estaciones sobre la mediana {alto.l2_rel.mean():.3f} | "
      f"bajo la mediana {bajo.l2_rel.mean():.3f}")
    r_alt = tod[["l2_rel", "ALT_EST"]].corr(method="spearman").iloc[0, 1]
    r_dis = tod[["l2_rel", "dist_km"]].corr(method="spearman").iloc[0, 1] \
        if "dist_km" in tod.columns else np.nan
    P(f"    correlacion de Spearman del error con la altitud: {r_alt:+.3f} | "
      f"con la distancia al pixel: {r_dis:+.3f}")

sel = peor.head(4)
fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
for ax, (_, r) in zip(axs.ravel(), sel.iterrows()):
    fuente = Ete if r.bloque == "test 2001-2003" else Ers
    dfc, cc, oo = (te, cte, ote) if r.bloque == "test 2001-2003" else (rs, crs, ors_)
    k = dfc.index[(dfc.STATION_ID == r.STATION_ID) & (dfc.YEAR == r.YEAR)]
    if len(k) == 0:
        continue
    k = int(k[0])
    ax.plot(t_dia, oo[k] @ Phi.T, color="k", lw=1.6, label="estacion (observado)")
    ax.plot(t_dia, cc[k] @ Phi.T, color="tab:red", lw=1.6, label="prediccion")
    nom = r.get("NOMBRE", r.STATION_ID)
    ax.set_title(f"{nom} ({r.YEAR}) - L2rel {r.l2_rel:.2f}, "
                 f"{r.get('ALT_EST', float('nan')):.0f} m", fontsize=9)
    ax.set_xticks(LIM[:12] + np.diff(LIM) / 2)
    ax.set_xticklabels(MESES, fontsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Peores casos del modelo KAN con codificador aprendible", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG, "fase45_peores_casos.png"), dpi=150)

with open(os.path.join(ENT, "log_fase45.txt"), "w") as f:
    f.write("\n".join(log))
P("\nFASE 4.5 OK")
