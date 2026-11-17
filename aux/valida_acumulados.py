import numpy as np, pandas as pd
B = "/srv/mark02/analitica-final"
K = 5; PER = 365.0
COLS = [f"c{i+1}" for i in range(K)]

def Phi(t, K=K):
    t = np.asarray(t, float); cols = [np.ones_like(t)]
    for j in range(1, (K-1)//2+1):
        w = 2*np.pi*j*t/PER; cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols[:K])

ce = pd.read_csv(f"{B}/entrega/coef_estaciones.csv")
t = np.arange(0.5, 365.0, 1.0)
X = ce[COLS].to_numpy() @ Phi(t).T
rec = pd.DataFrame(dict(STATION_ID=ce.STATION_ID, YEAR=ce.YEAR, acum_rec=(X**2).sum(axis=1)))

d = pd.read_csv(f"{B}/entrega/estaciones_depuradas.csv")
obs = d.groupby(["STATION_ID","YEAR"], as_index=False).agg(acum_obs=("VALUE","sum"), n=("VALUE","size"))
m = rec.merge(obs, on=["STATION_ID","YEAR"], how="left")
print("curvas:", len(m), "| con obs:", m.acum_obs.notna().sum())
m["err"] = m.acum_rec - m.acum_obs
print(m[["acum_obs","acum_rec"]].describe().loc[["mean","50%","min","max"]].to_string())
print("sesgo medio: %.1f mm (%.1f%%)" % (m.err.mean(), 100*m.err.mean()/m.acum_obs.mean()))
print("RMSE: %.1f mm | r Pearson: %.4f | r Spearman: %.4f" % (
    np.sqrt((m.err**2).mean()), m[["acum_obs","acum_rec"]].corr().iloc[0,1],
    m[["acum_obs","acum_rec"]].corr(method="spearman").iloc[0,1]))
e = m[m.STATION_ID==5300900102]
print("\nestacion 5300900102 (la mas atipica):")
print(e[["YEAR","acum_obs","acum_rec","n"]].head(12).to_string(index=False))
print("\ntop5 acumulado observado:")
print(m.nlargest(5,"acum_obs")[["STATION_ID","YEAR","acum_obs","acum_rec"]].to_string(index=False))
