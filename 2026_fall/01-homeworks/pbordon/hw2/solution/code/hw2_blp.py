"""
================================================================================
 PAUTA TAREA 2 - OI EMPIRICA  |  Parte 2: BLP (coeficientes aleatorios)
================================================================================
 Estima el modelo de demanda con coeficientes aleatorios (BLP) con pyblp.

 Modelo:  u_ijmt = x_jmt b + a_i p_jmt + b_iG generic_jmt + xi_jmt + eps_ijmt
   a_i   = a + a_I inc_i          (interaccion precio x ingreso, demografico)
   b_iG  = sigma_G nu_i,  nu_i ~ N(0,1)   (coef. aleatorio sobre generico)
   x     = prom + dummies de marca (brand_1..brand_11)
   generic = 1 si brand in {10,11}

 Mercado = farmacia(store) x semana(week).  Tamano de mercado = count.
 Instrumentos de demanda: costo mayorista (cost) + precios del mismo producto
   en 30 farmacias (pricestore1..30)  [a la Hausman].

 Draws no observados nu_i: v.csv columnas 1-20.   Ingreso inc_i: hhincome1-20.

 Salidas: bld/results_blp.json
================================================================================
"""
import json
import os
import numpy as np
import pandas as pd
import pyblp

pyblp.options.verbose = False
np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "raw") + os.sep
OUT = os.path.join(HERE, "..", "bld") + os.sep

# ------------------------------------------------------------------ 1. Datos de productos
d = pd.read_csv(RAW + "OTC_Data.csv").sort_values(["store", "week", "brand"]).reset_index(drop=True)
ins = pd.read_csv(RAW + "OTC_Instruments.csv").sort_values(["store", "week", "brand"]).reset_index(drop=True)

d["market_ids"] = d["store"] * 100 + d["week"]
d["shares"] = d["sales"] / d["count"]
d["prices"] = d["price"]
d["generic"] = (d["brand"] >= 10).astype(float)
for b in range(1, 12):
    d[f"brand_{b}"] = (d["brand"] == b).astype(float)

# Instrumentos de demanda: cost + 30 precios en otras farmacias (misma semana/producto)
pcols = [f"pricestore{i}" for i in range(1, 31)]
iv = ins[["store", "week", "brand", "cost"] + pcols].copy()
d = d.merge(iv, on=["store", "week", "brand"], how="left", suffixes=("", "_iv"))
inst = d[["cost"] + pcols].values
for j in range(inst.shape[1]):
    d[f"demand_instruments{j}"] = inst[:, j]

product_data = d

# ------------------------------------------------------------------ 2. Datos de agentes (20 por mercado)
dem = pd.read_csv(RAW + "OTC_Demographics.csv").sort_values(["store", "week"]).reset_index(drop=True)
v = pd.read_csv(RAW + "v.csv")  # 3504 mercados x 40 draws; usar 1-20
dem["market_ids"] = dem["store"] * 100 + dem["week"]
assert len(dem) == len(v), "demograficos y v deben tener una fila por mercado"

ns = 20
rows = []
inc = dem[[f"hhincome{i}" for i in range(1, ns + 1)]].values
nu = v[[f"var{i}" for i in range(1, ns + 1)]].values
mids = dem["market_ids"].values
for m in range(len(dem)):
    for a in range(ns):
        rows.append((mids[m], 1.0 / ns, inc[m, a], 0.0, nu[m, a]))
agent_data = pd.DataFrame(rows, columns=["market_ids", "weights", "income", "nodes0", "nodes1"])
# nodes0 <-> prices (sigma fija en 0); nodes1 <-> generic (coef. aleatorio nu)

# ------------------------------------------------------------------ 3. Formulaciones
X1 = pyblp.Formulation("0 + prices + prom + " + " + ".join(f"brand_{b}" for b in range(1, 12)))
X2 = pyblp.Formulation("0 + prices + generic")
agent_formulation = pyblp.Formulation("0 + income")
problem = pyblp.Problem((X1, X2), product_data, agent_formulation, agent_data)
print(problem)

# ------------------------------------------------------------------ 4. Resolver
# sigma: solo coef. aleatorio en 'generic' (los ceros quedan FIJOS en 0)
sigma0 = np.array([[0.0, 0.0],
                   [0.0, 0.5]])
# pi: solo interaccion 'prices' x income (la fila de 'generic' queda fija en 0)
pi0 = np.array([[0.1],
                [0.0]])

results = problem.solve(
    sigma=sigma0, pi=pi0,
    optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-5}),
    method="1s",
)
print(results)

# ------------------------------------------------------------------ 5. Elasticidades farmacia 109 semana 39
mid_109_39 = 109 * 100 + 39
elas = results.compute_elasticities(name="prices")  # apilado por mercado
md = product_data[["market_ids", "brand"]].reset_index(drop=True)
idx = md.index[md["market_ids"] == mid_109_39].tolist()
E = elas[idx][:, :len(idx)]  # bloque del mercado
brands_109 = md.loc[idx, "brand"].tolist()

# medias de elasticidad propia por marca (toda la muestra)
own = np.array(results.extract_diagonals(elas)).flatten()
product_data = product_data.assign(_own=own)
own_by_brand = product_data.groupby("brand")["_own"].mean().round(4).to_dict()

# ------------------------------------------------------------------ 6. Guardar
beta = {n: float(b) for n, b in zip(results.beta_labels, results.beta.flatten())}
out = {
    "beta": beta,
    "alpha_mean_price": float(results.beta[results.beta_labels.index("prices")]),
    "sigma_generic": float(results.sigma[1, 1]),
    "pi_price_income": float(results.pi[0, 0]),
    "objective": float(np.atleast_1d(results.objective)[0]),
    "elas_matrix_109_39": np.round(E, 4).tolist(),
    "brands_109_39": [int(b) for b in brands_109],
    "own_elas_by_brand": {int(k): float(v) for k, v in own_by_brand.items()},
    "mean_income": float(agent_data["income"].mean()),
}
out["price_mean_effect"] = out["alpha_mean_price"] + out["pi_price_income"] * out["mean_income"]
with open(OUT + "results_blp.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 60)
print("BLP (Python pyblp)  vs  pauta del profesor")
print("=" * 60)
print(f"  alpha (precio, medio lineal) = {out['alpha_mean_price']:+.4f}   pauta = -1.3925")
print(f"  alpha_I (precio x ingreso)   = {out['pi_price_income']:+.4f}   pauta = +0.0877")
print(f"  sigma_G (generico)           = {out['sigma_generic']:+.4f}   pauta = +0.7821")
print(f"  efecto medio precio          = {out['price_mean_effect']:+.4f}   pauta = -0.4604")
print(f"  objetivo GMM                 = {out['objective']:.3f}        pauta = 363.1")
print(f"  ingreso medio                = {out['mean_income']:.3f}")
print("\n  Elasticidades BLP propias (diag) farmacia 109/39:")
print("   ", np.round(np.diag(E), 4).tolist())
print("OK -> bld/results_blp.json")
