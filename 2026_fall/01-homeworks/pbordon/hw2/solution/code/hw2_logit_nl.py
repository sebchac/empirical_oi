"""
================================================================================
 PAUTA TAREA 2 - OI EMPIRICA  |  Parte 1: Logit y Nested Logit
================================================================================
 Reproduce los resultados de demanda Logit (MCO e IV) y Nested Logit del
 hw2-solution.do (Stata) usando Python (pandas + statsmodels + linearmodels).

 Mercado relevante: farmacia (store) x semana (week).  Tamano de mercado = count.
 Cuotas: s_j = sales/count ; outside good s_0 = 1 - sum_j s_j.
 Variable dependiente Logit (Berry): y = ln(s_j) - ln(s_0).

 Salidas: bld/results_logit_nl.json  (coeficientes, elasticidades, matrices)
================================================================================
"""
import json
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "raw") + os.sep   # hw2/raw/
OUT = os.path.join(HERE, "..", "bld") + os.sep          # hw2/solution/bld/

# ------------------------------------------------------------------ 1. Datos
d = pd.read_csv(RAW + "OTC_Data.csv").sort_values(["store", "week", "brand"]).reset_index(drop=True)

# Cuotas de mercado
d["share"] = d["sales"] / d["count"]
d["inshare"] = d.groupby(["store", "week"])["share"].transform("sum")
d["outshare"] = 1 - d["inshare"]
d["y"] = np.log(d["share"]) - np.log(d["outshare"])

# Instrumento Hausman: media leave-one-out del precio de la misma marca,
# misma semana, en las otras farmacias (replica el .do).
g = d.groupby(["week", "brand"])["price"]
d["totprice"] = g.transform("sum")
d["nmarket"] = g.transform("count")
d["ivhausman"] = (d["totprice"] - d["price"]) / (d["nmarket"] - 1)

# Segmentos por marca (Nested Logit)
seg = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4}
d["segment"] = d["brand"].map(seg)
d["share_segm"] = d.groupby(["store", "week", "segment"])["share"].transform("sum")
d["sjg"] = d["share"] / d["share_segm"]
d["l_sjg"] = np.log(d["share"]) - np.log(d["share_segm"])
d["n_segm"] = d.groupby(["store", "week", "segment"])["share"].transform("count")

results = {"logit": {}, "elas_mean": {}, "elas_matrix": {}, "nested": {}}


def add_fe(df, cols, fe):
    """Demean cols within FE groups (FWL) -> abruma FE de brand#store."""
    out = df[cols].copy()
    grp = df.groupby(fe)
    for c in cols:
        out[c] = df[c] - grp[c].transform("mean")
    return out


# ------------------------------------------------------------------ 2. Logit MCO
# Modelo 1: prom
m1 = smf.ols("y ~ price + prom", d).fit(cov_type="cluster", cov_kwds={"groups": d["store"]})
# Modelo 2: prom + dummies de marca
m2 = smf.ols("y ~ price + prom + C(brand)", d).fit(cov_type="cluster", cov_kwds={"groups": d["store"]})
# Modelo 3: prom + dummies brand#store (FE absorbidos por demeaning)
dd = add_fe(d, ["y", "price", "prom"], ["brand", "store"])
m3 = smf.ols("y ~ price + prom - 1", dd).fit(cov_type="cluster", cov_kwds={"groups": d["store"]})

for name, m in [("ols1", m1), ("ols2", m2), ("ols3", m3)]:
    results["logit"][name] = {
        "price": float(m.params["price"]), "price_se": float(m.bse["price"]),
        "prom": float(m.params["prom"]), "prom_se": float(m.bse["prom"]),
    }

# ------------------------------------------------------------------ 3. Logit IV
def iv(df, instrument, brand_fe=False, brandstore_fe=False):
    data = df.copy()
    if brandstore_fe:
        cols = ["y", "price", "prom", instrument]
        dm = add_fe(data, cols, ["brand", "store"])
        dep, exog, endog, instr = dm["y"], dm[["prom"]], dm[["price"]], dm[[instrument]]
        return IV2SLS(dep, exog, endog, instr).fit(cov_type="clustered", clusters=data["store"])
    exog_vars = ["prom"]
    X = data[["prom"]].copy()
    X["const"] = 1.0
    if brand_fe:
        bd = pd.get_dummies(data["brand"], prefix="b", drop_first=True).astype(float)
        X = pd.concat([X, bd], axis=1)
    return IV2SLS(data["y"], X, data[["price"]], data[[instrument]]).fit(
        cov_type="clustered", clusters=data["store"])


iv_specs = [
    ("ivc1", "cost", False, False), ("ivc2", "cost", True, False), ("ivc3", "cost", False, True),
    ("ivh1", "ivhausman", False, False), ("ivh2", "ivhausman", True, False), ("ivh3", "ivhausman", False, True),
]
for name, instr, bfe, bsfe in iv_specs:
    r = iv(d, instr, bfe, bsfe)
    results["logit"][name] = {
        "price": float(r.params["price"]), "price_se": float(r.std_errors["price"]),
        "prom": float(r.params["prom"]), "prom_se": float(r.std_errors["prom"]),
    }

# ------------------------------------------------------------------ 4. Elasticidades medias (9 modelos)
# Elasticidad propia Logit por obs: eta_jj = beta_price * price * (1 - share)
order = ["ols1", "ols2", "ols3", "ivc1", "ivc2", "ivc3", "ivh1", "ivh2", "ivh3"]
elas_by_brand = {}
for name in order:
    b = results["logit"][name]["price"]
    d[f"el_{name}"] = b * d["price"] * (1 - d["share"])
    elas_by_brand[name] = d.groupby("brand")[f"el_{name}"].mean().round(4).to_dict()
    results["elas_mean"][name] = {"by_brand": {int(k): float(v) for k, v in elas_by_brand[name].items()},
                                  "total": float(d[f"el_{name}"].mean())}

# ------------------------------------------------------------------ 5. Matriz elasticidades 109/39 (Logit)
# Logit: own eta_jj = a*p_j*(1-s_j) ; cross eta_jk = -a*p_k*s_k
mkt = d[(d.store == 109) & (d.week == 39)].sort_values("brand").reset_index(drop=True)
p = mkt["price"].values; s = mkt["share"].values; J = len(mkt)
for name in ["ivh1", "ivh2"]:
    a = results["logit"][name]["price"]
    E = np.empty((J, J))
    for j in range(J):
        for k in range(J):
            E[j, k] = a * p[j] * (1 - s[j]) if j == k else -a * p[k] * s[k]
    results["elas_matrix"][f"logit_{name}"] = E.round(4).tolist()

# ------------------------------------------------------------------ 6. Nested Logit
nl_ols = smf.ols("y ~ price + l_sjg + prom", d).fit(cov_type="cluster", cov_kwds={"groups": d["store"]})
# IV: solo price instrumentado con ivhausman y n_segm ; l_sjg y prom exogenos (replica el .do)
Xn = d[["l_sjg", "prom"]].copy(); Xn["const"] = 1.0
nl_iv = IV2SLS(d["y"], Xn, d[["price"]], d[["ivhausman", "n_segm"]]).fit(
    cov_type="clustered", clusters=d["store"])
results["nested"]["ols"] = {k: float(nl_ols.params[k]) for k in ["price", "l_sjg", "prom", "Intercept"]}
results["nested"]["ols_se"] = {k: float(nl_ols.bse[k]) for k in ["price", "l_sjg", "prom"]}
results["nested"]["iv"] = {k: float(nl_iv.params[k]) for k in ["price", "l_sjg", "prom", "const"]}
results["nested"]["iv_se"] = {k: float(nl_iv.std_errors[k]) for k in ["price", "l_sjg", "prom"]}

# Matriz elasticidades NL 109/39 con estimacion IV
a = results["nested"]["iv"]["price"]; sig = results["nested"]["iv"]["l_sjg"]
sjg = mkt["sjg"].values; segm = mkt["segment"].values
Enl = np.empty((J, J))
for j in range(J):
    for k in range(J):
        if j == k:
            Enl[j, k] = a * p[j] * (1.0 / (1 - sig) - sig / (1 - sig) * sjg[j] - s[j])
        elif segm[j] == segm[k]:
            Enl[j, k] = -a * p[k] * (sig / (1 - sig) * sjg[k] + s[k])
        else:
            Enl[j, k] = -a * p[k] * s[k]
results["elas_matrix"]["nested_iv"] = Enl.round(4).tolist()
results["mkt_109_39"] = {"brand": mkt["brand"].tolist(), "price": p.round(3).tolist(),
                         "share": s.round(6).tolist()}

# ------------------------------------------------------------------ guardar
with open(OUT + "results_logit_nl.json", "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------------ validacion vs pauta
print("=" * 70)
print("LOGIT - coeficiente de PRECIO (Python vs pauta Stata)")
print("=" * 70)
pauta = {"ols1": -0.0514, "ols2": -0.3413, "ols3": -0.3302,
         "ivc1": -0.0106, "ivc2": -0.0080, "ivc3": -0.0346,
         "ivh1": -0.0511, "ivh2": -0.5468, "ivh3": -0.5480}
for k in order:
    print(f"  {k:5s}  py={results['logit'][k]['price']:+.4f}   pauta={pauta[k]:+.4f}")
print("\nLOGIT - coeficiente de PROM (modelo 1):",
      round(results["logit"]["ols1"]["prom"], 4), " pauta=0.2132")
print("\nELASTICIDAD media TOTAL por modelo (py vs pauta):")
pe = {"ols1": -0.2252, "ols2": -1.497, "ols3": -1.448, "ivc1": -0.0463, "ivc2": -0.0352,
      "ivc3": -0.1515, "ivh1": -0.2240, "ivh2": -2.398, "ivh3": -2.403}
for k in order:
    print(f"  {k:5s}  py={results['elas_mean'][k]['total']:+.4f}   pauta={pe[k]:+.4f}")
print("\nNESTED LOGIT (py vs pauta):")
print(f"  OLS: price={results['nested']['ols']['price']:+.4f} (p=0.1103) "
      f"l_sjg={results['nested']['ols']['l_sjg']:.4f} (p=1.1001) "
      f"prom={results['nested']['ols']['prom']:+.4f} (p=-0.0079)")
print(f"  IV : price={results['nested']['iv']['price']:+.4f} (p=0.1134) "
      f"l_sjg={results['nested']['iv']['l_sjg']:.4f} (p=1.1032) "
      f"prom={results['nested']['iv']['prom']:+.4f} (p=-0.0071)")
print("\nMatriz Logit IV (ivh1) diagonal 109/39:",
      [round(results['elas_matrix']['logit_ivh1'][i][i], 2) for i in range(J)])
print("Pauta elas_logit_iv diag: [-0.24,-0.32,-0.16,-0.14,-0.27,-0.42,-0.17,-0.20,-0.13,-0.08,-0.22]")
print("OK -> bld/results_logit_nl.json")
