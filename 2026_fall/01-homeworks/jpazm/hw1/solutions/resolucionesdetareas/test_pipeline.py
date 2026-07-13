import numpy as np, pandas as pd, os
import coffee_solution as cs

rng = np.random.default_rng(7)

# ---- ground truth ----
B_TRUE = 0.04
A0, AX, AZ = 30.0, 0.5, 2.0          # a_t = A0 + AX*X + AZ*Z
LEADERS = cs.LEADERS
followers_pool = ["Peru","Honduras","Mexico","Ethiopia","Guatemala","India",
                  "Uganda","CostaRica","Kenya","Tanzania","ElSalvador","Nicaragua",
                  "IvoryCoast","Cameroon","PapuaNG","Ecuador"]
countries = LEADERS + followers_pool

# country base costs (leaders cheaper)
base_cost = {c:(3.0 if c in LEADERS else 6.0) + rng.uniform(0,2) for c in countries}

years = list(range(1990, 2007))
rows = []
for yr in years:
    X = 20 + 0.6*(yr-1990) + rng.normal(0,1)
    Z = 15 + rng.normal(0,2)
    fert = 10 + rng.normal(0,2)
    tmax = 30 + rng.normal(0,3)
    tmin = 15 + rng.normal(0,3)
    a_t = A0 + AX*X + AZ*Z            # NO demand shock -> exact round-trip test
    # costs move with fert/temp (valid supply instruments)
    mc = {c: base_cost[c] + 0.1*fert + 0.02*tmax - 0.01*tmin for c in countries}
    lead = LEADERS[:]; foll = followers_pool[:]
    eq = cs.solve_stackelberg(a_t, B_TRUE, mc, lead, foll)
    for c in countries:
        rows.append(dict(year=yr, country=c, price_y=eq["P"], qe_yc=eq["q"][c],
                         importGDP_y=X, teaPrices_y=Z, tmax_ye=tmax, tmin_ye=tmin,
                         farm_prices_yc=mc[c], fert_y=fert))
df = pd.DataFrame(rows)
df.to_csv("coffee_synth.csv", index=False)
print("synthetic panel:", df.shape, "years", years[0], "-", years[-1])

# ---- TEST 1: follower_block tau matches closed form 1/(1+N_F) for Omega=I ----
nF = len(followers_pool)
_,_,tau = cs.follower_block(100.0, B_TRUE, np.zeros(nF), np.eye(nF))
print(f"[T1] tau={tau:.5f}  closed-form 1/(1+N_F)={1/(1+nF):.5f}  ",
      "PASS" if abs(tau-1/(1+nF))<1e-9 else "FAIL")

# ---- TEST 2: recover costs then re-solve -> reproduce observed q (fitted mode) ----
cs.RECOVER_WITH = "fitted"
os.makedirs("out_test", exist_ok=True)
dfr, ann = cs.load_and_prepare("coffee_synth.csv")
demand, _ = cs.estimate_demand(ann, "out_test")
b_est = -demand["lin_IV2"]["aQ"]
print(f"[T2] b_true={B_TRUE}  b_est(IV2)={b_est:.5f}  ",
      "PASS" if abs(b_est-B_TRUE)/B_TRUE<0.02 else "FAIL")

mc_df, b, dem_int = cs.recover_costs(dfr, demand, "out_test")
# forward-solve baseline for one year, compare to observed q
yr = years[5]
myr = mc_df[mc_df["year"]==yr]
g = dfr[(dfr.year==yr)&(dfr.qty>0)]
a0,aX,aZ = dem_int
a_t = a0 + aX*g["X"].iloc[0] + aZ*g["Z"].iloc[0]
active = myr["country"].tolist()
lead=[c for c in LEADERS if c in active]; foll=[c for c in active if c not in lead]
mc_vec = myr.set_index("country")["mc_stack"].to_dict()
eq = cs.solve_stackelberg(a_t, b, mc_vec, lead, foll, nonneg=False)
obs = dfr[dfr.year==yr].set_index("country")["qty"].to_dict()
maxerr = max(abs(eq["q"][c]-obs[c]) for c in active)
print(f"[T2b] baseline reproduces observed q, max abs err={maxerr:.2e}  ",
      "PASS" if maxerr<1e-6 else "FAIL")

# ---- TEST 3: kappa=0.5 bloc stays interior & gain>0; kappa->1 misbehaves ----
def bloc_solve(kappa):
    OmF = np.eye(len(foll)); idx={c:i for i,c in enumerate(foll)}
    pres=[c for c in cs.BLOC if c in idx]
    for i in pres:
        for j in pres:
            if i!=j: OmF[idx[i],idx[j]]=kappa
    return cs.solve_stackelberg(a_t, b, mc_vec, lead, foll, OmegaF=OmF)

base = cs.solve_stackelberg(a_t, b, mc_vec, lead, foll)
for kap in [0.0, 0.5, 0.95, 1.0]:
    try:
        e = bloc_solve(kap)
        bloc_q = [e["q"][c] for c in cs.BLOC if c in active]
        gain = sum(e["profit"][c]-base["profit"][c] for c in cs.BLOC if c in active)
        print(f"[T3] kappa={kap:>4}: bloc q={np.round(bloc_q,3)}  gain={gain:+.4f}  "
              f"min bloc q={min(bloc_q):+.3f}")
    except Exception as ex:
        print(f"[T3] kappa={kap}: EXCEPTION {ex}")
