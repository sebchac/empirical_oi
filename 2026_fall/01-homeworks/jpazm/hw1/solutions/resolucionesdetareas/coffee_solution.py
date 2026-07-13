"""
Empirical IO -- Homework 3: Coffee Demand, Conduct, and Counterfactuals
Solution pipeline (Igami 2015 logic).

Sections:
  1. Data preparation
  2. Demand estimation (OLS + IV, linear & log), elasticities over time
  3. Marginal cost / markup recovery: Cournot vs Stackelberg
  4. Counterfactuals (CF1 uses kappa = 0.5 partial internalization, NOT a merger)

Dependencies: numpy, pandas, matplotlib (statsmodels optional, not required).

Run:
    python coffee_solution.py --data coffee.csv --out out/

The dataset is expected to be long (one row per country-year) with columns:
  year, country, price_y, qe_yc, importGDP_y, teaPrices_y,
  tmax_ye, tmin_ye, farm_prices_yc, fert_y
Adjust COLS below if your names differ.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
COLS = dict(
    year="year", country="country",
    price="price_y", qty="qe_yc",
    X="importGDP_y", Z="teaPrices_y",
    tmax="tmax_ye", tmin="tmin_ye",
    farm="farm_prices_yc", fert="fert_y",
)

LEADERS = ["Brazil", "Viet Nam", "Colombia", "Indonesia"]
BLOC     = ["Peru", "Honduras", "Mexico"]     # CF1 cooperative bloc (followers)
KAPPA    = 0.5                                # CF1 partial internalization
LEADER_COST_SHOCK = 0.05                      # CF2: +5% on leader mc
TEA_SHOCK         = -0.05                     # CF3: tea price -5%

# How to recover mc: 'observed' uses observed price P_t (baseline may differ from
# observed data, matching the HW's "baseline vs observed" framing);
# 'fitted' uses the fitted demand curve so the baseline reproduces observed q
# exactly (useful as an internal consistency check).
RECOVER_WITH = "fitted"

np.set_printoptions(suppress=True, precision=4)


# --------------------------------------------------------------------------- #
# Small linear-algebra / IV helpers
# --------------------------------------------------------------------------- #
def safe_inv(M):
    """Invert M; fall back to Moore-Penrose pseudoinverse if ill-conditioned.
    (Relevant at kappa->1 in the homogeneous good, where 11'+Omega degenerates.)"""
    try:
        if np.linalg.cond(M) < 1e12:
            return np.linalg.solve(M, np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(M)


def ols(y, X):
    """Return beta, residuals for y = X beta."""
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return beta, y - X @ beta


def tsls(y, Xexo, endog, Z):
    """
    2SLS. Xexo: exogenous regressors incl. constant (n x k1).
    endog: endogenous regressors (n x ke). Z: excluded instruments (n x kz).
    Returns full beta ordered [Xexo | endog].
    """
    Xexo  = np.atleast_2d(Xexo);   endog = np.atleast_2d(endog); Z = np.atleast_2d(Z)
    if Xexo.shape[0] != len(y):  Xexo  = Xexo.T
    if endog.shape[0] != len(y): endog = endog.T
    if Z.shape[0] != len(y):     Z = Z.T
    W  = np.hstack([Xexo, Z])                 # full instrument set
    # first stage: project endog on W
    Pw = W @ np.linalg.solve(W.T @ W, W.T)
    endog_hat = Pw @ endog
    Xfull = np.hstack([Xexo, endog])
    Xhat  = np.hstack([Xexo, endog_hat])
    beta  = np.linalg.solve(Xhat.T @ Xhat, Xhat.T @ y)
    resid = y - Xfull @ beta
    return beta, resid


# --------------------------------------------------------------------------- #
# 1. DATA PREPARATION
# --------------------------------------------------------------------------- #
def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in COLS.items()})
    # annual world aggregate exports Q_t
    Q = df.groupby("year")["qty"].sum().rename("Q")
    # annual (year-level) demand frame: price, X, Z, instruments are year-level
    ann = (df.groupby("year")
             .agg(price=("price", "first"),
                  X=("X", "first"),
                  Z=("Z", "first"),
                  tmax=("tmax", "first"),
                  tmin=("tmin", "first"),
                  fert=("fert", "first"))
             .join(Q)
             .reset_index()
             .sort_values("year"))
    return df, ann


# --------------------------------------------------------------------------- #
# 2. DEMAND ESTIMATION
# --------------------------------------------------------------------------- #
def estimate_demand(ann, out):
    y  = ann["price"].values
    Q  = ann["Q"].values
    X  = ann["X"].values
    Z  = ann["Z"].values
    n  = len(y)
    c1 = np.ones(n)

    Xexo = np.column_stack([c1, X, Z])           # [const, X, Z]
    instr_full = np.column_stack([ann["fert"].values, ann["tmax"].values, ann["tmin"].values])
    instr_temp = np.column_stack([ann["tmax"].values, ann["tmin"].values])

    results = {}
    # ---- linear inverse demand: P = a0 + aQ Q + aX X + aZ Z ----
    b_ols, _ = ols(y, np.column_stack([c1, Q, X, Z]))
    results["lin_OLS"]  = dict(a0=b_ols[0], aQ=b_ols[1], aX=b_ols[2], aZ=b_ols[3])
    b1, _ = tsls(y, Xexo, Q, instr_full)         # spec 1: fert + temp
    results["lin_IV1"]  = dict(a0=b1[0], aX=b1[1], aZ=b1[2], aQ=b1[3])
    b2, _ = tsls(y, Xexo, Q, instr_temp)         # spec 2: temp only
    results["lin_IV2"]  = dict(a0=b2[0], aX=b2[1], aZ=b2[2], aQ=b2[3])

    # ---- log-linear inverse demand ----
    ly, lQ, lX, lZ = np.log(y), np.log(Q), np.log(X), np.log(Z)
    lc = np.ones(n)
    lXexo = np.column_stack([lc, lX, lZ])
    lfert = np.log(ann["fert"].values)
    linstr_full = np.column_stack([lfert, ann["tmax"].values, ann["tmin"].values])
    linstr_temp = np.column_stack([ann["tmax"].values, ann["tmin"].values])
    lb_ols, _ = ols(ly, np.column_stack([lc, lQ, lX, lZ]))
    results["log_OLS"] = dict(b0=lb_ols[0], bQ=lb_ols[1], bX=lb_ols[2], bZ=lb_ols[3])
    lb1, _ = tsls(ly, lXexo, lQ, linstr_full)
    results["log_IV1"] = dict(b0=lb1[0], bX=lb1[1], bZ=lb1[2], bQ=lb1[3])
    lb2, _ = tsls(ly, lXexo, lQ, linstr_temp)
    results["log_IV2"] = dict(b0=lb2[0], bX=lb2[1], bZ=lb2[2], bQ=lb2[3])

    # ---- table ----
    tab = pd.DataFrame({
        "spec": ["lin_OLS", "lin_IV1", "lin_IV2", "log_OLS", "log_IV1", "log_IV2"],
        "coef_Q": [results["lin_OLS"]["aQ"], results["lin_IV1"]["aQ"], results["lin_IV2"]["aQ"],
                   results["log_OLS"]["bQ"], results["log_IV1"]["bQ"], results["log_IV2"]["bQ"]],
        "coef_X": [results["lin_OLS"]["aX"], results["lin_IV1"]["aX"], results["lin_IV2"]["aX"],
                   results["log_OLS"]["bX"], results["log_IV1"]["bX"], results["log_IV2"]["bX"]],
        "coef_Z": [results["lin_OLS"]["aZ"], results["lin_IV1"]["aZ"], results["lin_IV2"]["aZ"],
                   results["log_OLS"]["bZ"], results["log_IV1"]["bZ"], results["log_IV2"]["bZ"]],
    })
    tab.to_csv(os.path.join(out, "demand_estimates.csv"), index=False)

    # ---- elasticities over time ----
    # linear: eps_t = (1/aQ)*(P_t/Q_t);  log: eps = 1/bQ (constant)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for spec in ["lin_OLS", "lin_IV1", "lin_IV2"]:
        aQ = results[spec]["aQ"]
        eps = (1.0 / aQ) * (y / Q)
        ax.plot(ann["year"], eps, label=spec)
    for spec in ["log_IV2"]:
        ax.axhline(1.0 / results[spec]["bQ"], ls="--", color="k",
                   label=f"{spec} (const={1.0/results[spec]['bQ']:.2f})")
    ax.set_xlabel("year"); ax.set_ylabel("demand elasticity")
    ax.set_title("Implied price elasticity of demand over time")
    ax.legend(fontsize=8); ax.axhline(-1, color="grey", lw=0.6, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(out, "elasticities.png"), dpi=130)
    plt.close(fig)

    return results, tab


# --------------------------------------------------------------------------- #
# Stackelberg machinery (linear inverse demand P = a - b Q)
# --------------------------------------------------------------------------- #
def follower_block(a, b, cF, OmegaF):
    """
    Given leaders' aggregate QL as a free scalar, follower quantities solve
        b (11' + OmegaF) qF = (a - b QL) 1 - cF.
    Return affine map qF(QL) = A - B*QL and tau = 1 - 1'B (total pass-through of
    leader output into Q after followers react).
    """
    nF = len(cF)
    ones = np.ones(nF)
    M = np.outer(ones, ones) + OmegaF
    Minv = safe_inv(M)
    A = (1.0 / b) * (Minv @ (a * ones - cF))
    B = Minv @ ones
    tau = 1.0 - ones @ B
    return A, B, tau


def _solve_linear(a, b, mc, leaders, followers, OmegaF):
    """
    One linear Stackelberg solve for FIXED active sets (Omega^L = I).
    Handles the degenerate cases nL == 0 (pure follower Cournot) and
    nF == 0 (pure leader Cournot, tau = 1). Quantities may be negative.
    """
    nF, nL = len(followers), len(leaders)
    cF = np.array([mc[c] for c in followers], float)
    cL = np.array([mc[c] for c in leaders], float)

    if nF > 0:
        A, B, tau = follower_block(a, b, cF, OmegaF)
        sumA = A.sum()
    else:                                     # no followers left
        A = np.zeros(0); B = np.zeros(0); tau = 1.0; sumA = 0.0

    if nL > 0:
        onesL = np.ones(nL)
        KL = np.outer(onesL, onesL) + np.eye(nL)
        rhs = (a - b * sumA) * onesL - cL
        qL = (1.0 / (b * tau)) * np.linalg.solve(KL, rhs)
        QL = qL.sum()
    else:                                     # no leaders left
        qL = np.zeros(0); QL = 0.0

    qF = A - B * QL if nF > 0 else np.zeros(0)
    Q = QL + qF.sum()
    P = a - b * Q

    q = {c: qL[i] for i, c in enumerate(leaders)}
    q.update({c: qF[i] for i, c in enumerate(followers)})
    profit = {c: (P - mc[c]) * q[c] for c in q}
    CS = 0.5 * b * Q ** 2
    TS = CS + sum(profit.values())
    return dict(q=q, Q=Q, P=P, tau=tau, profit=profit, CS=CS, TS=TS)


def solve_stackelberg(a, b, mc, leaders, followers, bloc=(), kappa=1.0,
                      nonneg=True, tol=1e-7):
    """
    Stackelberg equilibrium with the non-negativity (shut-down) rule.

    If nonneg=True: solve the FOCs; any country whose equilibrium quantity is
    negative is shut down (q := 0, removed from the game) and the reduced game is
    re-solved. Iterate until every active country produces q >= 0. Because
    removing a producer raises the price and weakly raises everyone else's output,
    this drops the most-negative country each round and terminates in <= N steps.

    Omega^F is rebuilt for the *surviving* followers each round, so bloc
    coordination (bloc, kappa) is applied only among bloc members still active.

    If nonneg=False: a single linear solve, quantities may be negative (used only
    for the internal round-trip consistency check).
    """
    leaders = list(leaders); followers = list(followers)
    dropped = []
    aL, aF = leaders[:], followers[:]
    while True:
        OmF = build_OmegaF(aF, bloc, kappa)
        eq = _solve_linear(a, b, mc, aL, aF, OmF)
        if not nonneg:
            break
        worst = min(eq["q"], key=eq["q"].get) if eq["q"] else None
        if worst is not None and eq["q"][worst] < -tol:
            dropped.append(worst)
            if worst in aL:
                aL.remove(worst)
            else:
                aF.remove(worst)
            if not aL and not aF:             # everyone shut down (pathological)
                break
            continue
        break

    # reattach shut-down countries at q = 0, profit = 0
    for c in dropped:
        eq["q"][c] = 0.0
        eq["profit"][c] = 0.0
    eq["leaders"] = leaders
    eq["followers"] = followers
    eq["active_leaders"] = aL
    eq["active_followers"] = aF
    eq["shutdown"] = dropped
    return eq


# --------------------------------------------------------------------------- #
# 3. MC / MARKUP RECOVERY
# --------------------------------------------------------------------------- #
def recover_costs(df, demand, out):
    d = demand["lin_IV2"]                       # spec 2 (temp-only IV), linear
    a0, aQ, aX, aZ = d["a0"], d["aQ"], d["aX"], d["aZ"]
    b = -aQ                                     # P = a_t - b Q, b > 0

    rows = []
    for yr, g in df.groupby("year"):
        g = g[g["qty"] > 0]
        if len(g) == 0:
            continue
        Xt, Zt = g["X"].iloc[0], g["Z"].iloc[0]
        a_t = a0 + aX * Xt + aZ * Zt
        Q_t = g["qty"].sum()
        P_obs = g["price"].iloc[0]
        P_use = P_obs if RECOVER_WITH == "observed" else (a_t - b * Q_t)

        active = g["country"].tolist()
        lead = [c for c in LEADERS if c in active]
        foll = [c for c in active if c not in lead]
        # baseline follower reaction (Omega^F = I) -> tau for the leaders
        cF_dummy = np.zeros(len(foll))
        _, _, tau = follower_block(a_t, b, cF_dummy, np.eye(len(foll)))

        for _, r in g.iterrows():
            ctry, q_i = r["country"], r["qty"]
            mc_cournot = P_use - b * q_i
            if ctry in lead:
                mc_stack = P_use - b * q_i * tau
            else:
                mc_stack = P_use - b * q_i        # followers: same as Cournot
            rows.append(dict(year=yr, country=ctry, q=q_i, P=P_use, a_t=a_t,
                             is_leader=ctry in lead, tau=tau,
                             mc_cournot=mc_cournot, mc_stack=mc_stack,
                             markup_cournot=P_use - mc_cournot,
                             markup_stack=P_use - mc_stack))
    mc = pd.DataFrame(rows)
    mc.to_csv(os.path.join(out, "marginal_costs.csv"), index=False)

    # figures: one MC and one markup panel per leader
    for L in LEADERS:
        sub = mc[mc["country"] == L].sort_values("year")
        if sub.empty:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(sub["year"], sub["mc_cournot"], label="Cournot")
        ax[0].plot(sub["year"], sub["mc_stack"], label="Stackelberg")
        ax[0].set_title(f"{L}: recovered MC"); ax[0].legend(fontsize=8)
        ax[1].plot(sub["year"], sub["markup_cournot"], label="Cournot")
        ax[1].plot(sub["year"], sub["markup_stack"], label="Stackelberg")
        ax[1].set_title(f"{L}: markup P - mc"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"mc_markup_{L}.png"), dpi=120)
        plt.close(fig)
    return mc, b, (a0, aX, aZ)


# --------------------------------------------------------------------------- #
# 4. COUNTERFACTUALS
# --------------------------------------------------------------------------- #
def floor_costs(mc_year):
    """Section 4 rule: negative recovered mc -> p25 of non-negative mc that year."""
    m = mc_year.copy()
    neg = m < 0
    if neg.any():
        pos = m[~neg]
        fill = np.percentile(pos, 25) if len(pos) else 0.0
        m[neg] = fill
    return m


def build_OmegaF(followers, bloc, kappa):
    nF = len(followers)
    O = np.eye(nF)
    idx = {c: i for i, c in enumerate(followers)}
    present = [c for c in bloc if c in idx]
    for i in present:
        for j in present:
            if i != j:
                O[idx[i], idx[j]] = kappa
    return O


def summarize(eq):
    lead_prof = sum(eq["profit"][c] for c in eq["leaders"])
    foll_prof = sum(eq["profit"][c] for c in eq["followers"])
    n_shut = len(eq.get("shutdown", []))
    return dict(P=eq["P"], Q=eq["Q"], CS=eq["CS"],
                lead_profit=lead_prof, foll_profit=foll_prof,
                TS=eq["TS"], n_shutdown=n_shut)


def run_counterfactuals(mc, b, dem_intercept, df, out):
    a0, aX, aZ = dem_intercept
    years = sorted(mc["year"].unique())
    base_rows, cf_summ = [], {k: [] for k in ["CF1", "CF2", "CF3", "CF4"]}
    prof_changes = {}          # country -> list of pct changes per CF (avg over years)

    for yr in years:
        myr = mc[mc["year"] == yr].copy()
        g = df[(df["year"] == yr) & (df["qty"] > 0)]
        Xt, Zt = g["X"].iloc[0], g["Z"].iloc[0]
        a_t = a0 + aX * Xt + aZ * Zt

        active = myr["country"].tolist()
        lead = [c for c in LEADERS if c in active]
        foll = [c for c in active if c not in lead]

        # floored Stackelberg marginal costs (recovered under Stackelberg)
        mc_vec = floor_costs(myr.set_index("country")["mc_stack"]).to_dict()

        # baseline Stackelberg equilibrium (non-negativity / shut-down rule on)
        base = solve_stackelberg(a_t, b, mc_vec, lead, foll)
        base_rows.append(dict(year=yr, **summarize(base)))

        # ---- CF1: partial cooperation (kappa=0.5) among Peru/Honduras/Mexico ----
        cf1 = solve_stackelberg(a_t, b, mc_vec, lead, foll, bloc=BLOC, kappa=KAPPA)
        cf_summ["CF1"].append(dict(year=yr, **summarize(cf1)))

        # ---- CF2: +5% leader marginal cost ----
        mc2 = dict(mc_vec); [mc2.__setitem__(c, mc_vec[c] * (1 + LEADER_COST_SHOCK)) for c in lead]
        cf2 = solve_stackelberg(a_t, b, mc2, lead, foll)
        cf_summ["CF2"].append(dict(year=yr, **summarize(cf2)))

        # ---- CF3: tea price -5% -> shifts demand intercept ----
        a_t3 = a0 + aX * Xt + aZ * (Zt * (1 + TEA_SHOCK))
        cf3 = solve_stackelberg(a_t3, b, mc_vec, lead, foll)
        cf_summ["CF3"].append(dict(year=yr, **summarize(cf3)))

        # ---- CF4: Indonesia leader -> follower ----
        if "Indonesia" in lead:
            lead4 = [c for c in lead if c != "Indonesia"]
            foll4 = foll + ["Indonesia"]
        else:
            lead4, foll4 = lead, foll
        cf4 = solve_stackelberg(a_t, b, mc_vec, lead4, foll4)
        cf_summ["CF4"].append(dict(year=yr, **summarize(cf4)))

        # per-country profit % changes vs baseline (store for the cross-CF figure)
        for c in base["profit"]:
            base_p = base["profit"][c]
            for name, eq in [("CF1", cf1), ("CF2", cf2), ("CF3", cf3), ("CF4", cf4)]:
                pc = (eq["profit"].get(c, np.nan) - base_p) / base_p * 100 if base_p != 0 else np.nan
                prof_changes.setdefault(c, {}).setdefault(name, []).append(pc)

        # CF1 bloc object: gain and per-country pct
        if all(c in base["profit"] for c in BLOC if c in active):
            bloc_present = [c for c in BLOC if c in active]
            gain = sum(cf1["profit"][c] - base["profit"][c] for c in bloc_present)
            myr_cf1 = {c: dict(base=base["profit"][c], coop=cf1["profit"][c],
                               d=cf1["profit"][c] - base["profit"][c],
                               pct=(cf1["profit"][c] - base["profit"][c]) / base["profit"][c] * 100
                               if base["profit"][c] != 0 else np.nan)
                       for c in bloc_present}
            prof_changes.setdefault("_CF1_bloc", {})[yr] = dict(gain=gain, detail=myr_cf1)

        # CF4 Indonesia object
        if "Indonesia" in base["profit"] and "Indonesia" in cf4["profit"]:
            prof_changes.setdefault("_CF4_indonesia", {})[yr] = dict(
                base=base["profit"]["Indonesia"], follower=cf4["profit"]["Indonesia"],
                d=cf4["profit"]["Indonesia"] - base["profit"]["Indonesia"])

    base_df = pd.DataFrame(base_rows)
    # comparison tables (year-averaged) for each CF
    def avg_table(name):
        b_ = base_df.mean(numeric_only=True)
        c_ = pd.DataFrame(cf_summ[name]).mean(numeric_only=True)
        t = pd.DataFrame({"baseline": b_, name: c_})
        t["delta"] = t[name] - t["baseline"]
        return t
    tables = {name: avg_table(name) for name in ["CF1", "CF2", "CF3", "CF4"]}
    for name, t in tables.items():
        t.to_csv(os.path.join(out, f"table_{name}.csv"))

    # cross-CF profit-change figure (mean pct change per country)
    countries = [c for c in prof_changes if not c.startswith("_")]
    order = sorted(countries)
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.25), 4.5))
    width = 0.2
    xs = np.arange(len(order))
    for k, name in enumerate(["CF1", "CF2", "CF3", "CF4"]):
        vals = [np.nanmean(prof_changes[c].get(name, [np.nan])) for c in order]
        ax.bar(xs + (k - 1.5) * width, vals, width, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("mean % change in profit vs baseline")
    ax.set_title("Country-level profit changes across counterfactuals")
    ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "profit_changes.png"), dpi=130)
    plt.close(fig)

    return base_df, tables, cf_summ, prof_changes


# --------------------------------------------------------------------------- #
def main(data, out):
    os.makedirs(out, exist_ok=True)
    df, ann = load_and_prepare(data)
    demand, dtab = estimate_demand(ann, out)
    print("Demand (linear IV2 = temp-only):", demand["lin_IV2"])
    mc, b, dem_int = recover_costs(df, demand, out)
    print(f"b = -aQ = {b:.4f}")
    base_df, tables, cf_summ, extras = run_counterfactuals(mc, b, dem_int, df, out)
    for name, t in tables.items():
        print(f"\n=== {name} (year-averaged) ===")
        print(t.round(3))
    # CF1 bloc summary
    if "_CF1_bloc" in extras:
        gains = [v["gain"] for v in extras["_CF1_bloc"].values()]
        print(f"\nCF1 (kappa={KAPPA}) mean bloc gain: {np.mean(gains):.4f}")
    return df, ann, demand, mc, base_df, tables, cf_summ, extras


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coffee.csv")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    main(a.data, a.out)
