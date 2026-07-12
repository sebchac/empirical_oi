# -*- coding: utf-8 -*-
"""
Motor de la fusion (Seccion 2) para el problema de aerolineas
=============================================================
Extiende el motor de equilibrio de 'ta_session_costos_fijos.py' con las tres
cosas que ese loop NO hacia:

  (1) PROPIEDAD PARCIAL  Omega(kappa): AA y BB internalizan una fraccion kappa
      de las utilidades del otro al fijar tarifas (kappa in [0,1]).
  (2) UTILIDAD CONJUNTA   VP_M = VP_AA + VP_BB de la entidad fusionada.
  (3) EXCEDENTE DEL CONSUMIDOR para cualquier conjunto de vuelos.

Reutiliza tal cual la maquinaria que ya funcionaba: re-resuelve el equilibrio de
Bertrand para un conjunto ARBITRARIO de vuelos (permite eliminar uno) y promedia
sobre re-sorteos de (xi, omega) con numeros aleatorios comunes (timing Eizenberg).

API para el alumno (todo lo demas es interno):
  ctx = setup()
  dvp_pre  = delta_vp(ctx, market, flight, kappa=0.0,  mode='own',   n_draws=16)
  dvp_post = delta_vp(ctx, market, flight, kappa=0.5,  mode='joint', n_draws=16)
  cs       = consumer_surplus(ctx, market, keep_flights, kappa=0.5)
"""
import numpy as np, pandas as pd, pyblp, pickle
pyblp.options.verbose = False

MERGE_PAIR = ('AA', 'BB')
CHARS      = ['freq', 'premium', 'service']
COST_IV    = ['fuel', 'fees', 'distance']
ITER       = pyblp.Iteration('squarem', {'atol': 1e-12, 'max_evaluations': 5000})
F_LIN  = pyblp.Formulation('0 + prices + freq + premium + service')
F_RC   = pyblp.Formulation('1 + prices')
F_AGENT= pyblp.Formulation('0 + income_dev')

# ============================================================
# setup: carga objetos de demanda/oferta y arma el contexto
# ============================================================
def setup(demand_obj='informacion_demanda.obj', supply_obj='informacion_oferta.obj'):
    df, res, cost_iv, product_iv = pickle.load(open(demand_obj, 'rb'))
    df_s, gamma, kappa0, X_list, LOG_MC = pickle.load(open(supply_obj, 'rb'))
    df = df.copy().reset_index(drop=True)

    df['xi_resid'] = np.asarray(res.xi).flatten()
    df['xi_fe']    = np.asarray(res.xi_fe).flatten()
    df = df.merge(df_s[['market_ids', 'product_ids', 'firm_ids', 'omega']],
                  on=['market_ids', 'product_ids', 'firm_ids'], how='left')

    # media de costo (X'gamma) por vuelo: [const, fuel, fees, distance, freq, premium, service]
    Xcost = np.column_stack([np.ones(len(df))] +
                            [df[c].to_numpy() for c in COST_IV + CHARS])
    df['mc_lhs'] = Xcost @ np.asarray(gamma).flatten()

    # agentes (income_dev + nodos) reconstruidos del problema de demanda
    ag = res.problem.agents
    agents = pd.DataFrame({'market_ids': np.asarray(ag.market_ids).flatten(),
                           'weights':    np.asarray(ag.weights).flatten(),
                           'income_dev': np.asarray(ag.demographics)[:, 0]})
    nodes = np.asarray(ag.nodes)
    for k in range(nodes.shape[1]):
        agents['nodes' + str(k)] = nodes[:, k]

    # pool de residuos (xi_resid, omega) de los vuelos ofrecidos, para re-sortear
    pool = df[['firm_ids', 'xi_resid', 'omega']].dropna().reset_index(drop=True)

    return {'df': df, 'res': res, 'agents': agents, 'pool': pool, 'LOG_MC': LOG_MC,
            'rng': np.random.default_rng(1234)}

# ============================================================
# propiedad parcial Omega(kappa) para un subconjunto de vuelos
# ============================================================
def build_omega(firms, kappa):
    n = len(firms); O = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if firms[a] == firms[b]:
                O[a, b] = 1.0
            elif {firms[a], firms[b]} == set(MERGE_PAIR):
                O[a, b] = kappa
    return O

def _mc(ctx, mc_lhs, omega):
    lhs = np.asarray(mc_lhs) + np.asarray(omega)
    return np.exp(lhs) if ctx['LOG_MC'] else np.maximum(lhs, 1e-3)

# ============================================================
# equilibrio de un conjunto de vuelos bajo Omega(kappa)
#   devuelve precios, shares, costos, excedente del consumidor
# ============================================================
def _solve(ctx, rows, xi_vec, costs, kappa, seed_prices):
    m = rows['market_ids'].iloc[0]
    prod = rows[['market_ids', 'firm_ids', 'product_ids'] + CHARS].copy()
    prod['prices'] = np.asarray(seed_prices, float)
    prod['xi_end'] = np.asarray(xi_vec, float)
    agents_m = ctx['agents'][ctx['agents']['market_ids'] == m]
    sim = pyblp.Simulation((F_LIN, F_RC), prod, beta=ctx['res'].beta,
                           sigma=ctx['res'].sigma, pi=ctx['res'].pi,
                           agent_formulation=F_AGENT, agent_data=agents_m,
                           xi=prod['xi_end'].to_numpy())
    sr = sim.replace_endogenous(costs=np.asarray(costs), prices=np.asarray(seed_prices, float),
                                iteration=ITER)
    O = build_omega(prod['firm_ids'].tolist(), kappa)
    p  = np.asarray(sr.compute_prices(ownership=O, costs=np.asarray(costs))).flatten()
    s  = np.asarray(sr.compute_shares(p)).flatten()
    cs = float(np.asarray(sr.compute_consumer_surpluses(prices=p)).flatten()[0])
    return p, s, cs

def _vp(p, s, costs, firms, owners):
    return float(sum((p[i] - costs[i]) * s[i] for i in range(len(p)) if firms[i] in owners))

# ============================================================
# (1)(2) valor incremental esperado de un vuelo  E[VP(J) - VP(J\{j})]
#   mode='own'   -> propiedad standalone (kappa=0), VP de la firma del vuelo  => DeltaVP^pre
#   mode='joint' -> propiedad Omega(kappa),         VP_AA+VP_BB              => DeltaVP^post
# ============================================================
def delta_vp(ctx, market, flight, kappa=0.0, mode='own', n_draws=16, base_keep=None):
    """Valor incremental esperado del 'flight' DENTRO del conjunto base_keep.
       base_keep=None usa todos los vuelos del mercado (leave-one-out clasico)."""
    df = ctx['df']; g = df[df['market_ids'] == market].reset_index(drop=True)
    if base_keep is not None:
        g = g[g['product_ids'].isin(base_keep)].reset_index(drop=True)
    J = len(g); firms = g['firm_ids'].tolist(); prods = g['product_ids'].tolist()
    pos = prods.index(flight)
    owners = {firms[pos]} if mode == 'own' else set(MERGE_PAIR)
    if mode == 'own':
        kappa = 0.0
    obs_p = g['prices'].to_numpy()

    # re-sorteo con numeros comunes: XI[s,i], OM[s,i] para cada vuelo i
    pool = ctx['pool']; rng = ctx['rng']
    XI = np.empty((n_draws, J)); OM = np.empty((n_draws, J))
    for i in range(J):
        idx = rng.integers(0, len(pool), n_draws)
        XI[:, i] = pool['xi_resid'].to_numpy()[idx] + g['xi_fe'].iloc[i]   # FE propio + residuo
        OM[:, i] = pool['omega'].to_numpy()[idx]
    mc_lhs = g['mc_lhs'].to_numpy()

    keep = [i for i in range(J) if i != pos]
    dvps = []
    for s in range(n_draws):
        cJ  = _mc(ctx, mc_lhs, OM[s, :])
        try:
            pJ, sJ, _ = _solve(ctx, g, XI[s, :], cJ, kappa, obs_p)
            vJ = _vp(pJ, sJ, cJ, firms, owners)
            sub = g.iloc[keep].reset_index(drop=True)
            pD, sD, _ = _solve(ctx, sub, XI[s, keep], cJ[keep], kappa, obs_p[keep])
            vD = _vp(pD, sD, cJ[keep], [firms[i] for i in keep], owners)
            dvps.append(vJ - vD)
        except Exception:
            continue
    return float(np.mean(dvps)) if dvps else np.nan

# ============================================================
# (3) excedente del consumidor de un conjunto de vuelos (shocks realizados)
#   keep_flights: lista de product_ids a conservar; kappa fija la conducta
# ============================================================
def consumer_surplus(ctx, market, keep_flights, kappa):
    df = ctx['df']; g = df[df['market_ids'] == market]
    g = g[g['product_ids'].isin(keep_flights)].reset_index(drop=True)
    xi = (g['xi_fe'] + g['xi_resid']).to_numpy()
    costs = _mc(ctx, g['mc_lhs'].to_numpy(), g['omega'].to_numpy())
    _, _, cs = _solve(ctx, g, xi, costs, kappa, g['prices'].to_numpy())
    return cs

# ============================================================
# DEMO / solucionario: un mercado troncal de punta a punta
# ============================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # ELIGE AQUI la ruta y el kappa (el alumno edita estas dos lineas).
    #   ROUTE = 'R00_T0'  -> analiza esa ruta.
    #   ROUTE = None      -> usa la primera ruta head-to-head disponible.
    # ------------------------------------------------------------------
    ROUTE = None
    KAPPA = 0.5

    ctx = setup()
    df = ctx['df']

    # rutas disponibles donde AA y BB se solapan (comparten ruta)
    h2h = [m for m, g in df.groupby('market_ids')
           if {'AA', 'BB'}.issubset(set(g['firm_ids']))]
    print(f"Rutas head-to-head disponibles ({len(h2h)}): {h2h}")

    market = ROUTE if ROUTE is not None else h2h[0]
    if market not in h2h:
        raise SystemExit(f"La ruta '{market}' no es head-to-head. Elige una de: {h2h}")

    g = df[df['market_ids'] == market]
    ab = g[g['firm_ids'].isin(MERGE_PAIR)]['product_ids'].tolist()
    print(f"Analizando ruta: {market}   (kappa={KAPPA})   vuelos AA/BB: {ab}")

    print(f"\n{'flight':8s}{'dVP_pre':>10s}{'dVP_post':>10s}{'region':>13s}")
    D = []; pre_of = {}
    for j in ab:
        pre  = delta_vp(ctx, market, j, mode='own')
        post = delta_vp(ctx, market, j, kappa=KAPPA, mode='joint')
        pre_of[j] = pre
        if post < 0:        reg = 'must-drop'; D.append(j)
        elif post < pre:    reg = 'at-risk';   D.append(j)
        else:               reg = 'safe'
        print(f"{j:8s}{pre:10.4f}{post:10.4f}{reg:>13s}")

    Jall = g['product_ids'].tolist()

    # --- consistencia del retiro SIMULTANEO de todos los at-risk ---------------
    # Si al quedar SOLO (sin el hermano), el valor del vuelo supera su cota pre,
    # entonces NO se retiraria una vez solo: el doble retiro es inconsistente.
    print("\nChequeo de interaccion (valor del vuelo cuando el hermano ya se fue):")
    double_ok = True
    for j in D:
        lone_base = [x for x in Jall if x not in D or x == j]   # quita los otros at-risk
        lone = delta_vp(ctx, market, j, kappa=KAPPA, mode='joint', base_keep=lone_base)
        flag = 'retirable aun solo' if lone < pre_of[j] else 'se mantiene si queda solo'
        if lone >= pre_of[j]: double_ok = False
        print(f"  {j:8s} valor solo {lone:8.4f}  vs cota pre {pre_of[j]:7.4f}  -> {flag}")
    print(f"  => retiro simultaneo de todos los at-risk: "
          f"{'consistente' if double_ok else 'INCONSISTENTE (worst case = un solo retiro)'}")

    # --- excedente del consumidor: mejor caso vs peor caso CONSISTENTE ---------
    cs_pre  = consumer_surplus(ctx, market, Jall, kappa=0.0)     # standalone, sin fusion
    cs_full = consumer_surplus(ctx, market, Jall, kappa=KAPPA)   # fusion, sin retiro
    # peor caso consistente: el retiro UNICO mas danino entre los at-risk
    worst_cs, worst_j = cs_full, None
    for j in D:
        cs_j = consumer_surplus(ctx, market, [x for x in Jall if x != j], kappa=KAPPA)
        if cs_j < worst_cs:
            worst_cs, worst_j = cs_j, j

    print(f"\nDroppable (at-risk) set D = {D}")
    print(f"CS pre-merger                 {cs_pre:9.4f}")
    print(f"CS post, no withdrawal        {cs_full:9.4f}   (repricing  {cs_full-cs_pre:+.4f})")
    print(f"CS post, cancel '{worst_j}'      {worst_cs:9.4f}   (withdrawal {worst_cs-cs_full:+.4f})")
    print(f"\nDelta CS interval: [{worst_cs-cs_pre:+.4f}, {cs_full-cs_pre:+.4f}]"
          f"   width {cs_full-worst_cs:.4f}")
