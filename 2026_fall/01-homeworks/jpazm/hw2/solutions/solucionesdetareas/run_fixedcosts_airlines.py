# -*- coding: utf-8 -*-
"""
Costos fijos a la Eizenberg (2014): cotas por preferencia revelada
==================================================================

Identificacion PARCIAL de costos fijos via preferencia revelada (estilo PPHI).

TIMING (Eizenberg): la firma elige portafolio conociendo solo la DISTRIBUCION de
los shocks (xi, omega), no sus realizaciones; despues se realizan y se juega
Bertrand. => decision por utilidad variable ESPERADA, sin seleccion sobre los
shocks realizados.
  (a) OFRECIDO j:    F_j <= E[ pi_f(J)      - pi_f(J\\{j}) ]   (COTA SUPERIOR)
  (b) NO OFRECIDO k: F_k >= E[ pi_f(J u {k}) - pi_f(J)     ]   (COTA INFERIOR)

METODO: la E[.] se integra RE-SORTEANDO LOS SHOCKS DE TODOS LOS PRODUCTOS. En cada
draw s, a cada producto se le asigna un par (xi,omega) resampleado de la distribucion
recuperada de los OFRECIDOS, se recomputa el equilibrio, y se compara con/sin el
producto focal usando LOS MISMOS draws (numeros aleatorios comunes). El equilibrio
del mercado completo J por draw, vp_J[s], es el pivote reutilizable.

PRE-REQUISITOS:
  - 'informacion_demanda.obj' = [df, updated_rc_results, cost_iv, product_iv]
  - 'informacion_oferta.obj'  = [df, gamma, kappa, X_list, LOG_MC]
  - 'products.csv'            = catalogo maestro
"""

import numpy as np
import pandas as pd
import pyblp
from scipy.optimize import minimize

# ============================================================
# CONFIGURACION
# ============================================================
DEMAND_OBJ  = 'informacion_demanda.obj'
SUPPLY_OBJ  = 'informacion_oferta.obj'
CATALOG_CSV = 'products.csv'

RESTRICT_PRESENT_FIRMS = True
MARKET_SIZE_COL        = None

N_DRAWS    = 8                             # empieza chico (8) para validar, luego sube
DRAW_POOL  = 'firm'
MIN_FIRM_POOL = 6
DRAW_SEED  = 1234

# Robustez del equilibrio
MC_FLOOR   = None                           # piso de costo marginal; None => data-driven (>0)
ITER_METHOD = 'squarem'                     # iteracion del punto fijo de precios
ITER_OPTS   = {'atol': 1e-12, 'max_evaluations': 5000}

ESTIMATE_FC_FUNCTION = True
W_COLS               = ['const', 'freq', 'premium', 'service']
SAVE_OUTPUT = True

# ============================================================
# 0. Carga
# ============================================================
with open(DEMAND_OBJ, 'rb') as f:
	df, res, cost_iv, product_iv = pd.read_pickle(f)
with open(SUPPLY_OBJ, 'rb') as f:
	df_sup, gamma_sup, kappa_sup, XLIST_SUP, LOG_MC_SUP = pd.read_pickle(f)

df = df.copy().reset_index(drop=True)
catalog = pd.read_csv(CATALOG_CSV).rename(columns={'product_id': 'product_ids', 'firm_id': 'firm_ids'})
CHARS = ['freq', 'premium', 'service']

assert 'mc_rc' in df.columns, "df no trae 'mc_rc'."
assert XLIST_SUP[:1 + len(cost_iv)] == ['constant'] + list(cost_iv), \
	f"X_list de oferta inesperado (prefijo): {XLIST_SUP}."
# La oferta puede incluir caracteristicas en el costo (a la Eizenberg). Si las
# incluye, el mc del producto focal se predice con SUS caracteristicas.
COST_HAS_CHARS = all(c in XLIST_SUP for c in CHARS)

df['xi_resid'] = np.asarray(res.xi).flatten()        # residuo estructural (FE-free)
df['xi_fe']    = np.asarray(res.xi_fe).flatten()     # efectos fijos absorbidos (region+periodo+firma)
# xi_end = xi_resid + xi_fe, pero el FE NO se resamplea: es deterministico por mercado/firma.

_sup_cols = ['market_ids', 'product_ids', 'firm_ids', 'omega']
if 'mc_floored' in df_sup.columns:
	_sup_cols.append('mc_floored')
df = df.merge(df_sup[_sup_cols],
			  on=['market_ids', 'product_ids', 'firm_ids'], how='left')
assert df['omega'].notna().any(), "No se pudo traer 'omega' de informacion_oferta.obj."

# Pool de RESIDUOS (xi_resid, omega) de los OFRECIDOS, EXCLUYENDO mc pisados.
# Se resamplea SOLO el residuo; el efecto fijo del producto focal se re-agrega
# de forma deterministica en el barrido (fe_m), no se sortea.
_pool = df.copy()
if 'mc_floored' in _pool.columns:
	_pool = _pool[~_pool['mc_floored'].fillna(False)]
POOL = _pool[['firm_ids', 'xi_resid', 'omega']].dropna().reset_index(drop=True)

# Piso de costo marginal (solo aplica en modo NIVELES; con logs mc = exp(.) > 0).
if MC_FLOOR is None:
	_pos = df['mc_rc'][df['mc_rc'] > 0]
	MC_FLOOR = float(max(1e-3, 0.1 * _pos.min())) if len(_pos) else 1e-3

def transform_mc(lhs):
	"""lhs = X'gamma + omega -> mc en niveles. Con logs, exp() (siempre > 0); el piso
	   MC_FLOOR solo aplica en modo NIVELES, donde mc puede salir <= 0."""
	lhs = np.asarray(lhs, dtype=float)
	return np.exp(lhs) if LOG_MC_SUP else np.maximum(lhs, MC_FLOOR)

# ============================================================
# 1. Agentes de integracion
# ============================================================
def agents_from_results(res):
	ag = res.problem.agents
	out = pd.DataFrame({'market_ids': np.asarray(ag.market_ids).flatten(),
						'weights':    np.asarray(ag.weights).flatten()})
	out['income_dev'] = np.asarray(ag.demographics)[:, 0]
	nodes = np.asarray(ag.nodes)
	for k in range(nodes.shape[1]):
		out['nodes' + str(k)] = nodes[:, k]
	return out

agents = agents_from_results(res)

F_LIN   = pyblp.Formulation('0 + prices + freq + premium + service')
F_RC    = pyblp.Formulation('1 + prices')
F_AGENT = pyblp.Formulation('0 + income_dev')
ITER    = pyblp.Iteration(ITER_METHOD, ITER_OPTS)

# ============================================================
# 2. Conjunto potencial
# ============================================================
def build_potential_set(df, catalog, restrict_present=True):
	rows = []
	for m, mdf in df.groupby('market_ids'):
		offered = set(mdf['product_ids']); present = set(mdf['firm_ids'].unique())
		feas = catalog[catalog['firm_ids'].isin(present)] if restrict_present else catalog
		for _, r in feas.iterrows():
			rows.append({'market_ids': m, 'product_ids': r['product_ids'], 'firm_ids': r['firm_ids'],
						 'freq': r['freq'], 'premium': r['premium'],
						 'service': r['service'], 'offered': r['product_ids'] in offered})
	return pd.DataFrame(rows)

potential = build_potential_set(df, catalog, RESTRICT_PRESENT_FIRMS)
print(f"Conjunto potencial: {len(potential)} filas "
	  f"({int(potential['offered'].sum())} ofrecidos, "
	  f"{int((~potential['offered']).sum())} no ofrecidos). N_DRAWS={N_DRAWS}.")

# Lista de productos ofrecidos vs factibles-no-ofrecidos (por mercado). Se guarda
# AQUI, antes del barrido pesado, para tenerla aunque el loop tarde o falle.
if SAVE_OUTPUT:
	_plist = potential.copy()
	_plist['status'] = np.where(_plist['offered'], 'ofrecido', 'no_ofrecido')
	_plist = _plist.sort_values(['market_ids', 'firm_ids', 'product_ids']).reset_index(drop=True)
	_plist.to_csv('productos_ofrecidos.csv', index=False)
	print("  Guardado: productos_ofrecidos.csv  (market, firm, product, offered, status, caract.)")

# ============================================================
# 3. Draws y media de costo
# ============================================================
_rng = np.random.default_rng(DRAW_SEED)

def draw_pairs(firm, n):
	"""n pares (xi_resid, omega) resampleados del pool. El residuo es FE-free;
	   el efecto fijo se agrega aparte (deterministico) en el barrido."""
	if DRAW_POOL == 'firm':
		sub = POOL[POOL['firm_ids'] == firm]
		src = sub if len(sub) >= MIN_FIRM_POOL else POOL
	else:
		src = POOL
	idx = _rng.integers(0, len(src), n)
	return src['xi_resid'].to_numpy()[idx], src['omega'].to_numpy()[idx]

def mc_lhs_mean(mkt_df, firm, chars=None):
	"""X'gamma del producto focal: cost shifters firma-mercado + (si la oferta las
	   incluye) las CARACTERISTICAS del focal. Asi mc responde a la calidad. El orden
	   de 'row' replica X_list de oferta: [const, cost_iv..., CHARS...]."""
	sub = mkt_df[mkt_df['firm_ids'] == firm]
	use = sub if len(sub) else mkt_df
	row = [1.0] + [float(use[c].mean()) for c in cost_iv]
	if COST_HAS_CHARS:
		if chars is None:
			row += [float(use[c].mean()) for c in CHARS]    # fallback: media firma-mercado
		else:
			row += [float(chars[c]) for c in CHARS]          # caracteristicas del focal
	return float(gamma_sup @ np.array(row))

# ============================================================
# 4. Motor de equilibrio (con warm-start e iteracion robusta)
# ============================================================
PROD_COLS = ['market_ids', 'firm_ids', 'product_ids', 'prices',
			 'freq', 'premium', 'service', 'xi_end']

def build_prod(rows, xi_vec, m, seed_prices):
	out = rows[['firm_ids', 'product_ids', 'freq', 'premium', 'service']].copy().reset_index(drop=True)
	out.insert(0, 'market_ids', m)
	out['prices'] = np.asarray(seed_prices, dtype=float)
	out['xi_end'] = np.asarray(xi_vec, dtype=float)
	return out[PROD_COLS]

def solve_market(prod_m, agents_m, costs, seed_prices):
	"""Re-resuelve Bertrand. Warm-start desde seed_prices e iteracion robusta."""
	sim = pyblp.Simulation(
		product_formulations=(F_LIN, F_RC),
		product_data=prod_m, beta=res.beta, sigma=res.sigma, pi=res.pi,
		agent_data=agents_m, agent_formulation=F_AGENT,
		xi=prod_m['xi_end'].to_numpy())
	sr = sim.replace_endogenous(costs=np.asarray(costs),
								prices=np.asarray(seed_prices, dtype=float),
								iteration=ITER)
	p = sr.compute_prices(costs=np.asarray(costs))      # arranca del equilibrio simulado
	s = sr.compute_shares(p)
	return np.asarray(p).flatten(), np.asarray(s).flatten()

def firm_vp(prod_m, prices, costs, shares, firm, ms):
	mask = (prod_m['firm_ids'].to_numpy() == firm)
	costs = np.asarray(costs)
	return float(np.sum((prices[mask] - costs[mask]) * shares[mask]) * ms)

def solve_draw(rows, xi_vec, mc_lhs, om_vec, seed_prices, agents_m, m):
	"""Un draw: arma productos + costos y resuelve el equilibrio de Bertrand.
	   Devuelve (prod, prices, costs, shares) o None si no converge."""
	costs = transform_mc(np.asarray(mc_lhs) + np.asarray(om_vec))
	prod  = build_prod(rows, xi_vec, m, seed_prices)
	try:
		p, sh = solve_market(prod, agents_m, costs, seed_prices)
	except Exception:
		return None
	return prod, p, costs, sh

# ============================================================
# 5. Barrido
# ============================================================
def _brow(r, m, f):
	return {'market_ids': m, 'product_ids': r['product_ids'], 'firm_ids': f,
			'freq': r['freq'], 'premium': r['premium'],
			'service': r['service']}

bound_rows = []
S = N_DRAWS
n_mkt_done = n_mkt_skip = n_draw_fail = 0

for m, mdf in df.groupby('market_ids'):
	mdf = mdf.reset_index(drop=True)
	agents_m = agents[agents['market_ids'] == m].reset_index(drop=True)
	ms = float(mdf[MARKET_SIZE_COL].iloc[0]) if MARKET_SIZE_COL else 1.0

	jdf = mdf[['firm_ids', 'product_ids'] + CHARS].reset_index(drop=True)
	n = len(jdf)
	obs_p = mdf['prices'].to_numpy()                         # warm-start observado
	mc_means = np.array([mc_lhs_mean(mdf, jdf['firm_ids'].iloc[i], jdf.iloc[i])
						 for i in range(n)])
	# Efecto fijo deterministico por firma en este mercado (region+periodo+firma).
	fe_m = mdf.groupby('firm_ids')['xi_fe'].mean().to_dict()
	fe_default = float(mdf['xi_fe'].mean())                  # fallback (firma ausente)
	XI = np.empty((S, n)); OM = np.empty((S, n))
	for i in range(n):
		f_i = jdf['firm_ids'].iloc[i]
		xr_i, om_i = draw_pairs(f_i, S)
		XI[:, i] = xr_i + fe_m.get(f_i, fe_default)          # residuo sorteado + FE propio
		OM[:, i] = om_i

	# pivote: equilibrio del mercado COMPLETO J por draw (skip draw si no resuelve)
	firms = jdf['firm_ids'].unique()
	vp_J = {}
	for s in range(S):
		out = solve_draw(jdf, XI[s, :], mc_means, OM[s, :], obs_p, agents_m, m)
		if out is None:
			n_draw_fail += 1; continue
		prod, p, costs, sh = out
		vp_J[s] = {f: firm_vp(prod, p, costs, sh, f, ms) for f in firms}
	valid_s = sorted(vp_J.keys())
	if not valid_s:
		print(f"  [skip market {m}] ningun draw resolvio el pivote.")
		n_mkt_skip += 1
		continue
	n_mkt_done += 1

	pot_m = potential[potential['market_ids'] == m]

	# (a) OFRECIDOS: con J vs sin j (mismos draws); cota SUPERIOR
	for _, r in pot_m[pot_m['offered']].iterrows():
		j, f = r['product_ids'], r['firm_ids']
		pos = int(np.where(jdf['product_ids'].to_numpy() == j)[0][0])
		cols = [c for c in range(n) if c != pos]
		sub  = jdf.iloc[cols]
		dvps = []
		for s in valid_s:
			out = solve_draw(sub, XI[s, cols], mc_means[cols], OM[s, cols], obs_p[cols], agents_m, m)
			if out is None:
				continue
			prod, p, costs, sh = out
			dvps.append(vp_J[s][f] - firm_vp(prod, p, costs, sh, f, ms))
		if dvps:
			bound_rows.append({**_brow(r, m, f), 'bound_type': 'upper',
							   'bound_value': float(np.mean(dvps)), 'dvp_sd': float(np.std(dvps)),
							   'n_eff': len(dvps)})

	# (b) NO OFRECIDOS: con J u {k} vs J (mismos draws); cota INFERIOR
	for _, r in pot_m[~pot_m['offered']].iterrows():
		k, f = r['product_ids'], r['firm_ids']
		xr_k, omk = draw_pairs(f, S)
		xk = xr_k + fe_m.get(f, fe_default)                  # residuo + FE del candidato (m, f)
		mck = mc_lhs_mean(mdf, f, r)              # r trae las caracteristicas del candidato
		cand = pd.DataFrame([{'firm_ids': f, 'product_ids': k,
							  'freq': r['freq'], 'premium': r['premium'],
							  'service': r['service']}])
		rows_k = pd.concat([jdf, cand], ignore_index=True)
		dvps = []
		for s in valid_s:
			xi_vec = np.concatenate([XI[s, :], [xk[s]]])
			mc_lhs = np.concatenate([mc_means, [mck]])
			om_vec = np.concatenate([OM[s, :], [omk[s]]])
			seed   = np.concatenate([obs_p, [1.3 * transform_mc([mck + omk[s]])[0]]])
			out = solve_draw(rows_k, xi_vec, mc_lhs, om_vec, seed, agents_m, m)
			if out is None:
				continue
			prod, p, costs, sh = out
			dvps.append(firm_vp(prod, p, costs, sh, f, ms) - vp_J[s][f])
		if dvps:
			bound_rows.append({**_brow(r, m, f), 'bound_type': 'lower',
							   'bound_value': float(np.mean(dvps)), 'dvp_sd': float(np.std(dvps)),
							   'n_eff': len(dvps)})

# ============================================================
# 6. Ensamble robusto + reporte
# ============================================================
bounds = pd.DataFrame(bound_rows)
if len(bounds):
	bounds = bounds.dropna(subset=['bound_value']).reset_index(drop=True)

print("\n" + "=" * 62)
print("  Costos fijos: cotas por preferencia revelada (Eizenberg 2014)")
print("=" * 62)
print(f"  Mercados resueltos: {n_mkt_done}  |  saltados: {n_mkt_skip}  |  "
	  f"draws de pivote fallidos: {n_draw_fail}")

if len(bounds) == 0:
	print("  No se computaron cotas: revisa que el pivote resuelva "
		  "(LOG_MC=True en la oferta, agents_from_results, ITER_OPTS).")
	print("=" * 62)
else:
	up = bounds[bounds['bound_type'] == 'upper']['bound_value']
	lo = bounds[bounds['bound_type'] == 'lower']['bound_value']
	print(f"  Cotas SUPERIORES (ofrecidos)    : n={len(up):3d}  mediana={up.median():.4f}")
	print(f"  Cotas INFERIORES (no ofrecidos) : n={len(lo):3d}  mediana={lo.median():.4f}")
	if len(up) and len(lo):
		print(f"  Consistencia (mediana inf < mediana sup): "
			  f"{'OK' if lo.median() < up.median() else 'REVISAR'}")
	print("-" * 62)
	print("  Por premium (mediana de la cota):")
	for prem in sorted(bounds['premium'].unique()):
		b = bounds[bounds['premium'] == prem]
		bu = b[b['bound_type'] == 'upper']['bound_value']
		bl = b[b['bound_type'] == 'lower']['bound_value']
		su = f"{bu.median():.4f}" if len(bu) else "  -   "
		sl = f"{bl.median():.4f}" if len(bl) else "  -   "
		print(f"    premium={prem}:  inf={sl}   sup={su}")

	# ====== 7. funcion de costo fijo via desigualdades de momentos ======
	def estimate_fc_function(bounds, W_cols):
		B = bounds.copy(); B['const'] = 1.0
		W = B[W_cols].to_numpy(); upm = (B['bound_type'] == 'upper').to_numpy(); val = B['bound_value'].to_numpy()
		def Q(phi):
			pred = W @ phi
			viol = np.where(upm, np.maximum(0.0, pred - val), np.maximum(0.0, val - pred))
			return float(np.sum(viol ** 2))
		phi0 = np.zeros(W.shape[1]); phi0[0] = np.median(val)
		r = minimize(Q, phi0, method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 40000})
		pred = W @ r.x
		sat = np.where(upm, pred <= val + 1e-9, pred >= val - 1e-9).mean()
		return r.x, Q(r.x), sat

	if ESTIMATE_FC_FUNCTION:
		phi_hat, Qmin, sat = estimate_fc_function(bounds, W_COLS)
		print("-" * 62)
		print("  Funcion de costo fijo  F_j = phi' W_j  (un punto del conjunto identificado)")
		for name, b in zip(W_COLS, phi_hat):
			print(f"    {name:14s} {b:9.4f}")
		print(f"  Q(phi_hat) = {Qmin:.4e}   desigualdades satisfechas = {sat:.2%}")
	print("=" * 62)

	# ====== 8. guardado ======
	if SAVE_OUTPUT:
		bounds.to_csv('costos_fijos_cotas.csv', index=False)
		out = [bounds, (phi_hat if ESTIMATE_FC_FUNCTION else None), W_COLS]
		with open('informacion_costos_fijos.obj', 'wb') as f:
			pd.to_pickle(out, f)
		print("  Guardado: costos_fijos_cotas.csv  e  informacion_costos_fijos.obj")
