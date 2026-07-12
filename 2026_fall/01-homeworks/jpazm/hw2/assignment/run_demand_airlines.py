# -*- coding: utf-8 -*-
"""
DEMANDA (exacta) para los datos simulados de aerolineas
=======================================================
Estima la demanda BLP que genero los datos y guarda 'informacion_demanda.obj'
en el formato del pipeline:  [df, updated_rc_results, cost_iv, product_iv].

Especificacion (= DGP):
  u_ijt = alpha_i p + b1 freq + b2 premium + b3 service + xi_jt + eps
  alpha_i = alpha + sigma_p nu_i + pi_p income_dev_i   (coef. aleatorio en tarifa)
            (mas un coef. aleatorio en la constante: sigma_0 nu_0i + pi_0 income_dev)
  xi absorbe efectos fijos C(period)+C(region)+C(firm_ids); bien externo = no volar.
Instrumentos para la tarifa endogena:
  - cost shifters: combustible, tasas, distancia
  - diferenciacion Gandhi-Houde de RIVALES en freq y service
    (las 'propias' y las de premium se omiten: colineales con los FE en esta muestra)
"""
import numpy as np, pandas as pd, pyblp, pickle
from scipy.stats import qmc
pyblp.options.verbose = False

df  = pd.read_csv('blp_data.csv')
inc = pd.read_csv('income_draws.csv').rename(columns={'market_id': 'market_ids'})

# ---- agentes: income_dev + pesos + 2 nodos Halton (constante, tarifa) ----
inc['income_dev'] = inc['log_income'] - inc.groupby('market_ids')['log_income'].transform('mean')
samp = qmc.Halton(d=2, seed=7); ags = []
for m, g in inc.groupby('market_ids'):
    nd = samp.random(n=len(g))
    a = g.copy(); a['weights'] = 1.0 / len(g); a['nodes0'] = nd[:, 0]; a['nodes1'] = nd[:, 1]
    ags.append(a)
agents = pd.concat(ags, ignore_index=True)

# ---- instrumentos ----
cost_iv = ['fuel', 'fees', 'distance']
for x0, x in [('freq', 'freq'), ('service', 'service')]:
    t = pyblp.build_differentiation_instruments(pyblp.Formulation('0+' + x), df, version='local')
    df['diff_iv_or_rival_' + x0] = t[:, 1]          # solo RIVAL
product_iv = ['diff_iv_or_rival_freq', 'diff_iv_or_rival_service']
for i, v in enumerate(cost_iv + product_iv):
    df['demand_instruments' + str(i)] = df[v]

# ---- formulaciones ----
X1 = pyblp.Formulation('0 + prices + freq + premium + service',
                       absorb='C(period)+C(region)+C(firm_ids)')
X2 = pyblp.Formulation('1 + prices')          # coef. aleatorios en (constante, tarifa)
X3 = pyblp.Formulation('0 + income_dev')      # demografia

problem = pyblp.Problem((X1, X2), df, agent_data=agents, agent_formulation=X3)

sigma0 = np.diag([0.30, 0.006])
pi0    = np.array([[0.35], [0.004]])
rc = problem.solve(sigma=sigma0, pi=pi0, method='1s')

# instrumentos optimos + re-estimacion
oi = rc.compute_optimal_instruments(method='approximate')
updated_rc_results = oi.to_problem().solve(sigma=rc.sigma, pi=rc.pi, method='1s')

# objetos para oferta / costos fijos
df['mc_rc']    = updated_rc_results.compute_costs()
df['mu_rc']    = updated_rc_results.compute_markups()
df['lerner']   = (df['prices'] - df['mc_rc']) / df['prices']
own_el = np.asarray(updated_rc_results.extract_diagonals(
    updated_rc_results.compute_elasticities())).flatten()

print("=" * 56)
print("  DEMANDA estimada (verdad entre parentesis)")
print("=" * 56)
print(f"  alpha (tarifa)      {updated_rc_results.beta[0,0]:+.4f}  (-0.0120)")
print(f"  freq                {updated_rc_results.beta[1,0]:+.4f}  (+0.4000)")
print(f"  premium             {updated_rc_results.beta[2,0]:+.4f}  (+0.5500)")
print(f"  service             {updated_rc_results.beta[3,0]:+.4f}  (+0.3000)")
print(f"  sigma_const         {updated_rc_results.sigma[0,0]:+.4f}  (+0.3000)")
print(f"  sigma_price         {updated_rc_results.sigma[1,1]:+.4f}  (+0.0060)")
print(f"  pi(income->const)   {updated_rc_results.pi[0,0]:+.4f}  (+0.3500)")
print(f"  pi(income->price)   {updated_rc_results.pi[1,0]:+.4f}  (+0.0040)")
print(f"  elast. propia med.  {np.median(own_el):+.2f}")
print(f"  mc_rc > 0           {(df['mc_rc']>0).mean():.1%}   Lerner medio {df['lerner'].mean():.3f}")
print("=" * 56)

with open('informacion_demanda.obj', 'wb') as f:
    pickle.dump([df, updated_rc_results, cost_iv, product_iv], f)
print("  Guardado: informacion_demanda.obj  [df, updated_rc_results, cost_iv, product_iv]")

# ============================================================
# Parte 2(b): diversion de los pares AA-BB que se solapan
#   Un par AA-BB en la MISMA ruta (mismo sufijo _Rxx) es un solapamiento.
#   Se listan todos; la diversion mide cuan cerca estan -> presion de retiro (Parte 3).
# ============================================================
div = updated_rc_results.compute_diversion_ratios()   # formato N x J_max (col = producto del mercado)

print("\n" + "=" * 56)
print("  Parte 2(b): diversion de pares AA-BB que comparten ruta")
print("=" * 56)
print(f"  {'ruta':8s}{'par':16s}{'D(AA->BB)':>11s}{'D(BB->AA)':>11s}{'D_sym':>8s}")
for m, g in df.groupby('market_ids'):
    fr = g['firm_ids'].tolist()
    if not {'AA', 'BB'}.issubset(set(fr)):
        continue
    rows  = g.index.tolist(); J = len(rows)
    prods = g['product_ids'].tolist()
    Dm    = div[np.ix_(rows, range(J))]               # J x J (columnas locales)
    ia = fr.index('AA'); ib = fr.index('BB')
    d_ab, d_ba = Dm[ia, ib], Dm[ib, ia]
    print(f"  {m:8s}{prods[ia]+'/'+prods[ib]:16s}{d_ab:11.4f}{d_ba:11.4f}{0.5*(d_ab+d_ba):8.4f}")
print("  (cada par comparte ruta = solapamiento; la diversion mide cuan cerca estan)")
print("=" * 56)
