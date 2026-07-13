# -*- coding: utf-8 -*-
"""
SOLUCION Parte 3 (docente/RA): barre TODAS las rutas head-to-head x kappa.
Guarda results_part3.csv incrementalmente (reanudable) y, al terminar, produce
  - fig_dcs_vs_kappa.png : mediana del intervalo Delta CS (ambos extremos) vs kappa
  - fig_width_vs_div.png : ancho del intervalo vs diversion (Parte 2b), por ruta
  - summary_part3.csv    : mediana por kappa
Vuelve a correr el script para reanudar si se corta (lee lo ya hecho en el CSV).
"""
import os, csv, numpy as np, pandas as pd
import merger_engine as me

N_DRAWS = 4
KAPPAS  = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT     = 'results_part3.csv'
COLS    = ['route', 'kappa', 'n_atrisk', 'cs_pre', 'cs_full', 'cs_worst',
           'repricing', 'withdrawal', 'width', 'div_sym']

ctx = me.setup(); df = ctx['df']
h2h = [m for m, g in df.groupby('market_ids') if {'AA', 'BB'}.issubset(set(g['firm_ids']))]

# diversion simetrica AA-BB por ruta (para el scatter de la Parte 2b)
div = ctx['res'].compute_diversion_ratios()
def route_div(m):
    g = df[df['market_ids'] == m]; rows = g.index.tolist(); J = len(rows)
    fr = g['firm_ids'].tolist(); Dm = div[np.ix_(rows, range(J))]
    ia, ib = fr.index('AA'), fr.index('BB')
    return 0.5 * (Dm[ia, ib] + Dm[ib, ia])
DIV = {m: float(route_div(m)) for m in h2h}

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = {(r.route, round(r.kappa, 2)) for r in prev.itertuples()}
else:
    with open(OUT, 'w', newline='') as f:
        csv.writer(f).writerow(COLS)

def worst_consistent(m, J, ab, kappa, cs_full):
    worst, atrisk = cs_full, []
    for j in ab:
        pre  = me.delta_vp(ctx, m, j, mode='own',   n_draws=N_DRAWS)
        post = me.delta_vp(ctx, m, j, kappa=kappa, mode='joint', n_draws=N_DRAWS)
        if post < pre:
            atrisk.append(j)
            c = me.consumer_surplus(ctx, m, [x for x in J if x != j], kappa=kappa)
            worst = min(worst, c)
    return worst, atrisk

for m in h2h:
    g = df[df['market_ids'] == m]; J = g['product_ids'].tolist()
    ab = g[g['firm_ids'].isin(me.MERGE_PAIR)]['product_ids'].tolist()
    cs_pre = me.consumer_surplus(ctx, m, J, kappa=0.0)
    for k in KAPPAS:
        if (m, round(k, 2)) in done:
            continue
        cs_full = me.consumer_surplus(ctx, m, J, kappa=k)
        worst, atrisk = worst_consistent(m, J, ab, k, cs_full)
        row = [m, k, len(atrisk), cs_pre, cs_full, worst,
               cs_full - cs_pre, worst - cs_full, cs_full - worst, DIV[m]]
        with open(OUT, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    print(f"  {m} done", flush=True)

res = pd.read_csv(OUT)
if len(res) >= len(h2h) * len(KAPPAS):
    print("\nSWEEP COMPLETO. Resumen (mediana por kappa):")
    summ = res.groupby('kappa').agg(
        median_repricing=('repricing', 'median'),
        median_withdrawal=('withdrawal', 'median'),
        median_width=('width', 'median')).round(3)
    summ['median_lower'] = res.groupby('kappa').apply(
        lambda d: (d['cs_worst'] - d['cs_pre']).median()).round(3)
    print(summ.to_string()); summ.to_csv('summary_part3.csv')

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    up = res.groupby('kappa')['repricing'].median()
    lo = res.groupby('kappa').apply(lambda d: (d['cs_worst'] - d['cs_pre']).median())
    ks = up.index.values
    plt.figure(figsize=(5.2, 3.6))
    plt.fill_between(ks, lo.values, up.values, alpha=0.15, color='C0')
    plt.plot(ks, up.values, 'o-', label='best case (repricing only)')
    plt.plot(ks, lo.values, 's-', label='worst case (one cancellation)')
    plt.xlabel(r'$\kappa$'); plt.ylabel(r'median $\Delta CS$')
    plt.title(r'Traveler-welfare interval vs $\kappa$ (all routes)')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig('fig_dcs_vs_kappa.png', dpi=140)

    sub = res[res['kappa'] == 0.5]
    plt.figure(figsize=(5.2, 3.6))
    plt.scatter(sub['div_sym'], sub['width'], s=18, alpha=0.7)
    plt.xlabel(r'symmetrized AA-BB diversion $\bar D_{AB}$ (Part 2b)')
    plt.ylabel(r'$\Delta CS$ width ($\kappa=0.5$)')
    plt.title('Withdrawal ambiguity vs closeness of overlap')
    plt.tight_layout(); plt.savefig('fig_width_vs_div.png', dpi=140)
    print("Figuras: fig_dcs_vs_kappa.png, fig_width_vs_div.png")
else:
    print(f"\nParcial: {len(res)}/{len(h2h)*len(KAPPAS)} filas. Reanuda corriendo de nuevo.")
