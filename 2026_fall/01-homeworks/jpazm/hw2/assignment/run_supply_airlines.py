# -*- coding: utf-8 -*-
"""
Estimacion de OFERTA: conducta (kappa) O recuperacion simple de omega
=====================================================================

Este archivo hace UNA de dos cosas, segun el switch ESTIMATE_CONDUCT:

  (A) ESTIMATE_CONDUCT = True   -> estima el parametro de conducta kappa por GMM
      (Miller-Weinberg 2017 / Bresnahan-Lau en diferenciados). kappa in [0,1]
      mide cuanto internalizan dos firmas (A y B) las utilidades de la otra:
          kappa = 0  -> Bertrand-Nash (cada firma maximiza solo sus productos)
          kappa = 1  -> A y B maximizan utilidades conjuntas (colusion total)

  (B) ESTIMATE_CONDUCT = False  -> NO estima conducta. FIJA la conducta en
      KAPPA_FIXED (por defecto 0 = Bertrand-Nash multiproducto, el supuesto de
      Eizenberg 2014) y solo recupera la ecuacion de costo marginal para
      obtener los errores estructurales de oferta omega_jt.

      Esto es lo que necesita Eizenberg (2014): con la conducta SUPUESTA, el
      costo marginal de los productos OFRECIDOS queda determinado por la CPO
      (mc = p - markup); se proyecta sobre los cost shifters y el residuo
      omega_jt es el shock de costo estructural que alimenta la etapa de
      utilidades variables / costos fijos (tradicion BEW).

Idea comun a ambos modos:
  - La DEMANDA ya esta estimada (la tomamos como dada).
  - Una conducta (un kappa) implica una matriz de propiedad (ownership)
    -> markups -> COSTO MARGINAL implicito via la CPO:  mc_jt = precio_jt - markup_jt.
  - Modo (A): se BUSCA el kappa cuyo omega es ortogonal a instrumentos (rotadores
              de demanda). Eso es un GMM en kappa.
  - Modo (B): el kappa esta DADO; mc queda determinado; solo se proyecta mc sobre
              los cost shifters (regresion lineal) y se guarda el residuo omega.

PRE-REQUISITO: existir 'informacion_demanda.obj' con
   [df, updated_rc_results, cost_iv, product_iv]  (salida de ta_session.py).
"""

import numpy as np
import pyblp
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize

# ============================================================
# CONFIGURACION
# ============================================================
ESTIMATE_CONDUCT = False   # False -> oferta simple (recupera omega para Eizenberg 2014)
                           # True  -> estima conducta kappa por GMM (Miller-Weinberg)

KAPPA_FIXED = 0.0          # conducta SUPUESTA cuando ESTIMATE_CONDUCT = False.
                           # 0.0 = Bertrand-Nash multiproducto (supuesto de Eizenberg).

LOG_MC = True             # False -> ecuacion en niveles:  mc      = X gamma + omega
                           # True  -> ecuacion log-lineal:   log(mc) = X gamma + omega
                           #          (la forma usual en Eizenberg; omega en logs).

# --- Manejo de mc no positivos (solo importa con LOG_MC=True) ---
# La CPO puede devolver mc <= 0 (markup > precio) en algunos productos. log(mc) no
# existe ahi. En vez de fallar, se PISAN esos mc en un valor positivo = fraccion de
# los mc positivos. Se marca cuales se pisaron (df['mc_floored']) para diagnostico
# y para poder excluirlos aguas abajo. OJO: si la fraccion pisada es alta, la
# demanda esta mal especificada y el piso contamina gamma/omega.
FLOOR_NEGATIVE_MC = True    # True -> pisa mc<=0; False -> falla si hay mc<=0 (con logs)
MC_FLOOR_FRAC     = 0.10    # piso = MC_FLOOR_FRAC * referencia de los mc positivos
MC_FLOOR_REF      = 'median'  # referencia: 'min' | 'median' | 'mean' | 'p5' | 'p10'

# --- Caracteristicas del producto en la ecuacion de costo (a la Eizenberg) ---
# True  -> log(mc) = const + cost_shifters'gamma + caracteristicas'delta + omega
#          (mc responde a calidad/premium/estilo; los productos contrafactuales
#           tienen costo que depende de lo que SON, no solo del firma-mercado).
# False -> solo cost shifters (mc plano dentro de firma-mercado).
# Supuesto: caracteristicas EXOGENAS al shock de costo (E[omega|x]=0), estandar BLP.
COST_INCLUDES_CHARS = True

SAVE_OUTPUT = True         # guarda 'informacion_oferta.obj' con df (mc, omega) y gamma.

# El par que potencialmente coludimos / fusiona (solo relevante si ESTIMATE_CONDUCT=True
# o si KAPPA_FIXED != 0). Revisar df.firm_ids.unique().
FIRM_A, FIRM_B = "AA", "BB"

# ============================================================
# 0. Carga de insumos de demanda
# ============================================================
with open('informacion_demanda.obj', 'rb') as file:
	dpickle = pd.read_pickle(file)

df                 = dpickle[0]
updated_rc_results = dpickle[1]
cost_iv            = dpickle[2]
product_iv         = dpickle[3]

for _name, _obj in [('df', df), ('updated_rc_results', updated_rc_results),
					('cost_iv', cost_iv), ('product_iv', product_iv)]:
	assert _obj is not None, f"Falta '{_name}'. Corre primero ta_session.py (demanda)."

demand_result = updated_rc_results   # resultado de demanda (RC). Tambien sirve el logit.

# Rotadores de demanda (caracteristicas propias/rivales). Solo se usan como
# instrumentos cuando se ESTIMA conducta; identifican kappa moviendo los markups.
product_iv = ['diff_iv_or_own_x1', 'diff_iv_or_own_x2', 'diff_iv_or_own_x3',
			  'diff_iv_or_rival_x1', 'diff_iv_or_rival_x2', 'diff_iv_or_rival_x3']

###############################################################################
# 1. Modelo de conducta: quien internaliza a quien
###############################################################################

def make_kappa_spec(kappa):
	"""Entrada (f,g) de la matriz de propiedad para un kappa dado:
	   1 en la diagonal (cada firma es duena de sus productos),
	   kappa en el bloque cruzado A-B, 0 en el resto.
	   kappa = 0 reproduce la propiedad estandar (Bertrand-Nash multiproducto)."""
	def spec(f, g):
		if f == g:
			return 1.0
		elif {f, g} == {FIRM_A, FIRM_B}:
			return kappa
		else:
			return 0.0
	return spec

def implied_mc(kappa):
	"""Costo marginal implicito por la CPO de precios, dado kappa.
	   pyblp invierte  p - mc = markup(ownership)  =>  mc = p - markup."""
	ownership = pyblp.build_ownership(df, make_kappa_spec(kappa))
	return demand_result.compute_costs(ownership=ownership).flatten()

def apply_mc_floor(mc):
	"""Pisa los mc no positivos en un valor positivo (fraccion de los mc positivos)
	   para que log(mc) este definido. Devuelve (mc_pisado, mascara_pisados)."""
	mc = np.asarray(mc, dtype=float).copy()
	pos = mc[mc > 0]
	if len(pos) == 0:
		raise ValueError("Todos los mc son <= 0: revisa demanda/markups; no se puede pisar.")
	ref = {'min': pos.min(), 'median': float(np.median(pos)), 'mean': float(pos.mean()),
		   'p5': float(np.percentile(pos, 5)), 'p10': float(np.percentile(pos, 10))}[MC_FLOOR_REF]
	floor = MC_FLOOR_FRAC * ref
	bad = mc <= 0
	mc[bad] = floor
	return mc, bad

def cost_lhs(kappa):
	"""Variable dependiente de la ecuacion de costo (mc o log mc) y el mc en niveles.
	   Con LOG_MC, pisa los mc<=0 (si FLOOR_NEGATIVE_MC) para que log este definido."""
	mc = implied_mc(kappa)
	if LOG_MC:
		if FLOOR_NEGATIVE_MC:
			mc, _ = apply_mc_floor(mc)
		elif np.any(mc <= 0):
			raise ValueError("mc<=0: no se puede tomar log. Usa FLOOR_NEGATIVE_MC=True "
							 "o LOG_MC=False.")
		return np.log(mc), mc
	return mc, mc

###############################################################################
# 2. Ecuacion de costo e instrumentos
#    mc_jt(kappa) = X_jt' gamma + omega_jt
###############################################################################

df['constant'] = 1.0

# Caracteristicas del producto en la ecuacion de costo (a la Eizenberg). Bajo
# COST_INCLUDES_CHARS, mc depende de la calidad/premium/estilo ademas de los cost
# shifters. Las caracteristicas son exogenas al shock de costo (E[omega|x]=0).
CHARS      = ['freq', 'premium', 'service']
cost_chars = CHARS if COST_INCLUDES_CHARS else []

# X: regresores INCLUIDOS (constante + cost shifters [+ caracteristicas]).
X_list = ['constant'] + cost_iv + cost_chars

# Z: instrumentos.
#   - Conducta: exogenos incluidos (constante, cost shifters, caracteristicas) se
#     instrumentan a si mismos; los rotadores de demanda (excluidos) IDENTIFICAN
#     kappa. OJO: con caracteristicas incluidas, los rotadores 'own' (diff_iv_or_own)
#     pueden volverse casi colineales con ellas; si pasa, deja solo los 'rival'.
#   - Oferta simple (Eizenberg): Z = X => exactamente identificado => OLS, omega=residuo.
if ESTIMATE_CONDUCT:
	Z_list = list(X_list) + product_iv
else:
	Z_list = list(X_list)

X = df[X_list].to_numpy()
Z = df[Z_list].to_numpy()
N = X.shape[0]

###############################################################################
# 3. GMM lineal concentrado en gamma (comun a ambos modos)
###############################################################################

def concentrate_gamma(y, W):
	"""Dado y (mc o log mc) y matriz de pesos W, estima gamma por GMM/IV lineal y
	   devuelve gamma y el error estructural omega = y - X gamma.
	   Si Z = X (exactamente identificado) esto coincide con OLS para cualquier W."""
	XZ    = X.T @ Z                       # (k x L)
	A     = XZ @ W @ XZ.T                 # (k x k)
	b     = XZ @ W @ (Z.T @ y)            # (k,)
	gamma = np.linalg.solve(A, b)
	omega = y - X @ gamma
	return gamma, omega

###############################################################################
# 4A. Estimacion de conducta (solo si ESTIMATE_CONDUCT = True)
###############################################################################

if ESTIMATE_CONDUCT:

	def gmm_objective(kappa_arr, W):
		kappa    = kappa_arr[0]
		y, _     = cost_lhs(kappa)
		_, omega = concentrate_gamma(y, W)
		g        = Z.T @ omega / N        # condicion de momento (L,)
		return float(g @ W @ g)

	# Etapa 1: peso tipo 2SLS  W0 = (Z'Z)^{-1}
	W0   = inv(Z.T @ Z)
	res1 = minimize(gmm_objective, x0=[0.5], args=(W0,),
					method='L-BFGS-B', bounds=[(0.0, 1.0)],
					options={'ftol': 1e-12})
	kappa1 = res1.x[0]

	# Peso optimo  W1 = S^{-1},  S = (1/N) Z' diag(omega^2) Z
	y1, _     = cost_lhs(kappa1)
	_, omega1 = concentrate_gamma(y1, W0)
	Zo        = Z * omega1[:, None]
	S         = (Zo.T @ Zo) / N
	W1        = inv(S)

	# Etapa 2: re-optimiza con el peso eficiente
	res2  = minimize(gmm_objective, x0=[kappa1], args=(W1,),
					 method='L-BFGS-B', bounds=[(0.0, 1.0)],
					 options={'ftol': 1e-12})
	kappa = res2.x[0]

	# SE sandwich (theta = (gamma, kappa)); derivada de la LHS respecto a kappa numerica
	eps        = 1e-5
	lhs_p, _   = cost_lhs(kappa + eps)
	lhs_m, _   = cost_lhs(kappa - eps)
	dlhs       = (lhs_p - lhs_m) / (2 * eps)
	G          = Z.T @ np.column_stack([-X, dlhs]) / N      # (L x (k+1))
	Avar       = inv(G.T @ W1 @ G)
	se         = np.sqrt(np.diag(Avar) / N)
	se_kappa   = se[-1]
	se_gamma   = se[:-1]

	W_final = W1

###############################################################################
# 4B. Oferta simple: conducta FIJA, solo recupera omega (Eizenberg 2014)
###############################################################################

else:

	kappa = KAPPA_FIXED
	se_kappa = np.nan                      # no se estima: es un supuesto

	# Con Z = X esto es OLS; con instrumentos excluidos seria 2SLS.
	W_id      = inv(Z.T @ Z)
	y, mc_lev = cost_lhs(kappa)
	gamma_hat, omega_hat = concentrate_gamma(y, W_id)

	# SE robustas a heterocedasticidad (White; con Z=X reduce a la robusta de OLS).
	Zo       = Z * omega_hat[:, None]
	S        = (Zo.T @ Zo) / N
	G        = -(Z.T @ X) / N
	Avar     = inv(G.T @ inv(S) @ G)
	se_gamma = np.sqrt(np.diag(Avar) / N)

	W_final = W_id

###############################################################################
# 5. Recupera gamma/omega finales, guarda y reporta
###############################################################################

y_final, mc_level = cost_lhs(kappa)
gamma_hat, omega_hat = concentrate_gamma(y_final, W_final)

df['mc']    = mc_level     # costo marginal en niveles (pisado si hubo mc<=0 con logs)
df['omega'] = omega_hat    # shock estructural de costo (en logs si LOG_MC=True)

# Marca de productos cuyo mc fue pisado (mc<=0 originalmente). Util para diagnostico
# y para excluirlos del pool de resampling aguas abajo (Eizenberg / costos fijos).
if LOG_MC and FLOOR_NEGATIVE_MC:
	_, _floored = apply_mc_floor(implied_mc(kappa))
else:
	_floored = np.zeros(N, dtype=bool)
df['mc_floored'] = _floored
_nfl = int(_floored.sum())

_units = "log(mc)" if LOG_MC else "mc"

print("\n" + "=" * 56)
if ESTIMATE_CONDUCT:
	print("  Estimacion de CONDUCTA (GMM, 2 etapas)")
else:
	print("  Estimacion de OFERTA simple (conducta fija) -> omega")
print("=" * 56)
print(f"  Ecuacion de costo : {_units} = X gamma + omega")
if ESTIMATE_CONDUCT:
	print(f"  kappa (estimado)  = {kappa:8.4f}   se = {se_kappa:8.4f}")
	print(f"  t-stat (H0: kappa=0) = {kappa / se_kappa:6.2f}")
else:
	print(f"  kappa (SUPUESTO)  = {kappa:8.4f}   "
		  f"({'Bertrand-Nash' if kappa == 0 else 'conducta fija'})")
print("-" * 56)
print("  Cost shifters (gamma):")
for name, b, s in zip(X_list, gamma_hat, se_gamma):
	print(f"    {name:22s} {b:9.4f}  (se {s:7.4f})")
print("-" * 56)
print(f"  omega: media {omega_hat.mean():8.4f}  sd {omega_hat.std():8.4f}  "
	  f"(N={N})  -> guardado en df['omega']")
if LOG_MC and FLOOR_NEGATIVE_MC:
	print(f"  mc pisados (mc<=0 -> piso): {_nfl}/{N} ({_nfl/N:.1%})"
		  + ("   <- ALTO: la demanda implica markup>precio; el piso contamina gamma/omega."
			 if _nfl / N > 0.02 else "   (OK si es chico)"))
print("=" * 56)

if SAVE_OUTPUT:
	out = [df, gamma_hat, kappa, X_list, LOG_MC]
	with open('informacion_oferta.obj', 'wb') as f:
		pd.to_pickle(out, f)
	print("  Guardado: informacion_oferta.obj  [df, gamma, kappa, X_list, LOG_MC]")

######################################################
# Para Eizenberg (2014):
#   - df['omega'] son los shocks de costo de los productos OFRECIDOS, bajo la
#     conducta SUPUESTA (Bertrand-Nash). Esa es la materia prima de la etapa de
#     utilidades variables / costos fijos (BEW): con (gamma, omega, demanda) fijos
#     se re-resuelve el juego de precios bajo portafolios contrafactuales y se
#     construyen las cotas de costo fijo.
#   - Recordar: productos ofrecidos dan cotas superiores; no ofrecidos, inferiores
#     (alli omega no se observa: seleccion). Los supuestos 2-3 de Eizenberg
#     restauran los momentos incondicionales via el soporte de la utilidad variable.
######################################################
