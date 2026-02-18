"""
================================================================================
 ESTIMACIÓN DE DEMANDA BLP CON DATOS DE CEREALES DE NEVO
 Organización Industrial Empírica — Magíster en Análisis Económico
================================================================================
 Referencia principal:
   Nevo, A. (2000). "A Practitioner's Guide to Estimation of Random-Coefficients
   Logit Models of Demand." JEMS 9(4), 513–548.

 Implementación:
   Conlon, C. & Gortmaker, J. (2020). "Best Practices for Differentiated
   Products Demand Estimation with pyblp." RAND J. Econ. 51(4), 1108–1161.
   Documentación: https://pyblp.readthedocs.io

 Datos incorporados en pyblp:
   - NEVO_PRODUCTS_LOCATION : datos de productos (24 marcas, 94 ciudades)
   - NEVO_AGENTS_LOCATION   : datos de agentes/consumidores simulados

 Estructura del script:
   0. Opciones y librerías
   1. Cargar y explorar datos
   2. PARTE A — Logit Simple (benchmark)
   3. PARTE B — RC Logit sin demograficos (Monte Carlo)
   4. PARTE C — RC Logit sin demograficos (cuadratura exacta, como Nevo)
   5. PARTE D — RC Logit CON demograficos (replicación Nevo completa)
   6. POST-ESTIMACIÓN: elasticidades, costos, markups
   7. Visualización de resultados
================================================================================
"""

# ==============================================================================
# 0.  LIBRERÍAS Y OPCIONES
# ==============================================================================
import pyblp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Opciones globales de pyblp
pyblp.options.verbose = True   # Mostrar progreso de estimacion
pyblp.options.digits  = 3      # Decimales en tablas de resultados

print(f"pyblp version: {pyblp.__version__}")
print("=" * 60)


# ==============================================================================
# 1.  CARGAR Y EXPLORAR DATOS
# ==============================================================================

# ──────────────────────────────────────────────────────────────
# 1.1  Datos de productos
# ──────────────────────────────────────────────────────────────
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

print("\n=== DATOS DE PRODUCTOS ===")
print(f"Filas: {len(product_data)}")
print(f"Mercados únicos: {product_data['market_ids'].nunique()}")
print(f"Productos únicos: {product_data['product_ids'].nunique()}")
print(f"Productos por mercado: {len(product_data) / product_data['market_ids'].nunique():.1f} promedio")
print("\nPrimeras filas:")
print(product_data.head())

print("\nColumnas disponibles:")
print(product_data.columns.tolist())

# Descripción estadística de variables clave
print("\nEstadísticas descriptivas:")
print(product_data[['shares', 'prices', 'sugar', 'mushy']].describe())

# Verificar que los shares sumen menos de 1 por mercado (outside good > 0)
share_check = product_data.groupby('market_ids')['shares'].sum()
print(f"\nShare total máximo en algún mercado: {share_check.max():.4f} (debe ser < 1)")
print(f"Outside good mínimo: {(1 - share_check).min():.4f} (debe ser > 0)")

# ──────────────────────────────────────────────────────────────
# 1.2  Datos de agentes (consumidores simulados con demograficos)
# ──────────────────────────────────────────────────────────────
agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)

print("\n=== DATOS DE AGENTES ===")
print(f"Filas: {len(agent_data)}")
print(agent_data.head())
print("\nColumnas:")
print(agent_data.columns.tolist())

# Los agentes tienen:
#   market_ids  : a qué mercado pertenecen
#   weights     : pesos de cuadratura (suman 1 por mercado)
#   nodes*      : nodos de integración para los coef. aleatorios
#   income*     : datos demográficos (ingreso, etc.)


# ==============================================================================
# 2.  PARTE A — LOGIT SIMPLE (BENCHMARK)
# ==============================================================================
# El Logit Multinomial asume coeficientes homogéneos: todos los consumidores
# valoran igual cada característica. La elasticidad propia es proporcional al
# precio y el share de mercado, la cruzada es idéntica para todos los rivales
# (propiedad IIA — Independence of Irrelevant Alternatives).
# ==============================================================================

print("\n" + "=" * 60)
print("PARTE A: LOGIT SIMPLE")
print("=" * 60)

# Formulación: solo precio en X1, sin coeficientes aleatorios (X2 vacío)
# absorb='C(product_ids)' incluye efectos fijos de producto (captura la
# utilidad media de cada marca, como en Nevo 2000)
logit_X1 = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
logit_formulations = (logit_X1,)   # Tupla con solo X1

logit_problem = pyblp.Problem(logit_formulations, product_data)
print(logit_problem)

# Resolver: no hay parámetros no lineales que inicializar
logit_results = logit_problem.solve()
print(logit_results)

# Guardar el coeficiente de precio del logit (referencia)
alpha_logit = logit_results.beta[0]
print(f"\nCoef. de precio (Logit): {alpha_logit:.4f}")


# ==============================================================================
# 3.  PARTE B — RC LOGIT SIN DEMOGRAFICOS (MONTE CARLO)
# ==============================================================================
# Agregamos heterogeneidad en preferencias mediante coeficientes aleatorios.
# Aquí usamos Monte Carlo para aproximar la integral sobre los shocks de gusto.
# Nota: Monte Carlo tiene error estocástico → resultados varían con la semilla.
# ==============================================================================

print("\n" + "=" * 60)
print("PARTE B: RC LOGIT — MONTE CARLO (sin demograficos)")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 3.1  Formulaciones
# ──────────────────────────────────────────────────────────────
# X1: precio con EF de producto (coeficiente lineal)
# X2: intercepto + precio + azúcar + mushy (coeficientes ALEATORIOS)
#     El intercepto en X2 captura heterogeneidad general en la utilidad.

X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
product_formulations = (X1_formulation, X2_formulation)

print("Formulaciones:")
print(f"  X1: {X1_formulation}")
print(f"  X2: {X2_formulation}")

# ──────────────────────────────────────────────────────────────
# 3.2  Integración Monte Carlo
# ──────────────────────────────────────────────────────────────
# size=50 → 50 draws por mercado. Para producción usar ≥ 500 o cuadratura.
mc_integration = pyblp.Integration('monte_carlo', size=50,
                                    specification_options={'seed': 0})
print(f"\nIntegración: {mc_integration}")

# ──────────────────────────────────────────────────────────────
# 3.3  Construir el Problema
# ──────────────────────────────────────────────────────────────
mc_problem = pyblp.Problem(product_formulations, product_data,
                            integration=mc_integration)
print(f"\n{mc_problem}")

# El print() del Problem muestra:
#   - Número de mercados, productos, agentes simulados
#   - Dimensiones de X1, X2
#   - Número de IVs de demanda (debe ser >= # de params no lineales)

# ──────────────────────────────────────────────────────────────
# 3.4  Valores iniciales (Sigma)
# ──────────────────────────────────────────────────────────────
# Sigma es la matriz de Cholesky de la covarianza de los coef. aleatorios.
# Las entradas no-cero indican qué coef. tienen heterogeneidad no observada.
# Aquí solo el intercepto tiene un RC (como punto de partida).

# Sigma (4x4): intercepto, precio, azucar, mushy
# Solo diagonal → coeficientes aleatorios independientes
sigma_0 = np.diag([0.3, 0.0, 0.0, 0.0])

print("\nValores iniciales de Sigma:")
print(sigma_0)

# ──────────────────────────────────────────────────────────────
# 3.5  Resolver con GMM de dos etapas
# ──────────────────────────────────────────────────────────────
# method='2s'  → GMM de dos etapas (primero usa W=I, luego la eficiente)
# Optimization → L-BFGS-B (gradiente analítico, eficiente para BLP)
# Iteration    → SQUAREM: aceleración del operador de contracción de Berry

mc_results = mc_problem.solve(
    sigma        = sigma_0,
    method       = '2s',
    optimization = pyblp.Optimization('l-bfgs-b'),
    iteration    = pyblp.Iteration('squarem'),
)
print(mc_results)


# ==============================================================================
# 4.  PARTE C — RC LOGIT SIN DEMOGRAFICOS (CUADRATURA DE PRODUCTO)
# ==============================================================================
# La cuadratura de producto (Gauss-Hermite) es determinista y más precisa que
# Monte Carlo para el mismo número de nodos. Recomendada para resultados finales.
# ==============================================================================

print("\n" + "=" * 60)
print("PARTE C: RC LOGIT — CUADRATURA DE PRODUCTO (sin demograficos)")
print("=" * 60)

# Cuadratura de Gauss-Hermite con 5 puntos por dimensión
# Para K2 variables en X2, esto genera 5^K2 nodos de integración
pr_integration = pyblp.Integration('product', size=5)
print(f"Integración: {pr_integration}")

pr_problem = pyblp.Problem(product_formulations, product_data,
                            integration=pr_integration)
print(pr_problem)

# Mismos valores iniciales
pr_results = pr_problem.solve(
    sigma        = sigma_0,
    method       = '2s',
    optimization = pyblp.Optimization('l-bfgs-b'),
    iteration    = pyblp.Iteration('squarem'),
)
print(pr_results)


# ==============================================================================
# 5.  PARTE D — RC LOGIT CON DEMOGRAFICOS (REPLICACIÓN NEVO COMPLETA)
# ==============================================================================
# Nevo (2000) usa datos de agentes externos que combinan:
#   - Nodos de cuadratura para los shocks de gusto nu_i
#   - Datos demográficos simulados (ingreso, edad, niños)
# Los coeficientes Pi capturan cómo las características del producto
# interactúan con las características del consumidor.
# ==============================================================================

print("\n" + "=" * 60)
print("PARTE D: RC LOGIT CON DEMOGRAFICOS (Replicación Nevo)")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 5.1  Formulación de agentes
# ──────────────────────────────────────────────────────────────
# Especifica qué columnas demográficas de agent_data se usan en Pi.
# Las columnas 'income', 'income_squared', 'age', 'child' están en agent_data.

agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
print(f"Formulación de agentes: {agent_formulation}")

# ──────────────────────────────────────────────────────────────
# 5.2  Construir el Problema con agentes externos
# ──────────────────────────────────────────────────────────────
# Cuando se usa agent_data, NO se pasa integration (los nodos están en agent_data).
# agent_data contiene columnas:
#   market_ids, weights, nodes0..nodes3, income, income_squared, age, child

nevo_problem = pyblp.Problem(
    product_formulations = product_formulations,
    product_data         = product_data,
    agent_formulation    = agent_formulation,
    agent_data           = agent_data,
)
print(nevo_problem)

# ──────────────────────────────────────────────────────────────
# 5.3  Valores iniciales: Sigma y Pi (de Nevo 2000, Tabla IV)
# ──────────────────────────────────────────────────────────────
# X2 tiene 4 variables: [intercepto, precio, azúcar, mushy]
# Sigma (4x4): desviaciones estándar de los shocks de gusto nu_i
# Pi (4x4)   : interacciones con los 4 demograficos [income, income^2, age, child]

# Sigma inicial (diagonal): solo coeficientes en la diagonal principal
# Entradas cero = ese RC no tiene heterogeneidad no observada en la línea base
sigma_0_nevo = np.diag([0.3302, 2.4526, 0.0163, 0.2441])

# Pi inicial: interacciones caracts. producto x demograficos consumidor
# Filas: [intercepto, precio, azúcar, mushy]
# Cols : [income, income^2, age, child]
pi_0_nevo = np.array([
    [ 2.2912,  0.0000, 1.5848, 0.0000],   # intercepto x demograficos
    [-0.6936, -0.0751, 0.0000, 0.9280],   # precio     x demograficos
    [ 0.0000,  0.0000, 0.0000, 0.0000],   # azúcar     x demograficos
    [ 0.0000,  0.0000, 0.0000, 0.0000],   # mushy      x demograficos
])

print("\nSigma inicial (Nevo):")
print(sigma_0_nevo)
print("\nPi inicial (Nevo):")
print(pi_0_nevo)

# ──────────────────────────────────────────────────────────────
# 5.4  Resolver: GMM de dos etapas
# ──────────────────────────────────────────────────────────────
nevo_results = nevo_problem.solve(
    sigma        = sigma_0_nevo,
    pi           = pi_0_nevo,
    method       = '2s',
    optimization = pyblp.Optimization('l-bfgs-b'),
    iteration    = pyblp.Iteration('squarem'),
)
print(nevo_results)

# ──────────────────────────────────────────────────────────────
# 5.5  Inspeccionar resultados
# ──────────────────────────────────────────────────────────────
print("\n=== RESUMEN DE COEFICIENTES (Nevo con demograficos) ===")
print(f"\nCoef. medios (beta):\n{nevo_results.beta}")
print(f"\nSigma estimado:\n{nevo_results.sigma}")
print(f"\nPi estimado:\n{nevo_results.pi}")
print(f"\nOFV (objetivo GMM): {nevo_results.objective:.6f}")


# ==============================================================================
# 6.  POST-ESTIMACIÓN
# ==============================================================================
# Con el modelo estimado calculamos medidas de demanda e implicaciones de mercado
# ==============================================================================

print("\n" + "=" * 60)
print("POST-ESTIMACIÓN")
print("=" * 60)

# Usar los resultados de la especificación completa (Nevo con demograficos)
res = nevo_results

# ──────────────────────────────────────────────────────────────
# 6.1  Elasticidades precio
# ──────────────────────────────────────────────────────────────
# Retorna una lista de matrices (una por mercado)
# Elemento [j, k] = elasticidad del share_j respecto al precio_k
elasticities = res.compute_elasticities()

# Extraer la diagonal (elasticidades propias)
own_elasticities = res.extract_diagonal(elasticities)

print("\n=== ELASTICIDADES PRECIO ===")
print(f"Elasticidad propia media     : {own_elasticities.mean():.4f}")
print(f"Elasticidad propia mediana   : {np.median(own_elasticities):.4f}")
print(f"Elasticidad propia mín/máx   : {own_elasticities.min():.4f} / {own_elasticities.max():.4f}")

# Comparar con el Logit simple
logit_elast = logit_results.compute_elasticities()
logit_own   = logit_results.extract_diagonal(logit_elast)
print(f"\nLogit simple — elasticidad propia media: {logit_own.mean():.4f}")
print("(El Logit impone patrones IIA, BLP los relaja)")

# ──────────────────────────────────────────────────────────────
# 6.2  Divertissement: matriz de elasticidades en un mercado
# ──────────────────────────────────────────────────────────────
# Ver la primera ciudad como ejemplo
mkt_example = product_data['market_ids'].iloc[0]
idx_mkt     = product_data['market_ids'] == mkt_example
n_prods     = idx_mkt.sum()

# La lista elasticities está ordenada por mercado
market_list     = product_data['market_ids'].drop_duplicates().tolist()
mkt_idx_in_list = market_list.index(mkt_example)
elast_matrix    = elasticities[mkt_idx_in_list]

print(f"\nMatriz de elasticidades — mercado {mkt_example} ({n_prods} productos):")
print(pd.DataFrame(elast_matrix).round(3).to_string())

# ──────────────────────────────────────────────────────────────
# 6.3  Costos marginales y markups (Nash-Bertrand)
# ──────────────────────────────────────────────────────────────
# Bajo el supuesto de competencia Nash-Bertrand:
#   p - c = -(∂s/∂p)^{-1} s   →  c = p + (∂s/∂p)^{-1} s
costs   = res.compute_costs()
markups = res.compute_markups(costs=costs)

# Markup porcentual = (p - c) / p
prices_arr   = product_data['prices'].values
markup_pct   = markups / prices_arr

print("\n=== COSTOS MARGINALES Y MARKUPS (Nash-Bertrand) ===")
print(f"Markup sobre precio medio   : {markup_pct.mean():.4f} ({markup_pct.mean()*100:.1f}%)")
print(f"Markup sobre precio mediano : {np.median(markup_pct):.4f} ({np.median(markup_pct)*100:.1f}%)")
print(f"Costo marginal medio        : {costs.mean():.4f}")

# ──────────────────────────────────────────────────────────────
# 6.4  Concentración de mercado (HHI)
# ──────────────────────────────────────────────────────────────
# pyblp calcula el HHI basado en las participaciones de mercado predichas
hhi = res.compute_hhi()
print(f"\nHHI promedio (mercados)     : {hhi.mean():.1f}")
print(f"HHI mediano                 : {np.median(hhi):.1f}")
print("(HHI > 2500 → mercado muy concentrado según guías DOJ/FTC)")

# ──────────────────────────────────────────────────────────────
# 6.5  Dividir parámetros por producto (summary DataFrame)
# ──────────────────────────────────────────────────────────────
summary = product_data[['market_ids', 'product_ids', 'shares', 'prices']].copy()
summary['own_elasticity'] = own_elasticities
summary['markup']         = markups
summary['cost']           = costs
summary['markup_pct']     = markup_pct * 100

print("\n=== RESUMEN POR PRODUCTO (primeras 10 obs.) ===")
print(summary.head(10).to_string(index=False))


# ==============================================================================
# 7.  VISUALIZACIÓN DE RESULTADOS
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Resultados BLP — Datos de Cereales Nevo (2000)',
             fontsize=13, fontweight='bold')

# ── 7.1 Distribución de elasticidades propias ──────────────────────────────
ax = axes[0]
ax.hist(own_elasticities, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(own_elasticities.mean(), color='red', linestyle='--',
           label=f'Media: {own_elasticities.mean():.2f}')
ax.set_xlabel('Elasticidad precio propia')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Elasticidades Propias')
ax.legend()

# ── 7.2 Precio vs. Markup ──────────────────────────────────────────────────
ax = axes[1]
ax.scatter(prices_arr, markups, alpha=0.5, color='darkorange', s=25)
ax.set_xlabel('Precio')
ax.set_ylabel('Markup (p - c)')
ax.set_title('Precio vs. Markup (Nash-Bertrand)')

# ── 7.3 Shares observados vs. predichos ───────────────────────────────────
ax = axes[2]
predicted_shares = res.compute_shares()
observed_shares  = product_data['shares'].values
ax.scatter(observed_shares, predicted_shares, alpha=0.4, color='seagreen', s=20)
ax.plot([0, observed_shares.max()], [0, observed_shares.max()],
        'k--', linewidth=0.8, label='45°')
ax.set_xlabel('Shares observados')
ax.set_ylabel('Shares predichos')
ax.set_title('Ajuste del Modelo: Shares')
ax.legend()

plt.tight_layout()
plt.savefig('blp_nevo_resultados.png', dpi=150, bbox_inches='tight')
print("\nGráfico guardado en: blp_nevo_resultados.png")
plt.show()


# ==============================================================================
# 8.  TIPS ADICIONALES Y DIAGNÓSTICOS
# ==============================================================================

print("\n" + "=" * 60)
print("DIAGNÓSTICOS DEL MODELO")
print("=" * 60)

# ── 8.1 Verificar convergencia del loop interno ────────────────────────────
print(f"\nEvaluaciones de la contracción (loop interno): "
      f"{nevo_results.contraction_evaluations.sum()}")

# ── 8.2 Gradiente en el mínimo (debe ser ~0) ──────────────────────────────
print(f"Norma del gradiente en el mínimo: {np.linalg.norm(nevo_results.gradient):.6f}")

# ── 8.3 Función objetivo GMM ──────────────────────────────────────────────
print(f"Función objetivo GMM final: {nevo_results.objective:.8f}")

# ── 8.4 Test de sobre-identificación (J-test) ─────────────────────────────
print("\nPara el test J (sobre-identificación), usar:")
print("  nevo_results.run_hansen_j_test()")

# ── 8.5 Guardar resultados a disco ────────────────────────────────────────
nevo_results.to_pickle('blp_nevo_results.pkl')
print("\nResultados guardados en: blp_nevo_results.pkl")
print("Para recuperar: results = pyblp.ProblemResults.from_pickle('blp_nevo_results.pkl')")

# ── 8.6 Tabla comparativa de modelos ──────────────────────────────────────
print("\n=== TABLA COMPARATIVA DE ESPECIFICACIONES ===")
comparison = pd.DataFrame({
    'Modelo': ['Logit Simple', 'RC Logit (MC)', 'RC Logit (Quad)', 'RC + Demog (Nevo)'],
    'Elastic. propia media': [
        logit_own.mean(),
        logit_results.extract_diagonal(logit_results.compute_elasticities()).mean(),
        pr_results.extract_diagonal(pr_results.compute_elasticities()).mean(),
        own_elasticities.mean(),
    ],
    'OFV': [
        logit_results.objective,
        mc_results.objective,
        pr_results.objective,
        nevo_results.objective,
    ],
})
print(comparison.to_string(index=False))

print("\n" + "=" * 60)
print("Script completado exitosamente.")
print("=" * 60)
