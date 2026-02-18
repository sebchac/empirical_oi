# Random Coefficients Logit Tutorial with the Fake Cereal Data

import pyblp
import numpy as np
import pandas as pd

pyblp.options.digits = 2
pyblp.options.verbose = False
pyblp.__version__

# Loading data, setting up and solving the problem withouth demographics

product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.head()

X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
product_formulations = (X1_formulation, X2_formulation)
product_formulations

mc_integration = pyblp.Integration('monte_carlo', size=50, specification_options={'seed': 0})
mc_integration

pr_integration = pyblp.Integration('product', size=5)
pr_integration

mc_problem = pyblp.Problem(product_formulations, product_data, integration=mc_integration)
mc_problem