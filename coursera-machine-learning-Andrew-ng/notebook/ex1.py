# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#


import matplotlib as plt
import matplotlib.style
# %matplotlib inline
import numpy as np

from pathlib import Path
import pandas as pd
from linear_regression_functions import compute_cost

plt.style.use('seaborn-whitegrid')

# ### Identy matrix example

np.identity(5)

# ### Loading data

data_path = Path('..', 'data')
df = pd.read_csv(data_path / 'ex1data1.txt', header=None).rename(index=str, columns={0: 'Population', 1: 'Profit'})
df.head()

# ### Plotting Data

df.plot.scatter(x='Population', y='Profit', c='red', marker='+', s=100, figsize=(5, 3))

# ### Cost Function:

# Testing the cost function

m = df.Population.count()
X = np.ones((m, 2))
X[:, 1::] = df['Population'].values.reshape(df.Population.count(), 1)
y = df.Profit.values.reshape(df.Population.count(), 1)
theta = np.array([0, 0]).reshape(2, 1)
J = compute_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {J}'.format(J=J))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
theta2 = np.array([-1, 2]).reshape(2, 1)
J2 = compute_cost(X, y, theta2)
print('With theta = [-1 ; 2]\nCost computed = {J}'.format(J=J2))
print('Expected cost value (approx) 54.24')
