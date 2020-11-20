import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

''' Plot box plots of linear regression coefficients and correlation coefficients per element '''

outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom', 'O']

# Plot correlation coeffs:
slopedata = pickle.load(open('corrdata.pickle','rb'))

# Easy boxplot:
slopedata = slopedata[['V_min','primary_element']]
boxplot = slopedata.boxplot(by='primary_element')
plt.title('Correlation Coeffs')

# Plot linear regression coeffs:
slopedata = pickle.load(open('lincoeffdata.pickle', 'rb'))
fig, axa = plt.subplots(1, 4, figsize=(10,  5), sharey=False)
boxplot = slopedata.boxplot(outcols[:4], 'primary_element', axa)
fig.suptitle('Linear Coeffs')

fig, axb = plt.subplots(1, 3, figsize=(10,  5), sharey=False)
boxplot = slopedata.boxplot(outcols[4:], 'primary_element', axb)
fig.suptitle('Linear Coeffs')

plt.show()
