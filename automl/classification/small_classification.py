import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
from copy import deepcopy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import sklearn.metrics
from tpot import TPOTClassifier

# Get columns of interest:
outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom', 'O']

#slopedata = pickle.load(open('binarycorrdata_pvals.pickle', 'rb'))
slopedata = pickle.load(open('binarycorrdata.pickle', 'rb'))

# Split training and test:
traindata, testdata = train_test_split(slopedata, test_size=0.1, random_state=10)

### Start binary classification::
proplist = ['electronegativity', 'valency', 'dband', 'atomic_radius', 'ionization_energy', 'oxidation_state', 'standard_potential', 'atomic_mass', 'covalent_radius']
outcols = outcols[:-1]

meanproplist = list()
diffproplist = list()

for i in proplist:
    meanproplist.append('mean_{}'.format(i))
    diffproplist.append('diff_{}'.format(i))

# Mean descriptors:
meanresults = dict()
trainX = traindata[meanproplist]
testX = testdata[meanproplist]

for output in outcols:
    meanresults[output] = dict()
    trainy = traindata[output]
    testy = testdata[output]

    pipeline_optimizer = TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=1)
    pipeline_optimizer.fit(trainX, trainy)

    print(output)
    print(pipeline_optimizer.score(testX, testy))

print('Mean model:')
print(proplist)
print(meanresults)

# Plot coefficients:
plt.figure()
barwidth = 0.3
count = 0
plotvals = ['V_min', 'V_max', 'pH_min', 'pH_max']
for i in range(len(plotvals)):
    count += 1
    values = meanresults[plotvals[i]]['coeffs'][0]
    print(values)
    plt.bar(np.arange(len(values))*2 + barwidth*count, values, barwidth, edgecolor = 'k', label = plotvals[i])

plt.xticks(np.arange(len(proplist))*2 + barwidth*count*0.5, labels=proplist, fontsize = 13, rotation=90)
plt.yticks(fontsize = 13)
plt.ylabel('Coefficient', fontsize = 15)
#plt.legend(fontsize = 15).draggable()
plt.legend(fontsize = 15)
plt.plot([-0.5, len(proplist)*2 + barwidth*count + 0.5], [0, 0], 'k-', alpha = 0.5)
plt.tight_layout()

### End binary classification

plt.show()

