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

    pipeline_optimizer = TPOTClassifier(generations=10, population_size=100, verbosity=2, n_jobs=1)
    pipeline_optimizer.fit(trainX, trainy)

    print(output)
    meanresults[output]['test score'] = pipeline_optimizer.score(testX, testy)

print('Mean model:')
print(proplist)
print(meanresults)


# Diff descriptors:
diffresults = dict()
trainX = traindata[diffproplist]
testX = testdata[diffproplist]
for output in outcols:
    diffresults[output] = dict()
    trainy = traindata[output]
    testy = testdata[output]

    pipeline_optimizer = TPOTClassifier(generations=10, population_size=100, verbosity=2, n_jobs=1)
    pipeline_optimizer.fit(trainX, trainy)

    print(output)
    diffresults[output]['test score'] = pipeline_optimizer.score(testX, testy)


print('Diff model:')
print(proplist)
print(diffresults)

### End binary classification

pickle.dump(meanresults, open('mean_tpot_classification.pickle', 'wb'))
pickle.dump(diffresults, open('diff_tpot_classification.pickle', 'wb'))

plt.show()

