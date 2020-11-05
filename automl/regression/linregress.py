import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from tpot import TPOTRegressor

''' Linear regression of stability outputs wrt composition and new descriptors '''

# Specify whether we only want transition metals in ternary oxides:
tmonly = True

# Load data:
data = pickle.load(open('newdescriptor_stabilitydf.pickle', 'rb'))

# Filter out no stability area:
data = data[data['V_min'] != -100]

# Filter out TM:
elementlist = ['Ba', 'Ag', 'Nd', 'Fe', 'Mg', 'Ir', 'Mo', 'W', 'Y', 'Os', 'Li', 'Na', 'Tc', 'P', 'Co', 'K', 'Sr', 'Au', 'Zn', 'Rh', 'Cu', 'Ni', 'Bi', 'Pt', 'N', 'Gd', 'S', 'Re', 'Mn', 'Cr', 'Ca', 'Pd', 'Ru', 'O', 'H']

if tmonly == True:
    elementset = set(elementlist)
    tmlist = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
    tmlist.append('O') # we need O
    tmlist.append('H') # we need H
    tmset = set(tmlist)
    badset = elementset - tmset
    for badel in badset:
        data = data.loc[data[badel] == 0]

# Split training and test:
traindata, testdata = train_test_split(data, test_size=0.1, random_state = 1)

# Fit elemental linear regression:
outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom']
for output in outcols:
    trainX = traindata[elementlist]
    trainy = traindata[output]
    testX = testdata[elementlist]
    testy = testdata[output]

    pipeline_optimizer = TPOTRegressor(generations=20, population_size=100, verbosity=2, n_jobs=1)
    pipeline_optimizer.fit(trainX, trainy)

    testscore = pipeline_optimizer.score(testX, testy)

    print('{} Test Score: {}'.format(output, testscore))

