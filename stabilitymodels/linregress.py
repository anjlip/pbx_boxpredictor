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

''' Linear regression of stability outputs wrt composition and new descriptors.
    Both mean and difference of new descriptors for each  metal combination are considered.'''

# Load data:
data = pickle.load(open('newdescriptor_stabilitydf.pickle', 'rb'))
# Filter out no stability area:
data = data[data['V_min'] != -100]

# Split training and test:
traindata, testdata = train_test_split(data, test_size=0.1, random_state = 1)

# Fit elemental linear regression:
elementlist = ['Ba', 'Ag', 'Nd', 'Fe', 'Mg', 'Ir', 'Mo', 'W', 'Y', 'Os', 'Li', 'Na', 'Tc', 'P', 'Co', 'K', 'Sr', 'Au', 'Zn', 'Rh', 'Cu', 'Ni', 'Bi', 'Pt', 'N', 'Gd', 'S', 'Re', 'Mn', 'Cr', 'Ca', 'Pd', 'Ru', 'O', 'H']
outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom']
trainX = traindata[elementlist]
trainy = traindata[outcols]
testX = testdata[elementlist]
testy = testdata[outcols]

elementlm = LinearRegression()
elementmodel = elementlm.fit(trainX, trainy)
elementpredictions = elementlm.predict(testX)
elementpredictionstrain = elementlm.predict(trainX)

for col in range(len(outcols)):
    print('Element Fit RMSE ({}): {}'.format(outcols[col], (mean_squared_error(testy[outcols[col]], elementpredictions[:, col]))**0.5))
    print('Element Fit Train RMSE ({}): {}'.format(outcols[col], (mean_squared_error(trainy[outcols[col]], elementpredictionstrain[:, col]))**0.5))

fontsize = 19
bufferval = 0.5

for col in range(len(outcols)):
    plt.figure()
    plt.title(outcols[col] + ' from composition')
    xlim = [testy[outcols[col]].min() - bufferval, testy[outcols[col]].max() + bufferval]
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.plot(xlim, xlim, 'k--', alpha = 0.5)
    plt.plot(testy[outcols[col]], elementpredictions[:,col], marker = 'o', markeredgecolor = 'k', linestyle = 'None')
    label = outcols[col]
    label = label[:label.find('_')+1] + '{' + label[label.find('_')+1:] + '}'
    plt.xlabel('True ' + r'$\rm{}$'.format(label), fontsize = fontsize)
    plt.ylabel('Predicted ' + r'$\rm{}$'.format(label), fontsize = fontsize)
    plt.annotate('RMSE = {:.2f} {}'.format((mean_squared_error(testy[outcols[col]], elementpredictions[:, col]))**0.5, label[:label.find('_')]), xy = (xlim[0] + 0.1, xlim[0] + 0.1), xytext = (xlim[0] + 0.1, xlim[0] + 0.1), fontsize = fontsize).draggable()
    plt.xticks(fontsize = fontsize - 3)
    plt.yticks(fontsize = fontsize - 3)
    plt.tight_layout()

# Note: Correlations of outputs to energy per atom shows this isn't everything

# Fit new descriptor linear regression:
proplist = ['electronegativity', 'valency', 'dband', 'atomic_radius', 'ionization_energy', 'oxidation_state', 'standard_potential', 'atomic_mass', 'covalent_radius']
meanproplist = list()
diffproplist = list()
for i in proplist:
    meanproplist.append('mean_{}'.format(i))
    diffproplist.append('diff_{}'.format(i))

trainX = traindata[meanproplist]
trainy = traindata[outcols]
testX = testdata[meanproplist]
testy = testdata[outcols]

meanlm = LinearRegression()
meanmodel = meanlm.fit(trainX, trainy)
meanpredictions = meanlm.predict(testX)

for col in range(len(outcols)):
    print('Mean New Descriptor Fit RMSE ({}): {}'.format(outcols[col], (mean_squared_error(testy[outcols[col]], meanpredictions[:, col]))**0.5))

for col in range(len(outcols)):
    plt.figure()
    plt.title(outcols[col] + ' from mean descriptors')
    plt.plot(testy[outcols[col]], meanpredictions[:,col], marker = 'o', linestyle = 'None')

trainX = traindata[diffproplist]
trainy = traindata[outcols]
testX = testdata[diffproplist]
testy = testdata[outcols]

difflm = LinearRegression()
diffmodel = difflm.fit(trainX, trainy)
diffpredictions = difflm.predict(testX)

for col in range(len(outcols)):
    print('Diff New Descriptor Fit RMSE ({}): {}'.format(outcols[col], (mean_squared_error(testy[outcols[col]], diffpredictions[:, col]))**0.5))


for col in range(len(outcols)):
    plt.figure()
    plt.title(outcols[col] + ' from diff descriptors')

    plt.plot(testy[outcols[col]], diffpredictions[:,col], marker = 'o', linestyle = 'None')

plt.show()

