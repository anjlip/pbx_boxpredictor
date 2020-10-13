import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from copy import deepcopy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats

''' Compute new descriptors (more general) and get correlations wrt outputs '''

# Functions:
def getlogitp(model, x, y):
    sse = np.sum((model.predict(x) - y)**2.0, axis=0)/float(x.shape[0] - x.shape[1])
    #sse = (model.predict(x) - y)**2.0
    #print(sse)
    #print('hi')
    se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(x.T, x))))])
    #se = np.array([np.sqrt(np.diagonal(sse[i]*np.linalg.inv(np.dot(x.T, x)))) for i in range(sse.shape[0])])
    tval = model.coef_/se
    pval = 2*(1-stats.t.cdf(np.abs(tval), y.shape[0] - x.shape[1]))
    return pval

# Load data:
data = pickle.load(open('fullstabilitydf.pickle', 'rb'))

# Filter out no stability area:
data = data[data['V_min'] != -100]


# Element property data:
#elementlist = ['Ba', 'Ag', 'Nd', 'Fe', 'Mg', 'Ir', 'Mo', 'W', 'Y', 'Os', 'Li', 'Na', 'Tc', 'P', 'Co', 'K', 'Sr', 'Au', 'Zn', 'Rh', 'Cu', 'Ni', 'Bi', 'Pt', 'N', 'Gd', 'S', 'Re', 'Mn', 'Cr', 'Ca', 'Pd', 'Ru', 'O', 'H']

propdf = pd.read_csv('elementproperties.csv')


# Rename problematic columns:
regresscols = ['O', 'H']
metalslist = list()
for i in data.columns:
    if '%' in i:
        if 'noOH' in i:
            newname = i[1:i.find('_')]
            data.rename(columns={i:newname}, inplace=True)
            regresscols.append(newname)
            metalslist.append(newname)
        elif i in ['%O', '%H']:
            newname = i[1:]
            data.rename(columns={i:newname}, inplace=True)
        else:
            data.rename(columns={i:i[1:]+'_full'}, inplace=True)
    elif i == 'energy per atom':
        data.rename(columns={i:'energy_per_atom'}, inplace=True)

# Filter oxides:
oxidesonly = True
if oxidesonly == True:
    data = data[data['O'] > 0]

# Get columns of interest:
outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom', 'O']

'''
# Initialize slope result dictionary: (primary element corresponds to correlations, secondary element is other element in system (or na if O is primary element))
slopedict = dict()
slopedict['primary_element'] = list()
slopedict['secondary_element'] = list()
for i in outcols:
    slopedict[i] = list()

# Coefficient of Determination:
count = 0
for el1 in metalslist:
    count += 1
    print('element {} of {}'.format(count, len(metalslist)))
    secondmetalslist = deepcopy(metalslist)
    secondmetalslist.remove(el1)

    for el2 in secondmetalslist:
        subdata = data[data[el1] > 0.0]
        subdata = subdata[subdata[el2] > 0.0]

        # Add first metal stats:
        slopedict['primary_element'].append(el1)
        slopedict['secondary_element'].append(el2)
        outputlist = list()
        for i in range(len(outcols)):
            result = subdata[[el1]].corrwith(subdata[outcols[i]])[0]
            if result > 0.0:
                slopedict[outcols[i]].append(0)
            elif result < 0.0:
                slopedict[outcols[i]].append(1)
            else:
                slopedict[outcols[i]].append(np.nan)

slopedata = pd.DataFrame(slopedict)
slopedata.dropna(inplace=True)

# Add property columns to slope dataframe:
propdf = propdf.rename(index=propdf['element']) # rename indexes to elements
propdf = propdf.drop(columns=['element'])

# Get rid of nan values in propdf by replacing with mean:
propdf.fillna(propdf.mean(), inplace=True)

# Note: example for getting index of Pd row: a.index[a['element']=='Pd'].tolist()[0]
for col in propdf.columns:
    slopedata['mean_{}'.format(col)] = np.nan
    slopedata['diff_{}'.format(col)] = np.nan

for index, row in slopedata.iterrows():
    for col in propdf.columns:
        slopedata.at[index, 'mean_{}'.format(col)] = propdf.at[row['primary_element'], col]
        slopedata.at[index, 'diff_{}'.format(col)] = propdf.at[row['primary_element'], col]

pickle.dump(slopedata, open('binarycorrdata.pickle', 'wb'))
'''

slopedata = pickle.load(open('binarycorrdata.pickle', 'rb'))
print(slopedata)

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

    meanlm = LogisticRegression(random_state=0)
    meanmodel = meanlm.fit(trainX, trainy)
    meanpredictions = meanlm.predict(testX)
    meanpredictionstrain = meanlm.predict(trainX)
    meanresults[output]['score'] = meanmodel.score(testX, testy)
    meanresults[output]['coeffs'] = meanmodel.coef_
    meanresults[output]['train score'] = meanmodel.score(trainX, trainy)
    meanresults[output]['train f1 score'] = f1_score(trainy, meanpredictionstrain)
    meanresults[output]['test f1 score'] = f1_score(testy, meanpredictions)
    #meanresults[output]['p values'] = getlogitp(meanmodel, trainX.to_numpy, trainy.to_numpy)
    meanresults[output]['p values'] = getlogitp(meanmodel, trainX.values, trainy.values)

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
plt.legend(fontsize = 15).draggable()
plt.plot([-0.5, len(proplist)*2 + barwidth*count + 0.5], [0, 0], 'k-', alpha = 0.5)
plt.tight_layout()

# Diff descriptors:
diffresults = dict()
trainX = traindata[diffproplist]
testX = testdata[diffproplist]
for output in outcols:
    diffresults[output] = dict()
    trainy = traindata[output]
    testy = testdata[output]

    difflm = LogisticRegression(random_state=0)
    diffmodel = difflm.fit(trainX, trainy)
    diffpredictions = difflm.predict(testX)
    diffresults[output]['score'] = diffmodel.score(testX, testy)
    diffresults[output]['coeffs'] = diffmodel.coef_

print('Diff model:')
print(proplist)
print(diffresults)

### End binary classification

plt.show()

