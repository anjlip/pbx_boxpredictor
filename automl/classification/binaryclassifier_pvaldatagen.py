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
import statsmodels.api as sm

''' Compute new descriptors (more general) and get correlations wrt outputs (also filters by p-value)'''

# Specify whether we only want transition metals in ternary oxides:
tmonly = True

# Functions:
def getlogitp(model, x, y):
    sse = np.sum((model.predict(x) - y)**2.0, axis=0)/float(x.shape[0] - x.shape[1])
    #sse = (model.predict(x) - y)**2.0
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

# Get columns of interest:
outcols = ['V_min', 'V_max', 'pH_min', 'pH_max', 'area', 'energy_per_atom', 'O']

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
            #result = subdata[[el1]].corrwith(subdata[outcols[i]])[0]
            xdata = subdata[[el1]]
            xdata = sm.add_constant(xdata)
            if len(subdata[outcols[i]]) < 2:
                slopedict[outcols[i]].append(np.nan)
                continue
            linmodel = sm.OLS(subdata[outcols[i]], xdata)
            fitlinmodel = linmodel.fit()
            pvalel = fitlinmodel.pvalues[el1]
            result = fitlinmodel.params[el1]

            if pvalel <= 0.50:
                if result > 0.0:
                    slopedict[outcols[i]].append(0)
                elif result < 0.0:
                    slopedict[outcols[i]].append(1)
                else:
                    slopedict[outcols[i]].append(np.nan)
            else:
                slopedict[outcols[i]].append(np.nan)

slopedata = pd.DataFrame(slopedict)
print(slopedata)

#slopedata.dropna(inplace=True)

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

pickle.dump(slopedata, open('binarycorrdata_pvals.pickle', 'wb'))

