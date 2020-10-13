import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

''' Linear regression of stability outputs wrt composition and new descriptors '''

# Load data:
data = pickle.load(open('newdescriptor_stabilitydf.pickle', 'rb'))
# Filter out no stability area:
data = data[data['V_min'] != -100]

# Fit new descriptor linear regression:
proplist = ['electronegativity', 'valency', 'dband', 'atomic_radius', 'ionization_energy', 'oxidation_state', 'standard_potential', 'atomic_mass', 'covalent_radius']
meanproplist = list()
diffproplist = list()
for i in proplist:
    meanproplist.append('mean_{}'.format(i))
    diffproplist.append('diff_{}'.format(i))

meandata = pd.DataFrame()
diffdata = pd.DataFrame()

for i in proplist:
    meandata[i] = data['mean_{}'.format(i)]
    diffdata[i] = data['diff_{}'.format(i)]

meanpca = PCA(n_components=3)
meanpca.fit(meandata)
meancomponents = meanpca.components_
meanvariance = meanpca.explained_variance_ratio_
print('Mean PCA:')
print(meancomponents)
print(meanvariance)

# Plot correlations:
meancorr = meandata.corr()
sns.heatmap(meancorr, annot=True, cmap='seismic')

# Plot bar plot of principal component scores:
plt.figure()
barwidth = 0.2
count = 0
for i in range(len(meancomponents)):
    count += 1
    plt.bar(np.arange(len(meancomponents[i]))+barwidth*count, np.array(meancomponents[i]), barwidth)

plt.xticks(np.arange(len(meancomponents[i]))+barwidth*count*0.5, labels = proplist)

# Plot percent variance:
plt.figure()
plt.plot(np.arange(len(meanvariance)), meanvariance, 'bo')

diffpca = PCA(n_components=3)
diffpca.fit(diffdata)
print('Diff PCA:')
print(diffpca.components_)
print(diffpca.explained_variance_ratio_)

stdcorr = diffdata.corr()
#sns.heatmap(stdcorr)

plt.tight_layout()

plt.show()

