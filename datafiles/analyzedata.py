import sys, os
sys.path.insert(0, '/Users/anjli/Documents/Research/PourbaixHull/gitpymatgen_beta')
from itertools import combinations
import numpy as np
from sympy import symbols, lambdify
from sympy.functions.elementary.piecewise import Piecewise
from random import uniform, seed # delete later; only for fn generation
from pymatgen.core.composition import Composition
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.pourbaix_diagram import PourbaixEntry, PourbaixDiagram, PourbaixPlotter
from pymatgen import MPRester
import pandas as pd
import glob
import pickle
from scipy.signal import convolve2d

### AP Pourbaix Functions - START

# elements as array
el = np.array("\
F,O,N,Cl,Br,I,Rb,H,Cs,K,Ag,Li,Na,Be,Hg,Au,\
Ba,Zn,Cd,Ca,B,Cu,Yb,Tl,Mg,Pd,Ni,Sr,Co,C,Cr,\
Ga,Eu,In,Mn,Al,Pt,Si,Pb,La,Rh,Sc,Nd,Ho,Tb,Dy,\
Fe,W,Os,Er,Sm,Ge,Lu,Gd,Pr,Ir,V,Ru,Tm,Th,Y,Ti,Pu,\
Bi,As,Hf,S,Te,U,Ce,Se,Zr,Ta,Sn,Mo,Re,Nb,P,Sb,Np".split(","))

# elements as integers
F,O,N,Cl,Br,I,Rb,H,Cs,K,Ag,Li,Na,Be,Hg,Au,\
Ba,Zn,Cd,Ca,B,Cu,Yb,Tl,Mg,Pd,Ni,Sr,Co,C,Cr,\
Ga,Eu,In,Mn,Al,Pt,Si,Pb,La,Rh,Sc,Nd,Ho,Tb,Dy,\
Fe,W,Os,Er,Sm,Ge,Lu,Gd,Pr,Ir,V,Ru,Tm,Th,Y,Ti,Pu,\
Bi,As,Hf,S,Te,U,Ce,Se,Zr,Ta,Sn,Mo,Re,Nb,P,Sb,Np = range(len(el))

# elements as dict
eld = {name:i for (name,i) in zip(el,range(len(el)))}

def comptocc(el, dictx):
    ''' Convert mesh grid of element compositions into cc entries for composition energy predictor
    '''
    ccx = np.zeros((1,len(el)), dtype=object)

    for elementname in dictx.keys():
        ccx[0][eld[elementname]] = dictx[elementname]

    return ccx

def removeelement(elementlist, removesymbol):
    if removesymbol in elementlist:
        elementlist.remove(removesymbol)
    return elementlist

def stabilitylimit(stability_filter_matrix, axisrange, arglist, axis):
    limitval = 0.0
    weightcount = 0

    if axis == 'V':
        weightaxis = 1
    elif axis == 'pH':
        weightaxis = 0
    else:
        raise Exception

    for ind in arglist:
        weight = np.absolute(stability_filter_matrix[ind[0]][ind[1]])
        axisind = ind[weightaxis]
        limitval += axisrange[axisind][axisind]*weight
        weightcount += weight

    if weightcount > 0.0:
        limitval = limitval/float(weightcount)
    else:
        limitval = -100

    return limitval

def stabilitymetrics(stability_matrix, pH, V):
    horz_filter = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    vert_filter = np.transpose(horz_filter)

    vert_stability = convolve2d(stability_matrix, vert_filter, mode='same')
    horz_stability = convolve2d(stability_matrix, horz_filter, mode='same')

    stability_area = np.count_nonzero(stability_matrix)

    minV_arg = np.argwhere(horz_stability > 0)
    minV_lim = stabilitylimit(horz_stability, V, minV_arg, 'V')
    maxV_arg = np.argwhere(horz_stability < 0)
    maxV_lim = stabilitylimit(horz_stability, V, maxV_arg, 'V')

    minpH_arg = np.argwhere(vert_stability > 0)
    minpH_lim = stabilitylimit(vert_stability, pH, minpH_arg, 'pH')
    maxpH_arg = np.argwhere(vert_stability < 0)
    maxpH_lim = stabilitylimit(vert_stability, pH, maxpH_arg, 'pH')

    return {'area': stability_area, 'V_lim': (minV_lim, maxV_lim), 'pH_lim': (minpH_lim, maxpH_lim), 'vert_matrix': vert_stability, 'horz_matrix': horz_stability}

def addsolid(datadict, solid, ellist):
    datadict['formula'].append(solid.composition.reduced_formula)
    datadict['energy per atom'].append(solid.energy_per_atom)
    datadict['entry'].append(solid)
    compositiondict = solid.composition.fractional_composition.as_dict()

    if 'O' in compositiondict.keys():
        datadict['%O'].append(compositiondict['O']*100)
    else:
        datadict['%O'].append(0)
    if 'H' in compositiondict.keys():
        datadict['%H'].append(compositiondict['H']*100)
    else:
        datadict['%H'].append(0)

    for el in ellist:
        if el in compositiondict.keys():
            datadict['%{}'.format(el)].append(compositiondict[el]*100)
            datadict['%{}_noOH'.format(el)].append(100*compositiondict[el]/(1.0 - compositiondict['O'] - compositiondict['H']))
        else:
            datadict['%{}'.format(el)].append(0.0)
            datadict['%{}_noOH'.format(el)].append(0.0)
    return datadict

def evalcomp(entry, comp, ellist):
    entrycomp = entry.composition.fractional_composition.as_dict()
    if 'O' not in entrycomp.keys():
        entrycomp['O'] = 0.0
    if 'H' not in entrycomp.keys():
        entrycomp['H'] = 0.0

    # get all non-OH compositions in a new dictionary for entry
    entrycomp_minusOH = dict()
    for compel in ellist:
        if compel in entrycomp.keys():
            entrycomp_minusOH[compel] = entrycomp[compel]/(1.0 - entrycomp['O'] - entrycomp['H'])
        else:
            entrycomp_minusOH[compel] = 0.0

    # check if all non-OH compositions of entry match current comp dict
    if comp != None:
        newcomp = False
        for el in ellist:
            if el in comp.keys():
                if entrycomp_minusOH[el] != comp[el]:
                    newcomp = True
            elif entrycomp_minusOH[el] > 0:
                newcomp = True
    else:
        newcomp = True
    if newcomp == True:
        newcompdict = dict()
        for entryel in entrycomp.keys():
            if entryel != 'O' and entryel != 'H':
                newcompdict[entryel] = entrycomp_minusOH[entryel]
        comp = newcompdict
    return (newcomp, comp)

def getentrystability(entry, stabilitydata, ehullmax, newcomp, plotterobj = None, compdict = None, all_entries = None):
    if newcomp == True:
        if compdict == None or all_entries == None:
            raise Exception('Need to provide all_entries and compdict when using new composition!')
        pbx = PourbaixDiagram(all_entries, comp_dict = compdict)
        plotterobj = PourbaixPlotter(pbx)
    elif plotterobj == None:
        raise Exception('Need to provide plotter object if using previous composition!')
    
    stability, (pH, V) = plotterobj.get_entry_stability(entry, e_hull_max=ehullmax)
    stability_metrics = stabilitymetrics(stability, pH, V)

    return (stability_metrics, plotterobj)


### AP Pourbaix Functions - END

##################################################

### Start here:
oxidesonly = False
e_hull_max = 1.0

entrylist = glob.glob('entries_*.pickle')
fulllist = list()
for entry in entrylist:
    el1 = entry[entry.find('_')+1 : entry.find('-')]
    el2 = entry[entry.find('-')+1 : entry.find('.')]
    fulllist.append(el1)
    fulllist.append(el2)
fulllist = set(fulllist)
# Initialize stability data dict:
datadict = dict()
datadict['formula'] = list()
datadict['energy per atom'] = list()
for el in fulllist:
    datadict['%{}'.format(el)] = list()
    datadict['%{}_noOH'.format(el)] = list()
datadict['%O'] = list()
datadict['%H'] = list()
datadict['entry'] = list()

for combo in entrylist:
    all_entries = pickle.load(open(combo, 'rb'))
    el1 = combo[combo.find('_')+1 : combo.find('-')]
    el2 = combo[combo.find('-')+1 : combo.find('.')]
    elementlist = [el1, el2]

    newcomp = True
    comp = None

    ### Sort solids containing all elements of interest:
    for entry in all_entries:
        if entry.entry_id[:3] != 'ion':
            entryelements = entry.composition.elements
            entryelementlist = list()
            for entrye in entryelements:
                entryelementlist.append(entrye.symbol)
            if oxidesonly == True:
                # For all oxides containing all elements:
                if 'O' in entryelementlist:
                    entryelementlist = removeelement(entryelementlist, 'O')
                    entryelementlist = removeelement(entryelementlist, 'H')
                    if len(entryelementlist) == len(elementlist):
                        datadict = addsolid(datadict, entry, fulllist)
            else:
                # For all solids containing all elements:
                entryelementlist = removeelement(entryelementlist, 'O')
                entryelementlist = removeelement(entryelementlist, 'H')
                if len(entryelementlist) == len(elementlist):
                    datadict = addsolid(datadict, entry, fulllist)

numentries = len(datadict['formula'])
datadict['V_min'] = np.zeros(numentries)
datadict['V_max'] = np.zeros(numentries)
datadict['pH_min'] = np.zeros(numentries)
datadict['pH_max'] = np.zeros(numentries)
datadict['area'] = np.zeros(numentries)

# Create data frame:
stabilitydata = pd.DataFrame(datadict)

# Filter out repeat entries:
stabilitydata = stabilitydata.sort_values(by='energy per atom').groupby('formula').head(1)

# Get stability values for each row:
newcomp = True
comp = None
count = 0

totentries = stabilitydata.shape[0]
print('Number of Entries: {}'.format(totentries))

for index, row in stabilitydata.iterrows():
    count += 1
    if count % 1000 == 0:
        pickle.dump(stabilitydata, open('total_stability_df_{}.pickle'.format(count),'wb'))
    print(row['formula'])
    print('Entry {}/{}'.format(count, totentries))
    entry = row['entry']
    newcomp, comp = evalcomp(entry, comp, fulllist)
    if newcomp == True:
        thisellist = list()
        for el in fulllist:
            if row['%{}'.format(el)] > 0.0:
                thisellist.append(el)
        try:
            all_entries = pickle.load(open('entries_{}-{}.pickle'.format(thisellist[0], thisellist[1]), 'rb'))
        except:
            all_entries = pickle.load(open('entries_{}-{}.pickle'.format(thisellist[1], thisellist[0]), 'rb'))
        stability_metrics, plotterobj = getentrystability(entry,  stabilitydata, e_hull_max, newcomp, all_entries = all_entries, compdict = comp)
    else:
        stability_metrics, plotterobj = getentrystability(entry,  stabilitydata, e_hull_max, newcomp, plotterobj = plotterobj)

    stabilitydata.at[index, 'area'] = stability_metrics['area']
    stabilitydata.at[index, 'V_min'], stabilitydata.at[index, 'V_max'] = stability_metrics['V_lim']
    stabilitydata.at[index, 'pH_min'], stabilitydata.at[index, 'pH_max'] = stability_metrics['pH_lim']

pickle.dump(stabilitydata, open('total_stability_df.pickle','wb'))

