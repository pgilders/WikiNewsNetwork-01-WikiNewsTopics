#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:36:29 2022

Get Structural Similarity Scores

@author: 
"""

import glob
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import WikiNewsNetwork as wnn

plt.style.use('seaborn-darkgrid')
with open('figures/figurestyle.json', 'r') as f:
    params = json.load(f)
plt.rcParams.update(params)


BPATH = ''

# %% Load data

evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}


wjdone = {x[:-8] for x in glob.glob(BPATH + 'events/*/wjac.h5')}
DD = sorted([x[:-13] for x in glob.glob(BPATH + 'events/*/tcweights.h5')])

allev = sorted(set(DD) - wjdone)


# %% Calculate SS scores

werrs = []
Eerrs = []
step = 400
resrange = np.exp(np.arange(-9, 0.5, 0.5))  # better res???????
for m in range(0, len(allev), step):
    print(m)
    try:

        print('Reading ev data')

        core_D = {}
        G_D = {}
        igname_D = {}

        for n, e in enumerate(allev[m:m+step]):
            if n % 40 == 0:
                print('%.2f %%' % (100*n/len(allev[m:m+step])))
            core_D[e], articles, G_D[e], igname_D[e] = wnn.cd.read_f_data(
                e, rdarts_rev)

        print('Reading weights')

        DDC = {}
        for n, x in enumerate(allev[m:m+step]):
            if n % 40 == 0:
                print('%.2f %%' % (100*n/len(allev[m:m+step])))
            with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
                for k in hdf.keys():
                    DDC[x + '/' + k[1:].replace('/', ':')
                        ] = pd.read_hdf(x+'/tcweights.h5', key=k)

        DDC2 = {}
        for n, (k, v) in enumerate(DDC.items()):
            if k.split('/')[-2] not in DDC2.keys():
                DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
            else:
                DDC2[k.split('/')[-2]][k.split('/')[-1]] = v

        print('Running Community Detection')
        out = Parallel(n_jobs=-1, verbose=10)(delayed(wnn.cd.flat_CD)
                                              (G_D[x], igname_D[x], core_D[x],
                                               resrange)
                                              for x in allev[m:m+step])

        print('Getting Js')
        wws = Parallel(n_jobs=-1, verbose=10)(
            delayed(wnn.cd.evr_matcher)(allev[m+n].split('/')[-1], x[2],
                                        DDC2.get(allev[m+n].split('/')[-1],
                                                 {}),
                                        resrange) for n, x in enumerate(out))

        print('Writing Js')
        for n, x in enumerate(wws):
            try:
                x[0].to_hdf(BPATH + 'events/%s/wjac.h5'
                            % allev[m+n].split('events/')[-1], key='df')
                x[1].to_hdf(BPATH + 'events/%s/jac.h5'
                            % allev[m+n].split('events/')[-1], key='df')
            except Exception as ex:
                print(ex)
                werrs.append([n, x, ex])

    except Exception as ex:
        print(ex)
        Eerrs.append((m, ex))

# %% load df with all SS scores

wjacs = sorted(glob.glob(BPATH + 'events/*/wjac.h5'))
allwj = pd.concat([pd.read_hdf(x).max(axis=1) for x in wjacs])
allwj.to_hdf('support_data/ss_scores.h5', key='df')
# %% Plot SS score distribution
allwj = pd.read_hdf('support_data/ss_scores.h5', key='df')

plt.figure(figsize=[16, 10])
sns.kdeplot(allwj, clip=(0, 1), bw_adjust=0.3, fill=True,
            linewidth=params['lines.linewidth'])
plt.ylim([0, 1.8])
plt.yticks(np.arange(0, 2, 0.25))
plt.title('Structural Similarity Distribution')
plt.xlabel('Structural Similarity')
plt.savefig('figures/ssdist.svg')
plt.savefig('figures/ssdist.pdf')
plt.savefig('figures/ssdist.png')
plt.show()

# %% Plot filtered SS score distribution

# filt_wj = allwj[allwj.index.str.split('/').str[0].isin(evlsn[evlsn['len'] > 100
#                                                              ].index)]
# plt.figure(figsize=[16, 10])
# # sns.kdeplot(allwj, clip=(0, 1), bw_adjust=0.3, fill=True, linewidth=4)
# sns.kdeplot(filt_wj, clip=(0, 1), bw_adjust=0.3, fill=True, linewidth=4)
# plt.ylim([0, 1.8])
# plt.yticks(np.arange(0, 2, 0.25))
# plt.title('Structural Similarity Distribution')
# plt.xlabel('Structural Similarity')
# plt.show()
