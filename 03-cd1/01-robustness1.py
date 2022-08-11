#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:27:01 2022

@author: Patrick
"""

import random
import json
from sklearn.metrics import adjusted_mutual_info_score as ami
from joblib import Parallel, delayed
from clusim.clustering import Clustering
import numpy as np
import pandas as pd
import clusim.sim as sim
import matplotlib.pyplot as plt
import functions1 as pgc

# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

evlsn = pd.read_hdf('support_data/evlsN.h5', key='df')

# %% Sample events and read graphs

random.seed(100)
evsample = random.sample(list(evlsn.index), 100)
edf = evlsn.loc[evsample].sort_values('len')
out1 = [pgc.read_ev_data(BPATH + 'events/' + x, rdarts_rev) for x in edf.index]

# %% Load existing data

resresults = {}
for res in np.exp(np.arange(-9, 0.1, 1)):
    print(res)
    try:
        resresults[res] = [pd.read_hdf(BPATH + 'aux_data/res_select.h5',
                                       key='%.5f_%d' % (res, n))
                           for n in range(100)]
    except KeyError:
        print('No ', res)


# %% Run community detection on sample over range of resolutions

for res in np.exp(np.arange(-9, 0.1, 1)):
    print(res)
    if res in resresults:
        continue
    try:

        out2 = Parallel(n_jobs=-1, verbose=10)(delayed(pgc.ev_reactions)(*x,
                                                                         res)
                                               for x in out1 if len(x) == 4)

        for n, x in enumerate(out2):
            x[0].to_hdf(BPATH + 'aux_data/res_select.h5',
                        key='%.5f_%d' % (res, n), mode='a')
        resresults[res] = [x[0] for x in out2]
    except Exception as ex:
        print(ex)

# %% Get similarities between partitions at different resolutions

amis = {}
clusims = {}

for n, e in enumerate(evsample):
    if n % 10 == 0:
        print("%.2f %%" % (100*n/len(evsample)))
    midcom = pd.concat([resresults[r][n][27] for r in
                        np.exp(np.arange(-9, 0.1, 1))], axis=1)
    midcom.columns = np.exp(np.arange(-9, 0.1, 1))
    mc = midcom.columns

    amis[e] = [ami(midcom[[mc[c-1], mc[c]]].dropna()[mc[c-1]],
                   midcom[[mc[c-1], mc[c]]].dropna()[mc[c]])
               for c in range(1, len(midcom.columns))]

    cs = []
    for c in range(1, len(midcom.columns)):
        comms = midcom[[mc[c-1], mc[c]]].dropna()
        c1 = Clustering(elm2clu_dict={k: [v] for k, v in
                                      dict(comms[mc[c]]).items()})
        c2 = Clustering(elm2clu_dict={k: [v] for k, v in
                                      dict(comms[mc[c-1]]).items()})
        cs.append(sim.element_sim(c1, c2, alpha=0.9))

    clusims[e] = cs

# %% save sim data
amis_df = pd.DataFrame(amis)
amis_df.index = np.exp(np.arange(-8, 0.1, 1))
clusims_df = pd.DataFrame(clusims)
clusims_df.index = np.exp(np.arange(-8, 0.1, 1))

amis_df.to_hdf('support_data/res_amis.h5', key='df')
clusims_df.to_hdf('support_data/res_clusims.h5', key='df')

# %% Plot sims

amis_df = pd.read_hdf('support_data/res_amis.h5', key='df')
clusims_df = pd.read_hdf('support_data/res_clusims.h5', key='df')

amis_df.mean(axis=1).plot(label='AMI')
clusims_df.mean(axis=1).plot(label='CluSim')
plt.xscale('log')
plt.title('Community detection partition similarity')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.legend()
plt.show()
