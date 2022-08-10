#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:52:40 2022

@author: Patrick
"""

import numpy as np
import pandas as pd
import json
import igraph
import glob
import leidenalg as la
from joblib import Parallel, delayed
import functions1 as pgc
import matplotlib.pyplot as plt

# %% Load data
BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open(BPATH + 'aux_data/H_edgelist.json', 'r') as f:
    tuples = json.load(f)

G = igraph.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])

del tuples

# Add isolate nodes

DD = sorted([x[:-13] for x in glob.glob(BPATH + 'events/*/tcweights.h5')])
allnodes = []
for n, x in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
        for k in hdf.keys():
            allnodes.append(x + '/' + k[1:].replace('/', ':'))

G.add_vertices(sorted({x.split('/events/')[-1] for x in allnodes}
                      - set(G.vs['name'])))

# %% Run community detection range and save as df

partitions = {}
for r in np.exp(np.arange(-5, 0.1, 0.1)):  # parallelise? memory concerns
    print(r)
    partitions[r] = la.find_partition(G, la.CPMVertexPartition,
                                      resolution_parameter=r, weights='weight')


pi = pd.DataFrame()
for r in np.exp(np.arange(-5, 0.1, 0.1)):
    print(r)
    pi[r] = pd.Series({y: n for n, x in enumerate(partitions[r])
                       for y in x}).sort_index()
pi.index = G.vs['name']

pi.to_hdf('support_data/H_partitions.h5', key='df')

# %% Calculate AMI + Clusim scores

pi = pd.read_hdf('support_data/H_partitions.h5', key='df')

sims = Parallel(n_jobs=4, verbose=10)(
    delayed(pgc.H_sim)(pi[[pi.columns[n-1], pi.columns[n]]].dropna().values)
    for n in range(1, len(pi.columns)))

pd.Series([x[0] for x in sims],
          index=pi.columns[1:]).to_hdf('support_data/H_ami.h5', key='df')
pd.Series([x[1] for x in sims],
          index=pi.columns[1:]).to_hdf('support_data/H_clusim.h5', key='df')

# %% Plot and select resolution (r = 0.06)

data_ami = pd.read_hdf('support_data/H_ami.h5', key='df')
data_clusim = pd.read_hdf('support_data/H_clusim.h5', key='df')

plt.figure(figsize=[12, 8])
pd.Series(data_ami, index=pi.columns[1:]).plot(label='AMI', lw=3)
pd.Series(data_clusim, index=pi.columns[1:]).plot(label='CluSim', lw=3)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.9, 1])
plt.title('Higher Level Partition Stability')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.show()
