#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:52:40 2022

@author: Patrick
"""

import numpy as np
import pandas as pd
import igraph
import leidenalg as la
import sklearn.metrics
from clusim.clustering import Clustering
import clusim.sim as sim
import matplotlib.pyplot as plt

# %%
G = _

partitions = {}
for r in np.exp(np.arange(-5, 0.1, 0.1)):
    print(r)
    partitions[r] = la.find_partition(G, la.CPMVertexPartition,
                                      resolution_parameter=r, weights='weight')


pi = pd.DataFrame()
for r in np.exp(np.arange(-5, 0.1, 0.1)):
    print(r)
    pi[r] = pd.Series({y: n for n, x in enumerate(partitions[r])
                       for y in x}).sort_index()
pi.index = G.vs['name']

pi.to_hdf('active_data/H_partition.h5', key='df')

# %%

data = []
datc = []
for n in range(0, len(pi.columns)):  # range(47?
    print(n)
    ct = pi[[pi.columns[n-1], pi.columns[n]]]
    ct = ct.dropna()
    c1 = ct[pi.columns[n]]
    c2 = ct[pi.columns[n-1]]

    data.append(sklearn.metrics.adjusted_mutual_info_score(c1, c2))
    c1 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c1).items()})
    c2 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c2).items()})
    datc.append(sim.element_sim(c1, c2, alpha=0.9))

pd.Series(data, index=pi.columns[1:]).to_hdf('active_data/H_ami.h5', key='df')
pd.Series(datc, index=pi.columns[1:]).to_hdf(
    'active_data/H_clusim.h5', key='df')
# %%
plt.figure(figsize=[12, 8])
pd.Series(data, index=pi.columns[1:]).plot(label='AMI', lw=3)
pd.Series(datc, index=pi.columns[1:]).plot(label='CluSim', lw=3)
plt.legend()
plt.xscale('log')
plt.title('Higher Level Partition Stability')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.show()
