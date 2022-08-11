#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:01:58 2022

@author: Patrick
"""

import json
import glob
import igraph
import leidenalg as la
import pandas as pd


# %% Load data
BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open(BPATH + 'aux_data/H_edgelist.json', 'r') as f:
    tuples = json.load(f)

G = igraph.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])

del tuples

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

# %% Run community detection

r = 0.067
partition = la.find_partition(G, la.CPMVertexPartition,
                              resolution_parameter=r, weights='weight')
pip = pd.Series({y: n for n, x in enumerate(partition)
                 for y in x}).sort_index()
pip.index = G.vs['name']

# %% Save partition

pip.to_hdf('support_data/H_final_partition.h5', key='df')
