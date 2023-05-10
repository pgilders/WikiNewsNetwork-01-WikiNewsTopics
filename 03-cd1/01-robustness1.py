#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:27:01 2022

@author: 
"""

import random
import json
from sklearn.metrics import adjusted_mutual_info_score as ami
from joblib import Parallel, delayed
from clusim.clustering import Clustering
import numpy as np
import pandas as pd
import clusim.sim as sim
import igraph
import matplotlib.pyplot as plt

import WikiNewsNetwork as wnn


plt.style.use('seaborn-darkgrid')
with open('figures/figurestyle.json', 'r') as f:
    params = json.load(f)
plt.rcParams.update(params)


# %% Load data

BPATH = ''

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')

# %% Sample events and read graphs

random.seed(100)
evsample = random.sample(list(evlsn.index), 100)
edf = evlsn.loc[evsample].sort_values('len')
out1 = [wnn.cd.read_ev_data(BPATH + 'events/' + x, rdarts_rev)
        for x in edf.index]

#%% Run comm detection on range of resolutions, compare partitions, and save

resrange = np.exp(np.arange(-9, 0.1, 1))       

resresults, amis_df, clusims_df = wnn.cd.cd_range_test(out1, evsample, resrange,
                                                BPATH +
                                                'aux_data/res2_select.h5',
                                                'support_data/res_%s.h5',
                                                wnn.cd.ev_reactions)



# %% Plot sims

amis_df = pd.read_hdf('support_data/res_amis.h5', key='df')
clusims_df = pd.read_hdf('support_data/res_clusims.h5', key='df')

plt.figure(figsize=[16, 10])
amis_df.mean(axis=1).plot(label='AMI')
clusims_df.mean(axis=1).plot(label='CluSim')
plt.ylim([0.475, 0.725])
plt.yticks(np.arange(0.5, 0.75, 0.05))
plt.xscale('log')
plt.title('Community Detection Partition Similarity')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.legend()
for ext in ['svg', 'eps', 'pdf', 'png']:
    plt.savefig('figures/cdsim.%s' %ext)
plt.show()


#%% Replace graphs with flat, unweighted versions

for n, e in enumerate(edf.index):
    # print(n, e)    
    w_el = pd.read_hdf(BPATH+'events/'+e+'/all_el100NNN.h5')
    w_el = w_el[['prev', 'curr', 'n']]
    w_el.columns = ['source', 'target', 'weight']
    w_el['weight'] = 1
    g = igraph.Graph.TupleList([tuple(x) for x in w_el.values],
                               directed=True, edge_attrs=['weight'])
    ign = {x.index: x['name'] for x in g.vs}
    
    out1[n] = (*out1[n][:2], g, ign)



#%% Run flat, uw comm detection on range of resolutions, compare partitions, and save
  
resrange = np.exp(np.arange(-9, 0.1, 0.5))       
resresults, amis_df, clusims_df = wnn.cd.cd_range_test(out1, evsample, resrange,
                                                       BPATH + 'aux_data/f_res_select.h5',
                                                       'support_data/f_res_%s.h5',
                                                       wnn.cd.ev_reactions_flat)


# %% Plot flat unweighted sims

amis_df = pd.read_hdf('support_data/f_res_amis.h5', key='df')
clusims_df = pd.read_hdf('support_data/f_res_clusims.h5', key='df')

plt.figure(figsize=[16, 10])
amis_df.mean(axis=1).plot(label='AMI')
clusims_df.mean(axis=1).plot(label='CluSim')
# plt.ylim([0.475, 0.725])
# plt.yticks(np.arange(0.5, 0.75, 0.05))
plt.xscale('log')
plt.title('Community Detection Partition Similarity')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.legend()
for ext in ['svg', 'eps', 'pdf', 'png']:
    plt.savefig('figures/cdsim_f.%s' %ext)
plt.show()

#%% Replace graphs with flat, weighted versions

for n, e in enumerate(edf.index):
    # print(n, e)    
    w_el = pd.read_hdf(BPATH+'events/'+e+'/all_el100NNN.h5')
    w_el = w_el[['prev', 'curr', 'n']]
    
    w_el.columns = ['source', 'target', 'weight']
    g = igraph.Graph.TupleList([tuple(x) for x in w_el.values],
                               directed=True, edge_attrs=['weight'])
    ign = {x.index: x['name'] for x in g.vs}
    
    out1[n] = (*out1[n][:2], g, ign)


#%% Run flat, w comm detection on range of resolutions, compare partitions, and save

resrange = np.exp(np.arange(0, 10, 0.5))

resresults, amis_df, clusims_df = wnn.cd.cd_range_test(out1, evsample, resrange,
                                                       BPATH + 'aux_data/w3_res_select.h5',
                                                       'support_data/w3_res_%s.h5',
                                                       wnn.cd.ev_reactions_flat,
                                                       'weight')



# %% Plot sims

amis_df = pd.read_hdf('support_data/w3_res_amis.h5', key='df')
clusims_df = pd.read_hdf('support_data/w3_res_clusims.h5', key='df')

plt.figure(figsize=[16, 10])
amis_df.mean(axis=1).plot(label='AMI')
clusims_df.mean(axis=1).plot(label='CluSim')
# plt.ylim([0.475, 0.725])
# plt.yticks(np.arange(0.5, 0.75, 0.05))
plt.xscale('log')
plt.title('Community Detection Partition Similarity')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.legend()
for ext in ['svg', 'eps', 'pdf', 'png']:
    plt.savefig('figures/cdsim_w.%s' %ext)
plt.show()
