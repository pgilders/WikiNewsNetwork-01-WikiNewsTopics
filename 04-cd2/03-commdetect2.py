#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:01:58 2022

@author: Patrick
"""

import leidenalg as la
import pandas as pd

G = igraph.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])

# %%


r = 0.12
partition = la.find_partition(G, la.CPMVertexPartition,
                              resolution_parameter=r, weights='weight')
pip = pd.Series({y: n for n, x in enumerate(partition)
                 for y in x}).sort_index()
pip.index = G.vs['name']


pip.to_hdf('active_data/H_final_partition.h5', key='df')
