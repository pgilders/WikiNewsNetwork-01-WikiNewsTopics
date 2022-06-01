#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:07:36 2022



@author: Patrick
"""

from joblib import Parallel, delayed
import glob
import pandas as pd
import numpy as np
import json
import igraph
import unicodedata
import functions1 as pgc


pgpath = '/Volumes/PGPassport/'

DD = sorted([x for x in glob.glob(pgpath + 'evtLocal/*/tcweights.h5.h5')])
DDC = {}
for n, x in enumerate(DD):
    print(n/len(DD))
    with pd.HDFStore(x, mode='r') as hdf:
        for k in hdf.keys():
            DDC[x[:-16]+k+'.h5'] = pd.read_hdf(x, key=k)

DDC2 = {}
for n, (k, v) in enumerate(DDC.items()):
    print(n/len(DDC))
    if k.split('/')[-2] not in DDC2.keys():
        DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
    else:
        DDC2[k.split('/')[-2]][k.split('/')[-1]] = v
# %%
combos = [(DDC[k1], DDC[k2]) for n, k1 in enumerate(sorted(DDC.keys()))
          for m, k2 in enumerate(sorted(DDC.keys())) if n < m]

# %%
for n in range(37000000, len(combos), int(1E6)):
    print(n)
    # jacsw = Parallel(n_jobs=-1, verbose=3)(delayed(pgc.wjac)(x[0], x[1])
    # for x in combos[n:n+int(1E6)])
    jacsw = [pgc.wjac(dict(x[0]), dict(x[1])) for x in combos[n:n+int(1E6)]]
    with open('/Users/Patrick/active_data/jw2/jacsw%.1f.json' % (n/1E6), 'w+') as f:
        json.dump(jacsw, f)


del combos


# %%
jll = []
for n in range(0, 132787956, int(1E6)):
    print(n, 132787956)
    with open('/Users/Patrick/active_data/jw2/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll.extend(json.load(f))

# %%
# Create DataFrame, edgelist, graph from wjacs
simdf = pd.DataFrame(arr, index=sorted(DDC.keys()), columns=sorted(DDC.keys()))
del arr, ixsall, ixc, jll  # , ixspre, ixspost
sds = simdf.stack()
del simdf

sds = sds[sds > 0]
el = sds.reset_index()
del sds

el.columns = ['source', 'target', 'weight']
el.source = el.source.str.split('evtLocal/').str[-1]
el.target = el.target.str.split('evtLocal/').str[-1]

tuples = [tuple(x) for x in el.values]
del el

G = igraph.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])
