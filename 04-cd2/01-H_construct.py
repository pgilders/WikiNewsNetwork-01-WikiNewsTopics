#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:07:36 2022



@author: Patrick
"""

import glob
import json
import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

DD = sorted([x[:-13] for x in glob.glob(BPATH + 'events/*/tcweights.h5')])

ddc_count = 0
DDC = {}
for n, x in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
        for k in hdf.keys():
            ddc_count += 1
            DDC[x + '/' + k[1:].replace('/', ':')
                ] = pd.read_hdf(x+'/tcweights.h5', key=k)

# %% Create combinations

combos = [(k1, k2) for n, k1 in enumerate(sorted(DDC.keys()))
          for m, k2 in enumerate(sorted(DDC.keys())) if n < m]
lc = len(combos)
print(lc)

# %% Calculated weighted jacs

for n in range(0, lc, int(1E6)):
    if n % 10000000 == 0:
        print('%.2f %%' % (100*n/lc))

    jacsw = [wnn.utilities.wjac(DDC[x[0]], DDC[x[1]])
             for x in combos[n:n+int(1E6)]]

    with open(BPATH + 'evr_similarities/jacsw%.1f.json' % (n/1E6), 'w+') as f:
        json.dump(jacsw, f)

del jacsw, DDC


# %% Read weighted jacs

jll = []
for n in range(0, lc, int(1E6)):
    if n % 10000000 == 0:
        print('%.2f %%' % (100*n/lc))
    with open(BPATH + 'evr_similarities/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll.extend(json.load(f))

# %% Create edgelist from wjacs

tuples = [(*[y.split('/events/')[-1] for y in combos[n]], x)
          for n, x in enumerate(jll) if x > 0]

# %% Save edgelist

with open(BPATH + 'aux_data/H_edgelist.json', 'w+') as f:
    json.dump(tuples, f)
