#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:06:10 2022

@author: Patrick
"""
# %% load data

import json
import pandas as pd
import functions1 as pgc
from calendar import monthrange

BPATH = '/Volumes/PGPassport/DPhil redo data/'

eventsdf = pd.read_hdf('support_data/eventsdf.h5')

mcsdf = pd.read_hdf(BPATH+'clickstream/clickstream_artlinks_201711-201812.h5')
# %%
mrs = {frozenset(pgc.getmr(x.replace('/', ':')).keys())
       for x in eventsdf.index}

csd = {}
for x in mrs:
    try:
        print(x)
        mr = {'n_%s-%s' % (y[2:6], y[7:]):
              monthrange(int(y[2:6]), int(y[7:]))[1] for y in x}
        el = mcsdf[['prev', 'curr']+list(x)].copy().fillna(0)
        for k, v in mr.items():
            el[k] = el[k]*v
        el['n'] = el[el.columns[3:]].sum(axis=1)
        el = el[el['n'] > 100]
        for k, v in mr.items():
            el[k] = el[k]/v
        csd[x] = el.copy()
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(x, ex)


for k in csd.keys():
    csd[k] = csd[k][['prev', 'curr'] +
                    sorted(csd[k].columns[2:-1])+[csd[k].columns[-1]]]

# %% query for redirects

allcore = {y for e in eventsdf['Articles'] for y in e}
corerd = pgc.get_redirects(allcore)
# %%
allneighbours = []
for n, e in enumerate(eventsdf.index):
    print(n/len(eventsdf))
    allneighbours.append(pgc.get_neighbours_quick(
        e, csd, corerd, eventsdf))
# allneighbours = [pgc.get_neighbours_quick(e, csd, corerd, eventsdf)
#                  for e in eventsdf.index]
allneighbours_set = {y for x in allneighbours if len(x) == 2 for y in x[1]}

print('redirects')
m_map = pgc.fix_redirects(allneighbours_set)
rd_arts_map = pgc.get_redirects({m_map.get(x, x) for x in allneighbours_set})

# %% save redirects

# with open('support_data/megamap.json', 'w+') as f:
#     json.dump(m_map, f)

# with open('support_data/redir_arts_map.json', 'w+') as f:
#     json.dump(rd_arts_map, f)
