#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:36:09 2022

@author: Patrick
"""

import os
from calendar import monthrange
from joblib import Parallel, delayed
import json
import glob
import pandas as pd

import functions1 as pgc

BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('support_data/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

tsl = {}
for k in glob.glob(BPATH + '/daily/*'):
    tsl[k[-9:-3]] = pd.read_hdf(k, key='df')


mcsdf = pd.read_hdf(
    BPATH + 'clickstream/clickstream_artlinks_201711-201812.h5')

# events = list(storiesdf.index)
eventsdf = pd.read_hdf('support_data/eventsdf.h5')
events = [x for x in eventsdf.index if x[-4:] != '__CS']


# %% create edgelist for every month combo

mrs = {frozenset(pgc.getmr(x).keys()) for x in events}

csd = {}
for x in mrs:
    try:
        print(x)
        mr = {'n_%s-%s' %
              (y[2:6], y[7:]): monthrange(int(y[2:6]), int(y[7:]))[1] for y in x}
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

del mcsdf

for k in csd.keys():
    csd[k] = csd[k][['prev', 'curr'] +
                    sorted(csd[k].columns[2:-1])+[csd[k].columns[-1]]]


# %% Generate edgelist all_el100NN.h5 for each event

errors = []
for n in range(0, len(events), 100):
    print(n)
    try:
        # evwriter = Parallel(n_jobs=-1, verbose=5)(delayed(getel)(x) for x in events[n:n+24])
        evwriter = [pgc.getel(x, csd, megamap, redir_arts_map)
                    for x in events[n:n+100]]
        for e in evwriter:
            if len(e) == 2:
                e[1].to_hdf(e[0]+'/all_el100NN.h5', key='df')
                print('SUCCESS', e[0])
            else:
                errors.append((n, e))
                print('ERROR', e)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(n, ex)
