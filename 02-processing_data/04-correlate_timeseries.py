#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:51:08 2022

@author: Patrick
"""

import os
import datetime
from joblib import Parallel, delayed
import json
import glob
import pandas as pd
import numpy as np
import functions1 as pgc

BPATH = '/Volumes/PGPassport/DPhil redo data/'

events = [x for x in glob.glob(BPATH + 'events/all_el100NN.h5')]

# pgpath = '/Volumes/PGPassport/'
# t3path = '/Volumes/Samsung_T3/'

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('support_data/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

tsl = {}
for k in glob.glob(BPATH + 'pageviews/daily/finalts_d/*'):
    tsl[k[-9:-3]] = pd.read_hdf(k, key='df')

# %% Create df with events and all associated articles

if os.path.isfile('support_data/evlsN.h5'):
    evlsn = pd.read_hdf('support_data/evlsN.h5', key='df')
else:
    evlsn = pd.DataFrame(columns=['len', 'articles'])

remaining = sorted(set(events)-set(evlsn.index))
e2el = {}
for n in range(0, len(events), 100):
    print(n/len(events))
    evlt = [pgc.getevls(x, evlsn, tsl, megamap) for x in events[n:n+100]]
    # evlt = Parallel(n_jobs=-1, verbose=5)(delayed(getevls)(x) for x in events[n:n+100])
    e2 = {k: v for d in evlt for k, v in d.items()}
    e2g = {k: v for k, v in e2.items() if type(v) == dict}
    e2e = {k: v for k, v in e2.items() if type(v) != dict}
    e2el = {**e2el, **e2e}
    print(len(e2e), 'errors')
    evlsn = evlsn.append(pd.DataFrame.from_dict(e2g, orient='index'))
    print('total', len(evlsn), len(e2el))
    # evlsn.to_hdf('support_data/evlsN.h5', key='df')

# evlsn.to_hdf('support_data/evlsN.h5', key='df')


# %% Generate temporal network simtgraph5NN.npz etc

edf = evlsn.copy()
edf = edf[edf['len'] > 0].sort_values('len')
fn = 'simtgraph5NN.npz'
elf = '/all_el100NN.h5'
colt = 'coltitlesNN.h5'
adjname = 'adjNN.h5'
qn = 'quantilesNN.npz'

errors = []
for q, row in enumerate(edf.iterrows()):
    if q % 100 == 0:
        print(round(q/len(edf), 3))

    i = row[0]
    if os.path.isfile('%s/%s' % (i, adjname)):
        continue

    try:
        articles = row[1]['articles']
        date = datetime.datetime.strptime(i[:8], '%Y%m%d')
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)
        months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))

        el = pgc.csgraph(i, articles, megamap, start, stop, elf)

        print('filesread')

        adj, tsx = pgc.procts(el, tsl, articles, start, stop, months)

        print('getting scores', len(tsx.columns))
        scorelist = []
        scl = []

        scorelist = Parallel(n_jobs=-1, verbose=5)(delayed(pgc.scoreretlist)(x)
                                                   for x in
                                                   range(0, len(tsx)-6))
        print('got scores')
        tlen = 55
        pd.Series(tsx.columns).to_hdf(
            BPATH + 'events/%s/%s' % (i, colt), key='df')
        pcs = np.array([np.triu(adj.values)*scorelist[x] for x in range(tlen)])

        print('writing')
        np.savez_compressed(BPATH + 'events/%s/%s' % (i, fn), pcs)

        print('writing qs, adj')
        nonedges = (np.array(scorelist) - pcs).flatten()
        del pcs
        # print(1)
        nonedges = nonedges[nonedges != 0]
        # print(2)
        quantiles = pd.Series(np.nanquantile(nonedges, [0, 0.5, 0.9]),
                              index=[0, 0.5, 0.9])
        del nonedges
        # print(3)
        quantiles.to_hdf(BPATH + 'events/%s/%s' % (i, qn), key='df')
        adj.to_hdf(BPATH + 'events/%s/%s' % (i, adjname), key='df')

    except KeyboardInterrupt:
        print('interrupt')
        break
    except Exception as ex:
        # raise
        print(i, ex)
        errors.append((i, ex))
