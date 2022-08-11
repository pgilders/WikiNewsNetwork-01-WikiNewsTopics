#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:51:08 2022

@author: Patrick
"""

import os
import datetime
import json
import glob
# from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import functions1 as pgc


# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

events = [x.split('/')[-2]
          for x in glob.glob(BPATH + 'events/*/all_el100NNN.h5')]

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('support_data/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

tsl = {}
for k in sorted(glob.glob(BPATH + 'pageviews/daily/daily_t_series*')):
    tsl[k[-9:-3]] = pd.read_hdf(k, key='df')

if os.path.isfile(BPATH + 'aux_data/evlsN.h5'):
    evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')
else:
    evlsn = pd.DataFrame(columns=['len', 'articles'])

# %% Create df with events and all associated articles

adjname = 'adjNN.h5'
tsname = 'tsNN.h5'

remaining = sorted(set(events)-set(evlsn.index))
e2el = {}
for n in range(0, len(events), 100):
    print(n/len(events))
    evlt = [pgc.getevls_2(x, evlsn, tsl, rdarts_rev, BPATH+'events/')
            for x in events[n:n+100]]
    # evlt = Parallel(n_jobs=-1, verbose=5)(delayed(getevls)(x)
    #                                       for x in events[n:n+100])
    e2 = {k: v for d in evlt for k, v in d[0].items()}
    e2g = {k: v for k, v in e2.items() if isinstance(v, dict)}
    e2e = {k: v for k, v in e2.items() if not isinstance(v, dict)}
    e2el = {**e2el, **e2e}
    print(len(e2e), 'errors')

    for m, d in enumerate(evlt):
        if events[n+m] not in e2e:
            d[1].to_hdf(BPATH + 'events/%s/%s' % (events[n+m], adjname),
                        key='df')
            d[2].to_hdf(BPATH + 'events/%s/%s' % (events[n+m], tsname),
                        key='df')

    evlsn = evlsn.append(pd.DataFrame.from_dict(e2g, orient='index'))
    print('total', len(evlsn), len(e2el))
    evlsn.to_hdf('support_data/evlsN.h5', key='df', mode='w')


# %% Generate temporal network simtgraph5NN.npz etc

edf = evlsn.copy()
edf = edf[edf['len'] > 0].sort_values('len')
fn = 'simtgraphelNN.npz'
fnix = 'simtgraphixNN.npz'
colt = 'coltitlesNN.h5'

errors = []
for q, row in enumerate(edf.iterrows()):
    if q % 100 == 0:
        print(100*round(q/len(edf), 3), '%%')

    i = row[0]
    if os.path.isfile(BPATH + 'events/%s/%s' % (i, fn)):
        continue

    try:
        articles = row[1]['articles']
        date = datetime.datetime.strptime(i[:8], '%Y%m%d')
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)
        months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))

        adj = pd.read_hdf(BPATH + 'events/%s/%s' % (i, adjname))
        ts = pd.read_hdf(BPATH + 'events/%s/%s' % (i, tsname))

        # print('filesread')
        scaler = RobustScaler()
        tsxv = scaler.fit_transform(ts)

        # print('get scores')
        pd.Series(ts.columns).to_hdf(BPATH + 'events/%s/%s' % (i, colt),
                                     key='df')
        pcsel, ixs = pgc.rolling_pearson_ixs(tsxv, adj)
        # del scorelist

        # print('writing')
        np.savez_compressed(BPATH + 'events/%s/%s' % (i, fn), pcsel)
        del pcsel
        np.savez_compressed(BPATH + 'events/%s/%s' % (i, fnix), ixs)
        del ixs

    except KeyboardInterrupt:
        print('interrupt')
        break
    except Exception as ex:
        # raise
        print(i, ex)
        errors.append((i, ex))
