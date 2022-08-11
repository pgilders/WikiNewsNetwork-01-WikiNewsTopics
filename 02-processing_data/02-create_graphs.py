#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:36:09 2022

@author: Patrick
"""

import json
from calendar import monthrange
import pandas as pd
# from joblib import Parallel, delayed
import functions1 as pgc

# %% load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open('support_data/redir_arts_map.json', 'r') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('support_data/megamap.json', 'r') as json_data:
    megamap = json.load(json_data)
    json_data.close()


mcsdf = pd.read_hdf(
    BPATH + 'clickstream/clickstream_artlinks_201711-201812.h5')

eventsdf = pd.read_hdf('support_data/eventsdf.h5')


# %% generate edgelists for each event

errors = []
eventsdf['MR'] = eventsdf.apply(lambda x: frozenset(pgc.getmr(x.name).keys()),
                                axis=1)
mrset = set(eventsdf['MR'])

for mn, mrk in enumerate(mrset):
    print('\nMonth keys: %.2f %%\n==================' % (100*mn/len(mrset)))
    mr = {x: monthrange(int(x[2:6]), int(x[7:9]))[1] for x in mrk}
    csd = mcsdf[['prev', 'curr']+sorted(mr)].copy().fillna(0)
    for k, v in mr.items():
        csd[k] = csd[k]*v
    csd['n'] = csd[csd.columns[3:]].sum(axis=1)
    csd = csd[csd['n'] > 0]
    for k, v in mr.items():
        csd[k] = csd[k]/v

    events = list(eventsdf[(eventsdf['MR'] == mrk) &
                           (eventsdf['Articles'].str.len() > 0)].index)

    print('%d Events' % len(events))
    for n in range(0, len(events), 100):
        print('%.2f %%' % (100*n/len(events)))
        try:
            # evwriter = Parallel(n_jobs=-1, verbose=5)(delayed(getel)(x) for x in events[n:n+24])
            evwriter = [pgc.getel_2(x, csd, redir_arts_map, rdarts_rev,
                                    eventsdf) for x in events[n:n+100]]
            for e in evwriter:
                if len(e) == 2:
                    e[1].to_hdf(BPATH + 'events/' + e[0].replace('/', ':')
                                + '/all_el100NNN.h5', key='df')
                else:
                    errors.append((n, e))
                    print('ERROR', e)
        except KeyboardInterrupt:
            print('Interrupted')
            raise
        except Exception as ex:
            print(n, ex)

    del csd
