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

with open('support_data/redir_arts_map2.json', 'r') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('support_data/megamap2.json', 'r') as json_data:
    megamap = json.load(json_data)
    json_data.close()


mcsdf = pd.read_hdf(BPATH +
                    'clickstream/clickstream_artlinks_201711-201812.h5')

# events = list(storiesdf.index)
eventsdf = pd.read_hdf('support_data/eventsdf.h5')
events = [x for x in eventsdf.index if x[-4:] != '__CS']


# %% create edgelist for every month combo

# mrs = {frozenset(pgc.getmr(x).keys()) for x in events}

# csd = {}
# for x in mrs:
#     try:
#         print(x)
#         mr = {'n_%s-%s' %
#               (y[2:6], y[7:]): monthrange(int(y[2:6]), int(y[7:]))[1] for y in x}
#         el = mcsdf[['prev', 'curr']+list(x)].copy().fillna(0)
#         for k, v in mr.items():
#             el[k] = el[k]*v
#         el['n'] = el[el.columns[3:]].sum(axis=1)
#         # el = el[el['n'] > 100]
#         for k, v in mr.items():
#             el[k] = el[k]/v
#         csd[x] = el.copy()
#     except KeyboardInterrupt:
#         raise
#     except Exception as ex:
#         print(x, ex)

# del mcsdf
# del el

# for k in csd.keys():
#     csd[k] = csd[k][['prev', 'curr'] +
#                     sorted(csd[k].columns[2:-1])+[csd[k].columns[-1]]]


# %% Generate edgelist all_el100NN.h5 for each event

errors = []
for n in range(0, len(events), 100):
    print('%.2f %%' % (100*n/len(events)))
    try:
        # evwriter = Parallel(n_jobs=-1, verbose=5)(delayed(getel)(x) for x in events[n:n+24])
        evwriter = [pgc.getel(x, csd, megamap, redir_arts_map, eventsdf)  # !!!!!!!
                    for x in events[n:n+100]]
        for e in evwriter:
            if len(e) == 2:
                e[1].to_hdf(BPATH + 'events/' + e[0].replace('/', ':')
                            + '/all_el100NNN.h5', key='df')
            else:
                errors.append((n, e))
                print('ERROR', e)
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(n, ex)
# %%

eventsdf['MR'] = eventsdf.apply(lambda x: frozenset(pgc.getmr(x.name).keys()),
                                axis=1)
mrset = set(eventsdf['MR'])

for n, mrk in enumerate(mrset):
    print('\nMonth keys: %.2f %%\n==========' % (100*n/len(mrset)))
    mr = {x: monthrange(int(x[2:6]), int(x[7:9]))[1] for x in mrk}
    csd = mcsdf[['prev', 'curr']+list(mr)].copy().fillna(0)
    for k, v in mr.items():
        csd[k] = csd[k]*v
    csd['n'] = csd[csd.columns[3:]].sum(axis=1)
    csd = csd[csd['n'] > 0]
    # el = el[el['n'] > 100]
    for k, v in mr.items():
        csd[k] = csd[k]/v

    events = list(eventsdf[eventsdf['MR'] == mrk].index)
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
            raise
        except Exception as ex:
            print(n, ex)

    del csd
