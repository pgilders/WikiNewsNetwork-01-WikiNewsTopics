#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:06:10 2022

@author: Patrick
"""

import json
import pandas as pd
import functions1 as pgc
from calendar import monthrange
import datetime

# %% load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

eventsdf = pd.read_hdf('support_data/eventsdf.h5')

mcsdf = pd.read_hdf(BPATH+'clickstream/clickstream_artlinks_201711-201812.h5')


# %% query for redirects for core articles

allcore = {y for e in eventsdf['Articles'] for y in e}
corerd = pgc.get_redirects(allcore)

# %% get all articles across months from clickstreams (monthset wise)

allneighbours = {}
eventsdf['MR'] = eventsdf.apply(lambda x: frozenset(pgc.getmr(x.name).keys()),
                                axis=1)
mrset = set(eventsdf['MR'])

for n, mrk in enumerate(mrset):
    print('%.2f %%' % (100*n/len(mrset)))
    mr = {x: monthrange(int(x[2:6]), int(x[7:9]))[1] for x in mrk}
    csd = mcsdf[['prev', 'curr']+sorted(mr)].copy().fillna(0)
    for k, v in mr.items():
        csd[k] = csd[k]*v
    csd['n'] = csd[csd.columns[3:]].sum(axis=1)
    csd = csd[csd['n'] > 0]
    for k, v in mr.items():
        csd[k] = csd[k]/v

    corearts = {y for x in eventsdf[eventsdf['MR'] == mrk]['Articles']
                for y in x}

    allneighbours[mrk] = pgc.get_neighbours_quick_2(corearts, csd, corerd)
    del csd

allneighbours_set = {y for x in allneighbours.values() if len(x) != 2
                     for y in x}

# %% get all articles across months from clickstreams (event-wise, more stable?)

# allneighbours = {}
# prevmrk = frozenset()
# for n, e in enumerate(eventsdf.index):
#     if n % 100 == 0:
#         print('%.2f %%' % (100*n/len(eventsdf)))

#     date = pd.to_datetime(e[:8])
#     start = date-datetime.timedelta(days=30)
#     stop = date+datetime.timedelta(days=30)

#     months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))
#     mr = {'n_%s-%s' % (x[:4], x[4:]):
#           monthrange(int(x[:4]), int(x[4:]))[1] for x in months}
#     mrk = frozenset(mr.keys())

#     if mrk != prevmrk:
#         print('Refiltering edgelist')
#         csd = mcsdf[['prev', 'curr']+sorted(mr)].copy().fillna(0)
#         for k, v in mr.items():
#             csd[k] = csd[k]*v
#         csd['n'] = csd[csd.columns[3:]].sum(axis=1)
#         csd = csd[csd['n'] > 0]
#         for k, v in mr.items():
#             csd[k] = csd[k]/v
#         prevmrk = mrk
#         print('done')

#     allneighbours[e] = pgc.get_neighbours_quick_2(
#         eventsdf.loc[e, 'Articles'], csd, corerd)

# allneighbours_set = {y for x in allneighbours.values() if len(x) != 2
#                      for y in x}
# %% save all neighbours

with open('support_data/allneighbours.json', 'w+') as f:
    json.dump({k: list(v) for k, v in allneighbours.items()}, f)

# %% fix redirects

print('redirects')
m_map = pgc.fix_redirects(allneighbours_set)

rd_arts_map = pgc.get_redirects({m_map.get(x.replace('_', ' '),
                                           x.replace('_', ' '))
                                 for x in allneighbours_set})


rd_arts_map = {**corerd, **rd_arts_map}


# %% save redirects

with open('support_data/megamap.json', 'w+') as f:
    json.dump(m_map, f)

with open('support_data/redir_arts_map.json', 'w+') as f:
    json.dump(rd_arts_map, f)
