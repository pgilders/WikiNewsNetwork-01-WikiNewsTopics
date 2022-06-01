#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:40:17 2022

@author: Patrick
"""
import dask.dataframe as dd
import multiprocessing
import os
import glob
import pandas as pd
import json
import datetime
import functions1 as pgc
from dask.diagnostics import ProgressBar
ProgressBar().register()

BPATH = '/Volumes/PGPassport/DPhil redo data/'

montharticles = {k: set() for k in pgc.months_range(20171101, 20181231)}

for e in glob.glob(BPATH + 'events/*/all_el100NN.h5'):
    el = pd.read_hdf(e)
    allarts = set(el['prev']) | set(el['curr'])
    date = e.split('events/')[-1][:8]
    start = date - datetime.timedelta(days=30)
    stop = date + datetime.timedelta(days=30)
    mr = pgc.months_range(start, stop)
    for m in mr:
        montharticles[m] |= allarts

with open('support_data/montharticles.json') as fp:
    json.dump(montharticles, fp)
    fp.close()

# %%

with open('redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

with open('megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}
rdartsflat = list({y for x in redir_arts_map.values() for y in x})

with open('montharticles.json') as fp:
    montharticles = json.load(fp)
    fp.close()

pvfiles = glob.glob(BPATH + 'pageviews/raw/p*viewen')
pvfiles = sorted(pvfiles)


# check underscore/nonunderscore
# rogue encodings?
for i in pvfiles[-3:]:
    try:
        y = i.split('pagecounts-')[1][:4]
        m = i.split('pagecounts-')[1][5:7]
        if os.path.exists(BPATH + '/hourly/hourly_t_series_%s.h5'
                          % (str(y)+str(m))):
            print(i, 'exists')
            continue

        print(i)
        mdf = pd.read_csv(i, sep=' ', header=None)

        print('setting index')

        mdf = mdf.set_index(1)

        print('getting col')
        dfr = mdf[3]
        del mdf

        print('getting marticles')
        marticles = {z for x in montharticles[y+m] for z in
                     redir_arts_map[megamap.get(x, x).replace(' ', '_')]}

        print('loccing', len(marticles))
        common = set(marticles) & set(dfr.index)

        print('getting ts', len(common))
        t_s = dd.from_pandas(dfr.loc[common],
                             npartitions=16*multiprocessing.cpu_count())
        del dfr

        print('getting timeseries', len(t_s))
        timeseries = t_s.map_partitions(lambda df: df.apply(lambda x: pgc.text_to_tseries(
            x, int(y), int(m)))).compute(scheduler='processes', num_workers=4)

        print('aggregating')
        del t_s
        timeseries['article'] = timeseries.index.map(rdarts_rev)
        agg = timeseries.groupby('article').sum().T
        del timeseries

        print('saving', agg.T.memory_usage().sum()/1000000000)
        agg.to_hdf(BPATH + '/hourly/hourly_t_series_%s.h5' %
                   (str(y)+str(m)), key='df')
        agg.resample('d').sum().to_hdf(BPATH + '/daily/daily_t_series_%s.h5' %
                                       (str(y)+str(m)), key='df')
        del agg
    except Exception as ex:
        print(i, ex)
        pass
