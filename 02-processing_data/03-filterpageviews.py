#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:40:17 2022

@author: Patrick
"""
import json
import datetime
import multiprocessing
import os
import glob
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

ProgressBar().register()

# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

montharticles = {k: set()
                 for k in wnn.utilities.months_range(pd.to_datetime('20171101'),
                                                     pd.to_datetime('20181231'))}
allels = glob.glob(BPATH + 'events/*/all_el100NNN.h5')

with open('support_data/redir_arts_map.json', 'r') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

with open('support_data/megamap.json', 'r') as json_data:
    megamap = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

pvfiles = sorted(glob.glob(BPATH + 'pageviews/en/p*viewen'))


# %% generate articles for each month
if os.path.exists('support_data/montharticles.json'):

    with open('support_data/montharticles.json', 'r') as fp:
        montharticles = {k: set(v) for k, v in json.load(fp).items()}
        fp.close()

else:

    for n, e in enumerate(allels):
        if n % 100 == 0:
            print('%.2f %%' % (100*n/len(allels)))
        el = pd.read_hdf(e)
        allarts = set(el['prev']) | set(el['curr'])
        date = datetime.datetime.strptime(e.split('events/')[-1][:8], '%Y%m%d')
        start = date - datetime.timedelta(days=30)
        stop = date + datetime.timedelta(days=30)
        mr = wnn.utilities.months_range(start, stop)
        for m in mr:
            montharticles[m] |= allarts

    with open('support_data/montharticles.json', 'w+') as fp:
        json.dump({k: list(v) for k, v in montharticles.items()}, fp)
        fp.close()


# %% Create hourly and daily timeseries for relevant articles

for i in pvfiles:
    try:
        y = i.split('pagecounts-')[1][:4]
        m = i.split('pagecounts-')[1][5:7]
        if os.path.exists(BPATH + 'pageviews/daily/daily_t_series_%s.h5'
                          % (y+m)):
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
                     redir_arts_map[rdarts_rev.get(x, x)]}

        print('loccing', len(marticles))
        common = set(marticles) & set(dfr.index)

        print('getting ts', len(common))
        t_s = dd.from_pandas(dfr.loc[common],
                             npartitions=128*multiprocessing.cpu_count())
        del dfr, common, marticles

        print('getting timeseries', len(t_s))
        timeseries = t_s.map_partitions(lambda df:
                                        df.apply(lambda x:
                                                 wnn.data.text_to_tseries(
                                                     x, int(y), int(m)))
                                        ).compute(scheduler='processes',
                                                  num_workers=10)

        print('aggregating')
        del t_s
        timeseries['article'] = timeseries.index.map(rdarts_rev)  #
        agg = timeseries.groupby('article').sum().T
        del timeseries

        print('saving', agg.T.memory_usage().sum()/1000000000)
        agg.index = pd.to_datetime(agg.index)
        agg.to_hdf(BPATH + 'pageviews/hourly/hourly_t_series_%s.h5' %
                   (y+m), key='df')
        agg.resample('d').sum().to_hdf(BPATH + 'pageviews/daily/daily_t_series_%s.h5'
                                       % (y+m), key='df')
        del agg
    except Exception as ex:
        print(i, ex)
