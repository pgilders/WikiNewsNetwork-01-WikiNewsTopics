#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:31:14 2022

@author: Patrick
"""
import pandas as pd
import glob
import json
import numpy as np
from dateutil.relativedelta import relativedelta

# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

DD = sorted([x[:-13] for x in glob.glob(BPATH + 'events/*/tcweights.h5')])

DDC = {}
for n, x in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
        for k in hdf.keys():
            DDC[x + '/' + k[1:].replace('/', ':')
                ] = pd.read_hdf(x+'/tcweights.h5', key=k)

pip = pd.read_hdf('support_data/H_final_partition.h5', key='df')

wjs = pd.Series()
for n, m in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    wj = pd.read_hdf(m + '/wjac.h5',
                     key='df').max(axis=1)
    wjs = wjs.append(wj)

# %% Create df of event reactions

gjdf = pd.DataFrame()

gjdf['SS_score'] = wjs
gjdf['comm'] = pip
gjdf['evrsize'] = [len(DDC[BPATH + 'events/' + x]) for x in gjdf.index]
gjdf['PROM'] = np.nan

# %% Assign basic properties to each event reaction

ready = gjdf[gjdf['PROM'].isna()].index
errs2 = []
for n, i in enumerate(ready):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(ready)))
    try:
        evr = DDC[BPATH + 'events/' + i]
        arts = set(evr.index)
        date = pd.to_datetime(i[:8])

        ts = pd.read_hdf(BPATH + 'events/' + i.split('/')[0] + '/tsNN.h5',
                         key='df')[arts]

        wts = ts.dot(evr.reindex(ts.columns).fillna(0))

        prets = wts.loc[date-relativedelta(days=30):
                        date - relativedelta(days=1)]
        premed = prets.median()
        peakex = (wts.loc[date - relativedelta(days=1):
                          date + relativedelta(days=1)] - premed).max()
        peakdev = peakex/(prets.quantile(.75) - prets.quantile(.25))

        gjdf.loc[i, ['PROM', 'MAG', 'DEV']] = [premed, peakex, peakdev]

    except Exception as ex:
        # raise
        print(i, ex)
        errs2.append((i, ex))

# %% Read and group stats

meanwj = gjdf.groupby('comm').mean()
medianwj = gjdf.groupby('comm').median()
meanwj['count'] = gjdf['comm'].value_counts()
medianwj['count'] = gjdf['comm'].value_counts()
meanwj = meanwj.sort_values('count', ascending=False)

# %% Save dfs

gjdf.to_hdf('support_data/evr_props.h5', key='df')
meanwj.to_hdf('support_data/mean_topic_props.h5', key='df')
