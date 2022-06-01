#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:31:14 2022

@author: Patrick
"""
import pandas as pd
import glob
import json
from dateutil.relativedelta import relativedelta
import functions1 as pgc

pgpath = '/Volumes/PGPassport/'
t3path = '/Volumes/Samsung_T3/'

with open(pgpath+'active_data_2019/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open(pgpath + 'active_data_2019/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

DD = sorted([x for x in glob.glob(pgpath + 'evtLocal/*/tcweights.h5.h5')])
DDC = {}
for n, x in enumerate(DD):
    print(n/len(DD))
    with pd.HDFStore(x, mode='r') as hdf:
        for k in hdf.keys():
            DDC[x[:-16]+k+'.h5'] = pd.read_hdf(x, key=k)

pip = pd.read_hdf('active_data/H_final_partition.h5', key='df')

gjdf = pd.DataFrame()

for n, m in enumerate(DD):
    print(n/len(DD))
    wj = pd.read_hdf(m.replace('tcweights.h5.h5', 'wjac.h5'), key='df')
    gjdf = gjdf.append(wj)


gjdf['max'] = gjdf.max(axis=1)
gjdf['comm'] = pip

ixs = []
for i in gjdf.index:
    if i[-3:] != '.h5':
        ixs.append(i+'.h5')
    else:
        ixs.append(i)
gjdf.index = ixs

meanwj = gjdf.groupby('comm').mean()
medianwj = gjdf.groupby('comm').median()
meanwj['count'] = gjdf['comm'].value_counts()
medianwj['count'] = gjdf['comm'].value_counts()
meanwj = meanwj.sort_values('count', ascending=False)


mos = ['201710', '201711', '201712', '201801', '201802', '201803', '201804',
       '201805', '201806', '201807', '201808', '201809', '201810', '201811',
       '201812', '201901']
tsl = {}
for k in mos:
    tsl[k] = pd.read_hdf('active_data/finalts/daily_final__%s.h5' % k,  # find path
                         key='df')


# %%
errs2 = []
for n, i in enumerate(gjdf[gjdf['PROM'].isna()].index):
    print(n/len(gjdf[gjdf['PROM'].isna()]))

    try:
        cm = DDC['/Volumes/PGPassport/evtLocal/'+i]
        arts = set(cm.index)
        date = pd.to_datetime(i[:8])

        start = date-relativedelta(days=30)
        stop = date+relativedelta(days=30)

        months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))

        artsr = [y for x in arts
                 for y in redir_arts_map.get(x.replace(' ', '_'), [x])]

        cc = []
        for m in months:
            cc.append(tsl[m][sorted({x.replace(' ', '_') for x in artsr}
                                    & set(tsl[m].columns))])

        ts = pd.concat(cc, sort=True).fillna(
            0).loc[start:stop-relativedelta(days=1)]
        ts.columns = [megamap.get(x, x) for x in ts.columns]
        ts = ts.T.groupby(ts.T.index).sum().T

        weights = cm

        wts = ts.dot(weights.reindex(ts.columns).fillna(0))

        premed = wts.loc[date-relativedelta(days=30):date].median()

        peakex = (wts.loc[date-relativedelta(days=1):
                          date + relativedelta(days=1)]-premed).max()
        peakdev = peakex/(wts.loc[date-relativedelta(days=30):date].quantile(.75) -
                          wts.loc[date-relativedelta(days=30):date].quantile(.25))

        gjdf.loc[i, ['PROM', 'MAG', 'DEV']] = [premed, peakex, peakdev]

    except Exception as ex:
        # raise
        print(i, ex)
        errs2.append((i, ex))
