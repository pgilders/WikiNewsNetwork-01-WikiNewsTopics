#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:46:08 2022

@author: Patrick
"""

import pandas as pd
import glob
import functions1 as pgc

meanwj = pd.read_hdf('', key='df')
gjdf = pd.read_hdf('', key='df')

storiesdf = pd.read_hdf('active_data/storiesdf.h5')

pgpath = '/Volumes/PGPassport/'

DD = sorted([x for x in glob.glob(pgpath + 'evtLocal/*/tcweights.h5.h5')])
DDC = {}
for n, x in enumerate(DD):
    print(n/len(DD))
    with pd.HDFStore(x, mode='r') as hdf:
        for k in hdf.keys():
            DDC[x[:-16]+k+'.h5'] = pd.read_hdf(x, key=k)

for f in ['count', 'PROM', 'MAG', 'DEV']:
    for i in meanwj.sort_values(f, ascending=False)[meanwj['count'] >= 10].iloc[:20][meanwj['Name'] == '?'].index:
        print(i, len(meanwj.sort_values(f)[
              meanwj['count'] >= 5].iloc[:50][meanwj['Name'].isna()]))
        evs = gjdf[gjdf['comm'] == i].index.str.split('/').str[0]
        cores = pd.Series([x for y in gjdf[gjdf['comm'] == i].index.str.split(
            '/').str[-1].str[:-3] for x in y.split('---')]).value_counts()
        allarts = pd.concat([DDC[pgpath+'evtLocal/'+x] for x in gjdf[gjdf['comm'] == i].index]
                            ).reset_index().groupby('index').sum().sort_values(by=0, ascending=False)
        n = 0
        print(f, meanwj.loc[i, f])
        while True:
            print('Stories')
            print('\n\n'.join(storiesdf.loc[sorted(set(evs) & set(
                storiesdf.index)), 'Text'].iloc[n:n+10].values))
            print('\n\n')
            print('Cores')
            print(cores.iloc[n:n+10])
            print('All arts')
            print(allarts.iloc[n:n+10])

            mi = input('More info?\n')

            if mi.lower() == 'n':
                break
            n += 10
        nn = input('Enter name\n')
        meanwj.loc[i, 'Name'] = nn

        print('##'*20)
# %%

outdf = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    outdf[f] = meanwj.sort_values(f, ascending=False)[
        meanwj['count'] >= 10].iloc[:20]['Name'].reset_index(drop=True)


odfv = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    odfv[f] = outdf[f].apply(pgc.colourer, meanwj)

odfv.columns = ['# Events', 'Prominence', 'Magnitude', 'Deviance']
