#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:46:08 2022

@author: Patrick
"""

import pandas as pd
import numpy as np
import glob
import functions1 as pgc

# %% Read data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

gjdf = pd.read_hdf('support_data/evr_props.h5', key='df')
meanwj = pd.read_hdf('support_data/mean_topic_props.h5', key='df')
meanwj['Name'] = np.nan

eventsdf = pd.read_hdf('support_data/eventsdf.h5')

DD = sorted([x[:-13] for x in glob.glob(BPATH + 'events/*/tcweights.h5')])

ddc_count = 0
DDC = {}
for n, x in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
        for k in hdf.keys():
            ddc_count += 1
            DDC[x + '/' + k[1:].replace('/', ':')
                ] = pd.read_hdf(x+'/tcweights.h5', key=k)

# %%
for f in ['count', 'PROM', 'MAG', 'DEV']:
    ixs = meanwj.sort_values(f, ascending=False)[(meanwj['count'] >= 10) &
                                                 (meanwj['Name'].isna())
                                                 ].iloc[:20].index
    for i in ixs:
        print(i, len(ixs))
        evrs = sorted(set(gjdf[gjdf['comm'] == i].index))
        evs = [x.split('/')[0] for x in evrs]
        cores = pd.Series([x for y in evrs for x in
                           eventsdf.loc[y.split('/')[0],
                                        'Articles']]).value_counts()

        allarts = pd.concat([DDC[BPATH + 'events/'+x] for x in evrs]
                            ).reset_index().groupby('index').sum().sort_values(
                                by=0, ascending=False)

        n = 0
        print(f, meanwj.loc[i, f])
        while True:
            print('Stories')
            print('\n\n'.join(eventsdf.loc[evs, 'Text'].iloc[n:n+10].values))
            print('\n\n')
            print('Cores')
            print(cores.iloc[n:n+10])
            print('\n\n')
            print('All arts')
            print(allarts.iloc[n:n+10])

            mi = input('More info?\n')

            if mi.lower() == 'n':
                break
            n += 10

        meanwj.loc[i, 'Name'] = input('Enter name\n')

        print('##'*20)


# %%
n_labellings = len(glob.glob('support_data/topic_labels_*'))
meanwj.dropna().to_hdf('support_data/topic_labels_%d.h5' % (n_labellings+1),
                       key='df')
