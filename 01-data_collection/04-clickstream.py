#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:09:23 2022

@author: Patrick
"""

import glob
import pandas as pd
import calendar

# %% load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

cs_files = sorted(glob.glob(BPATH+'clickstream/clickstream-enwiki-*.tsv'))

dfall = pd.DataFrame(columns=['prev', 'curr', 'type', 'n'])
dflinks = pd.DataFrame(columns=['prev', 'curr', 'n'])

# %% combine clickstream months

pfilter = ['other-empty', 'other-search', 'other-other', 'other-external'
           'other-internal', 'Main_Page', 'Hyphen-minus']

for f in cs_files:
    print(f)
    y = int(f.split('-')[-2])
    m = int(f.split('-')[-1][:2])

    dfm = pd.read_csv(f, sep='\t', header=None,
                      names=['prev', 'curr', 'type', 'n_%d-%02d' % (y, m)])
    print('file read')
    dfall = pd.merge(dfall, dfm, on=['prev', 'curr', 'type'], how='outer')
    del dfm

    dfall['n'] = dfall['n'].fillna(0) + dfall['n_%d-%02d' % (y, m)].fillna(0)
    dfall['n_%d-%02d' % (y, m)] = dfall['n_%d-%02d' %
                                        (y, m)] / calendar.monthrange(y, m)[1]

print('files done')
dflinks = dfall[(dfall['type'] == 'link') & (~dfall['prev'].isin(pfilter)) &
                (~dfall['curr'].isin(pfilter))]

# %% Save dfs

dflinks = dflinks[['n', 'prev', 'curr', 'n_2017-11', 'n_2017-12', 'n_2018-01',
                   'n_2018-02', 'n_2018-03', 'n_2018-04', 'n_2018-05',
                   'n_2018-06', 'n_2018-07', 'n_2018-08', 'n_2018-09',
                   'n_2018-10', 'n_2018-11', 'n_2018-12']]
dfall.to_hdf(BPATH+'clickstream/clickstream_all_201711-201812.h5', key='df')
dflinks.to_hdf(BPATH+'clickstream/clickstream_artlinks_201711-201812.h5',
               key='df')
