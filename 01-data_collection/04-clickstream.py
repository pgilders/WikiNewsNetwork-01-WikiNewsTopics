#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:09:23 2022

@author: Patrick
"""
# clickstream (download + ?) combine

import glob
import pandas as pd
import calendar

BPATH = '/Volumes/PGPassport/DPhil redo data/'

cs_files = sorted(glob.glob(BPATH+'clickstream/clickstream-enwiki-*.tsv'))

dfall = pd.DataFrame(columns=['prev', 'curr', 'type', 'n'])
dflinks = pd.DataFrame(columns=['prev', 'curr', 'n'])

# %%
pfilter = ['other-empty', 'other-search', 'other-other', 'other-external'
           'other-internal', 'Main_Page', 'Hyphen-minus']

for f in cs_files:
    print(f)
    y = int(f.split('-')[-2])
    m = int(f.split('-')[-1][:2])

    dfm = pd.read_csv(f, sep='\t', header=None,
                      names=['prev', 'curr', 'type', 'n_%d-%d' % (y, m)])
    dfm_links = dfm[(dfm['type'] == 'link') & (~dfm['prev'].isin(pfilter)) &
                    (~dfm['curr'].isin(pfilter))]

    dfall = pd.merge(dfall, dfm, on=['prev', 'curr', 'type'], how='outer')
    del dfm
    dflinks = pd.merge(dflinks, dfm_links, on=['prev', 'curr'], how='outer')
    del dfm_links

    dfall['n'] = dfall['n'].fillna(0) + dfall['n_%d-%d' % (y, m)]
    dfall['n_%d-%d' % (y, m)] = dfall['n_%d-%d' %
                                      (y, m)] / calendar.monthrange(y, m)[1]

    dflinks['n'] = dflinks['n'].fillna(0) + dflinks['n']
    dflinks['n_%d-%d' % (y, m)] = dflinks['n_%d-%d' %
                                          (y, m)] / calendar.monthrange(y, m)[1]

# %%

# dfall.to_hdf(BPATH+'clickstream/clickstream_all_201711-201812.h5', key='df')
# dflinks.to_hdf(BPATH+'clickstream/clickstream_artlinks_201711-201812.h5',
#                key='df')
