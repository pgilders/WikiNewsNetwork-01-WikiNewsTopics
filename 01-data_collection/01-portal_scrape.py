#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:58:19 2022

@author: Patrick
"""
import os
import unicodedata
import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

# %% Scrape portal

sdf = wnn.data.wiki_news_articles([x.strftime('%B_%Y') for x in
                                   pd.date_range('20171201', '20181130',
                                                 freq='m')])

# %% Clean data

sdf['Date'] = pd.to_datetime(sdf['Date'], format='%Y_%B_%d')

catmap = {'sport': 'sports',
          'disasters and incidents': 'disasters and accidents',
          'armed conflict and attacks': 'armed conflicts and attacks',
          'politics and election': 'politics and elections',
          'health and medicine': 'health and environment',
          'arts and culture;': 'arts and culture',
          'art and culture': 'arts and culture',
          'disaster and accidents': 'disasters and accidents',
          'armed attacks and conflicts': 'armed conflicts and attacks',
          'crime and law': 'law and crime',
          'health': 'health and environment',
          'culture and media': 'arts and culture',
          'recreation and entertainment': 'arts and culture',
          'video games': 'arts and culture'}

sdf['Category'] = sdf['Category'].apply(lambda x: catmap.get(x.strip().lower(),
                                                             x.strip().lower()))


allarticles = {y for x in sdf['Articles'] for y in x}
rdmap = {}
idmap = {}
for c in wnn.data.chunks(list(allarticles), 50):
    for q in wnn.data.query({'titles': '|'.join(c),
                             'redirects': ''}):
        if 'redirects' in q.keys():
            for r in q['redirects']:
                rdmap[r['from']] = r['to']
        for k, v in q['pages'].items():
            idmap[v['title']] = k

sdf['Articles'] = sdf['Articles'].apply(lambda x: [rdmap.get(y, y) for y in x
                                                   if rdmap.get(y, y) in idmap])

sdf.index = sdf.apply(lambda x:
                      unicodedata.normalize('NFD',
                                            '_'.join([
                                                x['Date'].strftime('%Y%m%d'),
                                                '--'.join(x['Articles'])[:237],
                                                'CS']
                                            ).replace('/', ':')), axis=1)

sdf = sdf[~sdf.index.duplicated(keep='first')]

# %% Save df

sdf.to_hdf('support_data/eventsdf.h5', key='df')

# %% Create folders + core tsv for each event

evpath = '/Volumes/PGPassport/DPhil redo data/events/'

for i in sdf.index:
    if len(i) == 12:
        continue
    if not os.path.exists(evpath+i):
        os.mkdir(evpath+i)
    pd.Series({x: idmap[x] for x in sdf.loc[i, 'Articles']
               if x in idmap}).to_csv('%s%s/core.tsv' % (evpath, i),
                                      sep='\t', header=False)
