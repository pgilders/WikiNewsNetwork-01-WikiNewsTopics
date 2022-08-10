#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:21:56 2022

@author: Patrick
"""
from joblib import Parallel, delayed
import json
import glob
import pandas as pd
import functions1 as pgc
import random
import os

BPATH = ''

with open('support_data/redir_arts_map3a.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

evlsn = pd.read_hdf('support_data/evlsN.h5', key='df')

errors = []

ready = set([x.split('/')[-2] for x in
             glob.glob(BPATH + 'events/*/simtgraphixNN.npz')])
finished = set([x.split('/')[-2] for x in
                glob.glob(BPATH+'events_out/*/tcweights.h5')])

res = 0.25

# print('testing 1 ev')
# testev = random.choice(list(ready))
# print('reading 1 ev')
# out1 = [pgc.read_ev_data(BPATH + 'events/' + testev, rdarts_rev)]
# print('CD 1 ev', len(out1[0][1]))
# out2 = [pgc.ev_reactions_tcd(*out1[0], res)]
# print('writng 1 ev')
# names = [BPATH+'events_out/'+testev]

# for m, i in enumerate(out2):
#     try:
#         os.mkdir(names[m])
#         i[0].to_hdf('%s/membdf.h5' % names[m], key='df')
#         for k, v in i[1].items():
#             v.to_hdf('%s/evrs.h5' % names[m], key=k)
#         for k, v in i[2].items():
#             v.to_hdf('%s/tcweights.h5' % names[m], key=k)
#     except Exception as ex:
#         errors.append((m, ex))
#         print(ex)

# print('testing 1 ev complete')


print('testing 10 ev')

esamp = evlsn.loc[list((ready-finished) & set(evlsn.index))
                  ][evlsn['len'] > 20].sort_values('len').index[:96]
try:
    print('reading 10 ev')
    out1 = [pgc.read_ev_data(BPATH + 'events/' + x, rdarts_rev)
            for x in esamp]

    print('CD 10 ev', [len(x[1]) for x in out1])
    out2 = Parallel(n_jobs=-1, verbose=10
                    )(delayed(pgc.ev_reactions_tcd)(*x, res)
                      for x in out1 if len(x) == 4)

    print('writng 10 ev')
    names = [BPATH+'events_out/'+x for m, x in
             enumerate(esamp) if len(out1[m]) == 4]
    for m, i in enumerate(out2):
        try:
            os.mkdir(names[m])
            i[0].to_hdf('%s/membdf.h5' % names[m], key='df')
            for k, v in i[1].items():
                v.to_hdf('%s/evrs.h5' % names[m], key=k)
            for k, v in i[2].items():
                v.to_hdf('%s/tcweights.h5' % names[m], key=k)
        except Exception as ex:
            errors.append((m, ex))
            print(ex)
except Exception as ex:
    errors.append(ex)
    print(ex)
