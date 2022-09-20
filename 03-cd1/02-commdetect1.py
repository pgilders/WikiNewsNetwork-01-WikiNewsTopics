#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:57:12 2022

Initial community detection on news event networks

@author: Patrick
"""

import json
import glob
from joblib import Parallel, delayed
import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

# %% Load data

BPATH = '/Volumes/PGPassport/DPhil redo data/'

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')

# %% Run community detection & save output

errors = []
for maxthresh, step, njo in [(2000, 500, -1), (4000, 100, -1), (8000, 50, 8),
                             (30000, 25, 4)]:
    ready = {x.split('/')[-2] for x in
             glob.glob(BPATH + 'events/*/simtgraphixNN.npz')}
    finished = {x.split('/')[-2] for x in
                glob.glob(BPATH+'events/*/tcweights.h5')}

    edf = evlsn.loc[list(ready-finished)].sort_values('len')[evlsn['len']
                                                             < maxthresh]
    print(len(edf))
    res = 0.25
    for n in range(0, len(edf), step):
        print(n, edf.iloc[min(n+step, len(edf)-1)]['len'])
        try:

            out1 = [wnn.cd.read_ev_data(BPATH + 'events/' + x, rdarts_rev)
                    for x in edf.index[n:n + step]]
            out2 = Parallel(n_jobs=njo, verbose=10
                            )(delayed(wnn.cd.ev_reactions)(*x, res, tcd=True)
                              for x in out1 if len(x) == 4)

            names = [BPATH+'events/'+x for m, x in
                     enumerate(edf.index[n:n+step]) if len(out1[m]) == 4]
            for m, i in enumerate(out2):
                try:
                    i[0].to_hdf('%s/membdf.h5' % names[m], key='df')
                    for k, v in i[1].items():
                        v.to_hdf('%s/evrs.h5' % names[m], key=k)
                    for k, v in i[2].items():
                        v.to_hdf('%s/tcweights.h5' % names[m], key=k)
                except Exception as ex:
                    errors.append((n, m, ex))
                    print(ex)
        except Exception as ex:
            errors.append((n, ex))
            print(ex)
