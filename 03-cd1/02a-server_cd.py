#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:57:12 2022

Initial community detection on news event networks

Script for AWS server for remaining events, more powerful than my laptop...

@author: Patrick
"""

import json
import os
import glob
from joblib import Parallel, delayed
import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

BPATH = ''

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

# dict of folder names as filename limits stricter on server
with open('support_data/rem_folders.json') as json_data:
    namedict = json.load(json_data)
    json_data.close()

namedict = {k: v.split('/')[-1] for k, v in namedict.items()}

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}
namedictrev = {v: k for k, v in namedict.items()}

evlsn = pd.read_hdf('support_data/evlsN.h5', key='df')

errors = []
for maxthresh, step, njo in [(4000, 384, -1), (12000, 256, -1),
                             (30000, 128, -1)]:
    ready = {namedict[x.split('/')[-2]]
             for x in glob.glob(BPATH + 'events_N/*/simtgraphixNN.npz')}
    finished = {namedict[x.split('/')[-2]]
                for x in glob.glob(BPATH+'events_out_N/*/tcweights.h5')}

    edf = evlsn.loc[list((ready - finished) & set(evlsn.index))
                    ].sort_values('len')[evlsn['len'] < maxthresh]
    res = 0.25
    for n in range(0, len(edf), step):
        print(n+len(finished), edf.iloc[min(n+step, len(edf)-1)]['len'])
        try:

            out2 = Parallel(n_jobs=njo, verbose=10
                            )(delayed(wnn.cd.server_tcd)(BPATH + 'events_N/' +
                                                         str(namedictrev[x]),
                                                         rdarts_rev, res)
                              for x in edf.index[n:n + step])

            names = [BPATH+'events_out_N/'+str(namedictrev[x]) for m, x in
                     enumerate(edf.index[n:n+step])]

            for m, i in enumerate(out2):
                if len(i) == 2:
                    continue
                try:
                    try:
                        os.mkdir(names[m])
                    except FileExistsError:
                        print('Directory exists')
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
