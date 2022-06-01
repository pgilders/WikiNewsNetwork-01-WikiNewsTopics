#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:57:12 2022

Initial community detection on new event networks

@author: Patrick
"""

from joblib import Parallel, delayed
import json
import glob
import pandas as pd
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

evlsn = pd.read_hdf('active_data/evlsN.h5', key='df')


# %%

for maxthresh, step, njo in [(2000, 60, -1), (3000, 30, 8), (30000, 4, 4)]:
    ready = set([x[:-9] for x in glob.glob('active_data/tempevs/*/adjNN.h5')])
    finished = set([x.replace(pgpath + 'evtLocal', 'active_data/tempevs')[:-16]
                    for x in glob.glob(pgpath + 'evtLocal/*/tcweights.h5.h5')])

    edf = evlsn.loc[[x.replace('active_data/tempevs', t3path + 'evtest')
                     for x in ready-finished]].sort_values('len')[evlsn['len']
                                                                  < maxthresh]
    quantile = 0
    res = 0.25
    for n in range(0, len(edf), step):
        print(n)
        try:
            out1 = Parallel(n_jobs=njo, verbose=10)(delayed(pgc.cd1)(
                [x, evlsn.loc[x]], res, quantile, megamap,
                fp='/simtgraph5NN.npz')
                for x in edf.index[n:n+step])
            out2 = Parallel(n_jobs=njo, verbose=10)(delayed(pgc.cd2)(*x)
                                                    for x in out1
                                                    if len(x) == 5)

            names = [x for m, x in enumerate(
                edf.index[n:n+step]) if len(out1[m]) == 5]
            for m, i in enumerate(out2):
                try:
                    i[0].to_hdf('%s/%s'
                                % (names[m].replace('Samsung_T3/evtest',
                                                    'PGPassport/evtLocal'),
                                   'membdf.h5'), key='df')
                    for k, v in i[1].items():
                        v.to_hdf('%s/%s.h5'
                                 % (names[m].replace('Samsung_T3/evtest',
                                                     'PGPassport/evtLocal'),
                                    k), key='df')
                    for k, v in i[2].items():
                        v.to_hdf('%s/%s.h5' % (names[m].replace(
                            'Samsung_T3/evtest', 'PGPassport/evtLocal'),
                            'tcweights.h5'), key=k)
                except Exception as ex:
                    # raise
                    print(ex)
        except Exception as ex:
            print(ex)
