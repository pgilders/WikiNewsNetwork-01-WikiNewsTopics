#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:36:29 2022

Get Structural Similarity Scores

@author: Patrick
"""

from joblib import Parallel, delayed
import glob
import pandas as pd
import numpy as np
import json
import functions1 as pgc

pgpath = '/Volumes/PGPassport/'
t3path = '/Volumes/Samsung_T3/'

evlsn = pd.read_hdf('active_data/evlsN.h5', key='df')

with open(pgpath+'active_data_2019/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open(pgpath + 'active_data_2019/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

# %%


wjdone = set([x[:-8].replace(pgpath + 'evtLocal', 'active_data/tempevs')
              for x in glob.glob(pgpath + 'evtLocal/*/wjac.h5')])
doneHDD = set([x[:-9].replace(pgpath + 'evtLocal', 'active_data/tempevs')
               for x in glob.glob(pgpath + 'evtLocal/*/adjNN.h5')])

allev = [x.replace('active_data/tempevs', t3path + 'evtest')
         for x in doneHDD-wjdone]

core_D = {f[:-9]: [megamap.get(x, x)
                   for x in pd.read_csv(f, sep='\t', header=None)[0]]
          for f in glob.glob(t3path + 'evtest/*/core.tsv')}

flatadj_D = {x[:-9].replace('active_data/tempevs',
                            t3path + 'evtest'): pd.read_hdf(x, key='df')
             for x in glob.glob('active_data/tempevs/*/adjNN.h5')}


DD = sorted([x for x in glob.glob(pgpath + 'evtLocal/*/tcweights.h5.h5')])
DDC = {}
for n, x in enumerate(DD):
    print(n/len(DD))
    with pd.HDFStore(x, mode='r') as hdf:
        for k in hdf.keys():
            DDC[x[:-16]+k+'.h5'] = pd.read_hdf(x, key=k)

DDC2 = {}
for n, (k, v) in enumerate(DDC.items()):
    print(n/len(DDC))
    if k.split('/')[-2] not in DDC2.keys():
        DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
    else:
        DDC2[k.split('/')[-2]][k.split('/')[-1]] = v


werrs = []
step = 200
resrange = np.exp(np.arange(-9, 1, 1))  # better res???????
for m in range(0, len(allev), step):
    print(m)
    out = Parallel(n_jobs=-1, verbose=10)(delayed(pgc.cdflatR)
                                          ([x, evlsn.loc[x]], resrange,
                                           core_D[x],
                                           flatadj_D.get(x, pd.DataFrame()))
                                          for x in allev[m:m+step])

    print('Getting Js')
    wws = Parallel(n_jobs=-1, verbose=10)(delayed(pgc.cmmatcher)
                                          (n, x, DDC2.get(allev[m+n].split('/')[-1], {}),
                                           resrange, m, allev)
                                          for n, x in enumerate(out))

    print('Writing Js')
    for n, x in enumerate(wws):
        try:
            x[0].to_hdf(pgpath + 'evtLocal/%s/wjac.h5'
                        % allev[m+n].split('evtest/')[-1], key='df')
            x[1].to_hdf(pgpath + 'evtLocal/%s/jac.h5'
                        % allev[m+n].split('evtest/')[-1], key='df')
        except Exception as ex:
            print(ex)
            werrs.append([n, x, ex])
