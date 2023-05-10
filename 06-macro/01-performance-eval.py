#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:21:25 2023

@author: Patrick
"""
import json
import igraph
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
import WikiNewsNetwork as wnn

#%% Load data

BPATH = ''

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}
evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')
evlsn = evlsn.sort_values('len')
#%% Setup dicts

DDC = {}
DDC2 = {}
el_D = {}
comms_D = {}
commsw_D = {}
commst_D = {}
igname_D = {}
ignamew_D = {}
core_D = {}
G_D = {}
Gw_D = {}
capedf = pd.DataFrame()

#%% Load existing community detection results

for n, e in enumerate(evlsn.index):
    if n%100==0:
        print(n/len(evlsn))
    
    if (e not in DDC2)&(os.path.exists(BPATH + 'events/' + e +
                                       '/tcweights.h5')):
        with pd.HDFStore(BPATH + 'events/' + e +'/tcweights.h5', mode='r'
                         ) as hdf:
            for k in hdf.keys():
                DDC[e + '/' + k[1:].replace('/', ':')
                    ] = pd.read_hdf(BPATH + 'events/' + e +'/tcweights.h5',
                                    key=k) 
        
    if (e not in comms_D)&(os.path.exists(BPATH + 'events/' + e +
                                          '/f_comms.pkl')):
        with open(BPATH + 'events/' + e + '/f_comms.pkl', 'rb') as f:
            comms_D[e] = pickle.load(f)
         
    if (e not in commsw_D)&(os.path.exists(BPATH + 'events/' + e +
                                          '/w_comms.pkl')):    
        with open(BPATH + 'events/' + e + '/w_comms.pkl', 'rb') as f:
            commsw_D[e] = pickle.load(f)    

# print('Reading comms 2')
for n, (k, v) in enumerate(DDC.items()):
    if k.split('/')[-2] not in DDC2.keys():
        DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
    else:
        DDC2[k.split('/')[-2]][k.split('/')[-1]] = v 

commst_D = {e:{k:list(v.index) for k, v in c.items()} for e, c in DDC2.items()}

#%% Calculate partitions for all events, diff methods

fres = np.exp(-3.5)
wres = np.exp(4)
errors = []
for n, e in enumerate(evlsn.index):
    if n%100==0:
        print(n/len(evlsn))
        
    try:  
        if (e in comms_D)&(e in commsw_D)&(e in commst_D):
            continue
                  
        if e not in commst_D:
            commst_D[e] = {k:list(v.index) for k, v in DDC2[e].items()}
    
        if e not in core_D:      
            core_D[e] = [rdarts_rev.get(x.replace(' ', '_'),
                                        x.replace(' ', '_'))
                            for x in pd.read_csv(BPATH + 'events/' + e +
                                                 '/core.tsv', sep='\t',
                                                 header=None)[0]]
            
        if e not in el_D:
            el_D[e] = pd.read_hdf(BPATH + 'events/' + e +'/all_el100NNN.h5')
        
        if (e not in ignamew_D)|(e not in Gw_D):
            w_el = el_D[e].copy()
            w_el = w_el[['prev', 'curr', 'n']]
            w_el.columns = ['source', 'target', 'weight']
            Gw_D[e] = igraph.Graph.TupleList([tuple(x) for x in w_el.values],
                                       directed=True, edge_attrs=['weight'])
            ignamew_D[e] = {x.index: x['name'] for x in Gw_D[e].vs}
        
        if (e not in igname_D)|(e not in G_D):
            w_el = el_D[e].copy()
            w_el = w_el[['prev', 'curr', 'n']]
            w_el.columns = ['source', 'target', 'weight']
            w_el['weight'] = 1
            G_D[e] = igraph.Graph.TupleList([tuple(x) for x in w_el.values],
                                       directed=True)
            igname_D[e] = {x.index: x['name'] for x in G_D[e].vs}            

        if e not in comms_D:
            comms_D[e] = wnn.cd.flat_CD(G_D[e], igname_D[e], core_D[e], [fres]
                                     )[1][fres]
            with open(BPATH + 'events/' + e + '/f_comms.pkl', 'wb') as f:
                pickle.dump(comms_D[e], f)
                
        if e not in commsw_D:
            commsw_D[e] = wnn.cd.flat_CD(Gw_D[e], ignamew_D[e], core_D[e],
                                         [wres], 'weight')[1][wres]  
            with open(BPATH + 'events/' + e + '/w_comms.pkl', 'wb') as f:
                pickle.dump(commsw_D[e], f)            

        del core_D[e], el_D[e], G_D[e], Gw_D[e], igname_D[e], ignamew_D[e]
    
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(e, ex)
        errors.append((e, ex))


#%% Get stats for excess page views with different methods

for n, e in enumerate(evlsn.index):
    if n%100==0:
        print(n/len(evlsn))
    try:
        if e in capedf.dropna().index:
            continue
        ts = pd.read_hdf(BPATH + 'events/' + e + '/tsNN.h5')
        ts.index = range(-30,31)   
        for k, v in {'f':comms_D[e], 'w':commsw_D[e], 't':commst_D[e]}.items():        
            capedf.loc[e, 'X_%s' %k] = wnn.cd.captured_excess(v, ts)
            capedf.loc[e, 'X7_%s' %k] = wnn.cd.captured_excess(v, ts, 6)
                    
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(e, ex)
        errors.append((e, ex))
capedf.to_hdf('support_data/total_excessdf.h5', 'df')
        
#%% Calculate results based on capedf

print((capedf['X7_t'] >= capedf['X7_f']).sum()/len(capedf))
print((capedf['X7_t'] >= capedf['X7_w']).sum()/len(capedf) )                            


print(gmean((capedf['X7_t'].clip(0) / capedf['X7_f'].clip(0)
       ).replace({np.inf:np.nan, -np.inf:np.nan, 0:np.nan}).dropna()))                               
print(gmean((capedf['X7_t'].clip(0) / capedf['X7_w'].clip(0)
       ).replace({np.inf:np.nan, -np.inf:np.nan, 0:np.nan}).dropna()))                             
print((capedf['X7_t'] / capedf['X7_f']).median())                            
print((capedf['X7_t'] / capedf['X7_w']).median())                         


#%% narrow down events to find example candidate

ei = evlsn.reset_index()

excessdf = pd.DataFrame()
trial_ev = list(ei[(ei['len']<=1000)&
              (ei['index'].str.split('--').str.len()>2)]['index'])

del ei
#%% Get stats for diff methods 

devl_D = {}
tsd = {}
#%%
 
errors = []
# resrange = np.exp(np.arange(0, 10, 0.5))
for n, e in enumerate(trial_ev):
    print(n/len(trial_ev), len(excessdf))
    try:
        if e in excessdf.dropna().index:
            continue
        if e not in tsd:
            tsd[e] = pd.read_hdf(BPATH + 'events/' + e + '/tsNN.h5')
            tsd[e].index = range(-30,31)              
        for k, v in {'f':comms_D[e], 'w':commsw_D[e], 't':commst_D[e]}.items():           
            excessdf.loc[e, 'X_%s' %k] = wnn.cd.captured_excess(v, tsd[e])
            excessdf.loc[e, 'X7_%s' %k] = wnn.cd.captured_excess(v, tsd[e], 6)
            excessdf.loc[e, 'N_%s' %k] = len(v)        
            excessdf.loc[e, 'S_%s' %k] = np.mean([len(x) for x in v.values()])
            
        devl_D[e] = [wnn.cd.get_mdev(v, tsd[e])[0]
                     for v in commst_D[e].values()]
        excessdf.loc[e, 'maxdev'] = max(devl_D[e])
        excessdf.loc[e, 'meandev'] = np.mean(devl_D[e])
        excessdf.loc[e, 'mindev'] = min(devl_D[e])

        comms, commst = wnn.cd.rearrange_comms(comms_D[e].copy(),
                                               commst_D[e].copy())
        commsw, commst = wnn.cd.rearrange_comms(commsw_D[e].copy(),
                                                commst_D[e].copy())
        excessdf.loc[e, 'comm_match'] = (comms.keys() == commst.keys())
        excessdf.loc[e, 'commw_match'] = (commsw.keys() == commst.keys())       
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(e, ex)
        errors.append((e, ex))

excessdf.to_hdf('support_data/excessdf.h5', 'df')

#%% identify shortlist for examples
     
nc = excessdf['N_w']==excessdf['N_t']
xc = (excessdf['X7_t']>excessdf['X7_w'])&(excessdf['X7_t']>excessdf['X7_f'])
dc = excessdf['mindev']>3
cc = excessdf['comm_match']
cwc = excessdf['commw_match']

shortlist = excessdf[nc&xc&dc&cc&cwc].sort_values('mindev', ascending = False)
