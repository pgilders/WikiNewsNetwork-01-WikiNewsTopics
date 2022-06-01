#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:11:10 2021

@author: Patrick
"""

import os
from sklearn.preprocessing import RobustScaler
import calendar
from calendar import monthrange
import datetime
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import json
import glob
import pandas as pd
import igraph
import leidenalg as la
import numpy as np
# %%


def months_range(start, stop):
    xr = start
    stop2 = stop
    stop2 = stop2.replace(day=monthrange(stop2.year, stop2.month)[1])
    months = []
    while xr <= stop2:
        months.append('%d%02d' % (xr.year, xr.month))
        xr += relativedelta(months=1)
    return months


with open('/Volumes/PGPassport/active_data_2019/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('/Volumes/PGPassport/active_data_2019/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()
 # %%


def commmdectFRparallel(row, resrange, quantile=0, fp='/simtgraph5N.npz', flat=False):
    # global pcs, sel, membership, membdf
    #    global sel, nodenames
    i = fp

    try:
        toreturn = []
        core = [megamap.get(x, x) for x in pd.read_csv(
            row[0]+'/core.tsv', sep='\t', header=None)[0]]
        coreu = [x.replace(' ', '_') for x in core]
        # months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        # mr = {'n_%s-%s' %(x[:4], x[4:]):monthrange(int(x[:4]),int(x[4:]))[1] for x in months}

        flatadj = pd.read_hdf(row[0]+'/adjNN.h5', key='df')

        articles = list(flatadj.columns)
        ai = articles

        # print('generating graphs', len(articles))

        if flat:
            sel = flatadj.stack().reset_index()
            sel.columns = ['source', 'target', 'weight']
            sel = sel[sel['source'] != 'Main Page']
            sel = sel[sel['target'] != 'Main Page']

            # print('running community detection')
            flatel = sel[sel['weight'] != 0]
            flatel['weight'] = 1
            flattup = [tuple(x) for x in flatel.values]
            flatG = igraph.Graph.TupleList(flattup, directed=False)

            nodenames = {n: x for n, x in enumerate(flatG.vs['name'])}
            nodenamesr = {v: k for k, v in nodenames.items()}
            del sel, flattup

            membdfFD = {}
            collmemsFD = {}
            for res in resrange:
                partition = la.find_partition(
                    flatG, la.CPMVertexPartition, resolution_parameter=res)
                membdfF = pd.Series(
                    {y: n for n, x in enumerate(partition) for y in x}).sort_index()
                membdfF.index = flatG.vs['name']
                membdfF = membdfF.sort_index()

                # print('getting collmems')
                collmemsF = {}
                for c in set(membdfF.loc[set(coreu) & set(articles)]):
                    collmemsF['---'.join(sorted(membdfF.loc[set(coreu) & set(articles)]
                                         [membdfF == c].index))] = list(membdfF[membdfF == c].index)

                membdfFD[res] = membdfF
                collmemsFD[res] = collmemsF

            toreturn.extend([membdfFD, collmemsFD])
# =============================================================================
#
# =============================================================================

        print('loading data', i)
        pcs = np.nan_to_num(np.load(row[0]+i)['arr_0'])
        q = pd.read_hdf(row[0]+'/quantilesNN.npz')
        pcs = np.where(pcs >= q.loc[quantile], pcs, 0)
        print('generating graphs', len(articles))

        glist = []

        tlen = 55
        for n in range(tlen):
            sel = pd.DataFrame(pcs[n, :, :], index=articles,
                               columns=articles).stack().reset_index()
            sel = sel[sel[0] != 0]
            sel.columns = ['source', 'target', 'weight']
            # sel['source'] = sel['source'].map(nodenames)
            # sel['target'] = sel['target'].map(nodenames)
            sel = sel[sel['source'] != 'Main Page']
            sel = sel[sel['target'] != 'Main Page']
            tuples = [tuple(x) for x in sel.values]
            glist.append(igraph.Graph.TupleList(tuples, directed=False, edge_attrs=[
                         'weight']).simplify(combine_edges='first'))
            # glist[n].vs["label"] = [nodenamesr[x] for x in glist[n].vs["name"]]
            glist[n].vs["slice"] = n

        if sum([len(x.vs) for x in glist]) == 0:
            # print('empty')
            return [i, 'error', 'empty']

        # print('generating agg network')
        igname = [{x.index: x['name'] for x in glist[y].vs}
                  for y in range(tlen)]

        del pcs
        del sel
#        del tuples

        print('running community detection')

        def cdp(glist, res):
            membership, improvement = la.find_partition_temporal(
                glist, la.CPMVertexPartition, vertex_id_attr='name', interslice_weight=1, resolution_parameter=res)
            membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
                               for n, x in enumerate(membership)], axis=1, sort=True)
            return [res, membdf]

        L = dict(Parallel(n_jobs=-1, verbose=10)
                 (delayed(cdp)(glist, res) for res in resrange))
#        print('Lgot')

        membdfD = {}
        collmemsD = {}
        for res in resrange:
            # print(res)
            # membership, improvement = la.find_partition_temporal(glist, la.CPMVertexPartition, vertex_id_attr='name', interslice_weight=1, resolution_parameter=res)
            membdf = L[res]

            # print('getting comm data')
            for n in range(55):
                cattr = membdf[n].copy()
                cattr.index = cattr.index.map(
                    {v: k for k, v in igname[n].items()})
                glist[n].vs['community'] = cattr.sort_index().values
                glist[n].vs["label"] = glist[n].vs["name"]

            # print('getting collmems')
            collmems = {}
            t0comms = membdf.loc[set(coreu) & set(articles), 27]
            membdfc = membdf.loc[set(coreu) & set(articles)].T

            cmd = {x: [] for x in set(t0comms.dropna().values)}
            for x in t0comms.dropna().iteritems():
                cmd[x[1]].append(x[0])

            for k, c in cmd.items():
                segt = ((membdfc[c] == k).sum(axis=1) > .5*len(c))
                tf = segt == segt.shift(1)
                diff = np.where(tf.values == False)[0]-27
                beg = 27+diff[diff <= 0].max()
                try:
                    end = 26+diff[diff > 0].min()
                except ValueError:
                    end = 54
                tstep = membdfc.loc[beg:end, c].index
                t = membdf.T.loc[tstep][membdf.T.loc[tstep]
                                        == k].T.dropna(how='all')
                gg = {x: set(t[x].dropna().index) for x in t}
                js = pd.Series({k: jac(set(gg[27]), set(v))
                               for k, v in gg.items()})
                t = t[js[js > .5].index].dropna(how='all')
                collmems['---'.join(sorted(c))] = t

            # uniquify sets
            for k, v in collmems.items():
                for l, u in collmems.items():
                    if k != l:
                        for s in set(v.index) & set(u.index):
                            f1 = len(v.loc[s].dropna())/len(v.loc[s])
                            f2 = len(u.loc[s].dropna())/len(u.loc[s])
                            if f1 > f2:
                                collmems[l] = collmems[l].drop(s)
                            elif f2 > f1:
                                collmems[k] = collmems[k].drop(s)
                            elif len(v) < len(u):
                                collmems[l] = collmems[l].drop(s)
                            elif len(v) >= len(u):
                                collmems[k] = collmems[k].drop(s)

            # for k, v in collmems.items():
                # commsumm = commsumm.append(pd.Series(v.max().max(), index = v.index))

            membdfD[res] = membdf
            collmemsD[res] = collmems

        toreturn.extend([membdfD, collmemsD])

        return toreturn

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        raise
        return [i, 'error', ex]
# %%


def cdflatR(row, resrange, core, flatadj):
    # global pcs, sel, membership, membdf
    #    global sel, nodenames
    # i = fp

    try:
        toreturn = []
        # core = [megamap.get(x, x) for x in pd.read_csv(row[0]+'/core.tsv', sep='\t', header=None)[0]]
        coreu = [x.replace(' ', '_') for x in core]
        # months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        # mr = {'n_%s-%s' %(x[:4], x[4:]):monthrange(int(x[:4]),int(x[4:]))[1] for x in months}

        # flatadj = pd.read_hdf(row[0].replace('/Volumes/Samsung_T3/evtest', 'active_data/tempevs')+'/adjNN.h5', key = 'df')

        articles = list(flatadj.columns)
        ai = articles

        # print('generating graphs', len(articles))

        # if flat:
        sel = flatadj.stack().reset_index()
        sel.columns = ['source', 'target', 'weight']
        sel = sel[sel['source'] != 'Main Page']
        sel = sel[sel['target'] != 'Main Page']

        # print('running community detection')
        flatel = sel[sel['weight'] != 0]
        flatel['weight'] = 1
        flattup = [tuple(x) for x in flatel.values]
        flatG = igraph.Graph.TupleList(flattup, directed=False)

        nodenames = {n: x for n, x in enumerate(flatG.vs['name'])}
        nodenamesr = {v: k for k, v in nodenames.items()}
        del sel, flattup

        membdfFD = {}
        collmemsFD = {}
        for res in resrange:
            partition = la.find_partition(
                flatG, la.CPMVertexPartition, resolution_parameter=res)
            membdfF = pd.Series(
                {y: n for n, x in enumerate(partition) for y in x}).sort_index()
            membdfF.index = flatG.vs['name']
            membdfF = membdfF.sort_index()

            # print('getting collmems')
            collmemsF = {}
            for c in set(membdfF.loc[set(coreu) & set(articles)]):
                collmemsF['---'.join(sorted(membdfF.loc[set(coreu) & set(articles)]
                                     [membdfF == c].index))] = list(membdfF[membdfF == c].index)

            membdfFD[res] = membdfF
            collmemsFD[res] = collmemsF

        tcdD = {res: {k: centrality(flatG, v, nodenamesr)
                      for k, v in V.items()} for res, V in collmemsFD.items()}

        return membdfFD, collmemsFD, tcdD

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]

     # %%


def commmdectFparallel(row, res, quantile, fp='/simtgraph5N.npz'):
    # global pcs, sel, membership, membdf
    #    global sel, nodenames
    i = fp

    try:

        core = [megamap.get(x, x) for x in pd.read_csv(
            row[0]+'/core.tsv', sep='\t', header=None)[0]]
        coreu = [x.replace(' ', '_') for x in core]
        flatadj = pd.read_hdf(row[0]+'/adjNN.h5', key='df')

        articles = list(flatadj.columns)
        ai = articles

        # print('generating graphs', len(articles))

        sel = flatadj.stack().reset_index()
        sel.columns = ['source', 'target', 'weight']
        sel = sel[sel['source'] != 'Main Page']
        sel = sel[sel['target'] != 'Main Page']

        # print('running community detection')
        flatel = sel[sel['weight'] != 0]
        flatel['weight'] = 1
        flattup = [tuple(x) for x in flatel.values]
        flatG = igraph.Graph.TupleList(flattup, directed=False)

        nodenames = {n: x for n, x in enumerate(flatG.vs['name'])}
        nodenamesr = {v: k for k, v in nodenames.items()}

        del sel, flattup

        partition = la.find_partition(
            flatG, la.CPMVertexPartition, resolution_parameter=res)
        membdfF = pd.Series({y: n for n, x in enumerate(partition)
                            for y in x}).sort_index()
        membdfF.index = flatG.vs['name']
        membdfF = membdfF.sort_index()

        # print('getting collmems')
        collmemsF = {}
        for c in set(membdfF.loc[set(coreu) & set(articles)]):
            collmemsF['---'.join(sorted(membdfF.loc[set(coreu) & set(articles)]
                                 [membdfF == c].index))] = list(membdfF[membdfF == c].index)

        tcdF = {k: centrality(flatG, v, nodenamesr)
                for k, v in collmemsF.items()}

        del flatG, partition, flatel
# =============================================================================
#
# =============================================================================

        print('loading data', i)
        pcs = np.nan_to_num(np.load(row[0]+i)['arr_0'])
        q = pd.read_hdf(row[0]+'/quantilesNN.npz')
        pcs = np.where(pcs >= q.loc[quantile], pcs, 0)
        # articles = pd.read_hdf(row[0]+'/coltitles.h5')

        print('generating graphs', len(articles))

        glist = []

        tlen = 55
        for n in range(tlen):
            sel = pd.DataFrame(pcs[n, :, :], index=articles,
                               columns=articles).stack().reset_index()
            sel = sel[sel[0] != 0]
            sel.columns = ['source', 'target', 'weight']
            # sel['source'] = sel['source'].map(nodenames)
            # sel['target'] = sel['target'].map(nodenames)
            sel = sel[sel['source'] != 'Main Page']
            sel = sel[sel['target'] != 'Main Page']
            tuples = [tuple(x) for x in sel.values]
            glist.append(igraph.Graph.TupleList(tuples, directed=False, edge_attrs=[
                         'weight']).simplify(combine_edges='first'))
            # glist[n].vs["label"] = [nodenamesr[x] for x in glist[n].vs["name"]]
            glist[n].vs["slice"] = n

        if sum([len(x.vs) for x in glist]) == 0:
            # print('empty')
            return [i, 'error', 'empty']

        # print('generating agg network')
        igname = [{x.index: x['name'] for x in glist[y].vs}
                  for y in range(tlen)]
        # ignamer = [{v:k for k,v in x.items()} for x in igname]

        del pcs
        del sel
#        del tuples

        print('running community detection')

        membership, improvement = la.find_partition_temporal(
            glist, la.CPMVertexPartition, vertex_id_attr='name', interslice_weight=1, resolution_parameter=res)
        membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
                           for n, x in enumerate(membership)], axis=1, sort=True)

        print('getting comm data')
        for n in range(55):
            cattr = membdf[n].copy()
            cattr.index = cattr.index.map({v: k for k, v in igname[n].items()})
            glist[n].vs['community'] = cattr.sort_index().values
            glist[n].vs["label"] = glist[n].vs["name"]

        print('getting collmems')
        collmems = {}
        t0comms = membdf.loc[set(coreu) & set(articles), 27]
        membdfc = membdf.loc[set(coreu) & set(articles)].T

        cmd = {x: [] for x in set(t0comms.dropna().values)}
        for x in t0comms.dropna().iteritems():
            cmd[x[1]].append(x[0])

        for k, c in cmd.items():
            segt = ((membdfc[c] == k).sum(axis=1) > .5*len(c))
            tf = segt == segt.shift(1)
            diff = np.where(tf.values == False)[0]-27
            beg = 27+diff[diff <= 0].max()
            try:
                end = 26+diff[diff > 0].min()
            except ValueError:
                end = 54
            tstep = membdfc.loc[beg:end, c].index
            t = membdf.T.loc[tstep][membdf.T.loc[tstep]
                                    == k].T.dropna(how='all')
            gg = {x: set(t[x].dropna().index) for x in t}
            js = pd.Series({k: jac(set(gg[27]), set(v))
                           for k, v in gg.items()})
            t = t[js[js > .5].index].dropna(how='all')
            collmems['---'.join(sorted(c))] = t

        # uniquify sets
        for k, v in collmems.items():
            for l, u in collmems.items():
                if k != l:
                    for s in set(v.index) & set(u.index):
                        f1 = len(v.loc[s].dropna())/len(v.loc[s])
                        f2 = len(u.loc[s].dropna())/len(u.loc[s])
                        if f1 > f2:
                            collmems[l] = collmems[l].drop(s)
                        elif f2 > f1:
                            collmems[k] = collmems[k].drop(s)
                        elif len(v) < len(u):
                            collmems[l] = collmems[l].drop(s)
                        elif len(v) >= len(u):
                            collmems[k] = collmems[k].drop(s)

        tcd = {k: tcentrality(glist, v) for k, v in collmems.items()}

        return membdfF, collmemsF, tcdF, membdf, collmems, tcd

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]

# %%


def cd1(row, res, quantile, fp='/simtgraph5N.npz'):
    # global pcs, sel, membership, membdf
    #    global sel, nodenames
    i = fp

    try:

        core = [megamap.get(x, x) for x in pd.read_csv(
            row[0]+'/core.tsv', sep='\t', header=None)[0]]
        coreu = [x.replace(' ', '_') for x in core]
        flatadj = pd.read_hdf(row[0].replace(
            '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')+'/adjNN.h5', key='df')

        articles = list(flatadj.columns)
        ai = articles

        # print('generating graphs', len(articles))
# =============================================================================
#
# =============================================================================

        print('loading data', i)
        pcs = np.nan_to_num(np.load(row[0].replace(
            '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')+i)['arr_0'])
        q = pd.read_hdf(row[0].replace(
            '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')+'/quantilesNN.npz')
        pcs = np.where(pcs >= q.loc[quantile], pcs, 0)
        # articles = pd.read_hdf(row[0]+'/coltitles.h5')

        print('generating graphs', len(articles))

        glist = []

        tlen = 55
        for n in range(tlen):
            sel = pd.DataFrame(pcs[n, :, :], index=articles,
                               columns=articles).stack().reset_index()
            sel = sel[sel[0] != 0]
            sel.columns = ['source', 'target', 'weight']
            # sel['source'] = sel['source'].map(nodenames)
            # sel['target'] = sel['target'].map(nodenames)
            sel = sel[sel['source'] != 'Main Page']
            sel = sel[sel['target'] != 'Main Page']
            tuples = [tuple(x) for x in sel.values]
            glist.append(igraph.Graph.TupleList(tuples, directed=False, edge_attrs=[
                         'weight']).simplify(combine_edges='first'))
            # glist[n].vs["label"] = [nodenamesr[x] for x in glist[n].vs["name"]]
            glist[n].vs["slice"] = n

        if sum([len(x.vs) for x in glist]) == 0:
            # print('empty')
            return [i, 'error', 'empty']

        # print('generating agg network')
        igname = [{x.index: x['name'] for x in glist[y].vs}
                  for y in range(tlen)]
        # ignamer = [{v:k for k,v in x.items()} for x in igname]

        del pcs
        del sel
#        del tuples

        return [res, glist, igname, coreu, articles]

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return [i, 'error', ex]


def cd2(res, glist, igname, coreu, articles):
    try:
        print('running community detection')

        membership, improvement = la.find_partition_temporal(
            glist, la.CPMVertexPartition, vertex_id_attr='name', interslice_weight=1, resolution_parameter=res)
        membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
                           for n, x in enumerate(membership)], axis=1, sort=True)

        print('getting comm data')
        for n in range(55):
            cattr = membdf[n].copy()
            cattr.index = cattr.index.map({v: k for k, v in igname[n].items()})
            glist[n].vs['community'] = cattr.sort_index().values
            glist[n].vs["label"] = glist[n].vs["name"]

        print('getting collmems')
        collmems = {}
        t0comms = membdf.loc[set(coreu) & set(articles), 27]
        membdfc = membdf.loc[set(coreu) & set(articles)].T

        cmd = {x: [] for x in set(t0comms.dropna().values)}
        for x in t0comms.dropna().iteritems():
            cmd[x[1]].append(x[0])

        for k, c in cmd.items():
            segt = ((membdfc[c] == k).sum(axis=1) > .5*len(c))
            tf = segt == segt.shift(1)
            diff = np.where(tf.values == False)[0]-27
            beg = 27+diff[diff <= 0].max()
            try:
                end = 26+diff[diff > 0].min()
            except ValueError:
                end = 54
            tstep = membdfc.loc[beg:end, c].index
            t = membdf.T.loc[tstep][membdf.T.loc[tstep]
                                    == k].T.dropna(how='all')
            gg = {x: set(t[x].dropna().index) for x in t}
            js = pd.Series({k: jac(set(gg[27]), set(v))
                           for k, v in gg.items()})
            t = t[js[js > .5].index].dropna(how='all')
            collmems['---'.join(sorted(c))] = t

        # uniquify sets
        for k, v in collmems.items():
            for l, u in collmems.items():
                if k != l:
                    for s in set(v.index) & set(u.index):
                        f1 = len(v.loc[s].dropna())/len(v.loc[s])
                        f2 = len(u.loc[s].dropna())/len(u.loc[s])
                        if f1 > f2:
                            collmems[l] = collmems[l].drop(s)
                        elif f2 > f1:
                            collmems[k] = collmems[k].drop(s)
                        elif len(v) < len(u):
                            collmems[l] = collmems[l].drop(s)
                        elif len(v) >= len(u):
                            collmems[k] = collmems[k].drop(s)

        tcd = {k: tcentrality(glist, v) for k, v in collmems.items()}

        return membdf, collmems, tcd

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]


# %%
def tcentrality(gl, comm):
    cdf = pd.DataFrame(index=comm.index, columns=comm.columns)
    for x in comm.columns:
        sg = gl[x].subgraph(comm[x].dropna().index)
        cdf[x] = pd.Series(sg.pagerank(weights='weight'), index=sg.vs['name'])
    return cdf.mean(axis=1)


def centrality(g, comm, nodenamesr):
    cdf = pd.Series(index=comm)

    sg = g.subgraph(pd.Series(comm).map(nodenamesr))
    cdf = pd.Series(sg.pagerank(), index=sg.vs['name'])

    return cdf


def jac(x, y):
    return len(x & y)/len(x | y)

# %%


# %%


def months_range(start, stop):
    xr = start
    stop2 = stop
    stop2 = stop2.replace(day=monthrange(stop2.year, stop2.month)[1])
    months = []
    while xr <= stop2:
        months.append('%d%02d' % (xr.year, xr.month))
        xr += relativedelta(months=1)
    return months


with open('/Volumes/PGPassport/active_data_2019/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

with open('/Volumes/PGPassport/active_data_2019/megamap.json') as json_data:
    megamap = json.load(json_data)
    json_data.close()

print('Loading1')

evlsn = pd.read_hdf('active_data/evlsN.h5', key='df')

edf = evlsn
# %%

doneSSD = set([x[:-9]
              for x in glob.glob('/Volumes/Samsung_T3/evtest/*/adjNN.h5')])

# %%
donesamp = set([x.replace('active_data/robusttest21', '/Volumes/Samsung_T3/evtest')
               for x in glob.glob('active_data/robusttest21/*')])
# %%
evsamp = random.sample(doneSSD-donesamp, 2)
# %%
resrange = np.exp(np.arange(-9, 1, 1))
# outd = {}
for n, x in enumerate(evsamp):
    try:
        if x in outd.keys():
            continue
        out = {}
        for q in [0, 0.5, 0.9]:
            out[q] = commmdectFRparallel(
                [x, evlsn.loc[x]], resrange, quantile=q, fp='/simtgraph5NN.npz', flat=True)
        outd[x] = out
    except Exception as ex:
        print(ex)
        # %%

for k, v in outd.items():
    print(k)
    try:
        os.mkdir('active_data/robusttest21/%s' % k.split('evtest/')[-1])
    except:
        continue
    for k2, v2 in v.items():
        try:
            for k3, v3 in v2[0].items():
                v3.to_hdf('active_data/robusttest21/%s/membflat_%s.h5' %
                          (k.split('evtest/')[-1], k2), key=str(k3))
            for k3, v3 in v2[1].items():
                for k4, v4 in v3.items():
                    pd.Series(v4).to_hdf('active_data/robusttest21/%s/cmflat_%s.h5' %
                                         (k.split('evtest/')[-1], k2), key='_'.join([str(k3), str(k4)]))
            for k3, v3 in v2[2].items():
                v3.to_hdf('active_data/robusttest21/%s/membT_%s.h5' %
                          (k.split('evtest/')[-1], k2), key=str(k3))
            for k3, v3 in v2[3].items():
                for k4, v4 in v3.items():
                    v4.to_hdf('active_data/robusttest21/%s/cmT_%s.h5' %
                              (k.split('evtest/')[-1], k2), key='_'.join([str(k3), str(k4)]))
        except Exception as ex:
            print(ex)

# %%
dones = glob.glob('active_data/robusttest21/*')
# %%
resrange = np.exp(np.arange(-9, 1, 1))

for x in dones:
    if x.replace('active_data/robusttest21', '/Volumes/Samsung_T3/evtest') in outd.keys():
        continue
    out = {}
    for q in [0, 0.5, 0.9]:
        mdfd = {}
        mdffd = {}
        for r in resrange:
            mdfd[r] = pd.read_hdf(x+'/membT_%s.h5' % q, key=str(r))
            mdffd[r] = pd.read_hdf(x+'/membflat_%s.h5' % q, key=str(r))

        out[q] = [mdffd, None, mdfd, None]

    outd[x] = out

# %%
for k in l:
    if k in dones:
        del outd[k]

# %%
# organise data
mdfqd = {}
for q in [0, 0.5, 0.9]:
    mf = {}
    for k, v in outd.items():
        mdf_flat = pd.DataFrame()
        mdf = pd.DataFrame()
        for k2, v2 in v[q][0].items():
            mdf_flat[k2] = v2
        for k2, v2 in v[q][2].items():
            mdf[k2] = v2[27]
        mf[k] = [mdf_flat, mdf]
    mdfqd[q] = mf

    # %%
amidfd = {}
csdfd = {}

for q in [0, 0.5, 0.9]:
    print(q)
    amidf = pd.DataFrame(index=resrange[1:])
    csdf = pd.DataFrame(index=resrange[1:])
    for n, (k, v) in enumerate(mdfqd[q].items()):
        print(n/len((mdfqd[q])))
        data = []
        datc = []
        for n in range(1, len(v[1].columns)):
            ct = v[1][[resrange[n], resrange[n-1]]]
            ct = ct.dropna()
            c1 = ct[resrange[n]]
            c2 = ct[resrange[n-1]]

            data.append(sklearn.metrics.adjusted_mutual_info_score(c1, c2))
            c1 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c1).items()})
            c2 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c2).items()})
            datc.append(sim.element_sim(c1, c2, alpha=0.9))

        amidf[k] = data
        csdf[k] = datc

    amidfd[q] = amidf
    csdfd[q] = csdf
    # %%
plt.figure(figsize=[12, 8])
colors = {0: 'b', 0.5: 'r', 0.9: 'm'}
for q in [0, 0.5, 0.9]:
    plt.plot(amidfd[q].mean(axis=1), label='AMI - %.1f' % q, c=colors[q], lw=3)
for q in [0, 0.5, 0.9]:
    plt.plot(csdfd[q].mean(axis=1), label='CluSim - %.1f' %
             q, c=colors[q], ls=':', lw=3)
    # plt.plot(amidfd[q].median(axis=1), label = 'Median - %.1f' %q, c = colors[q], ls = ':')
plt.xscale('log')
plt.xlabel('Resolution')
# plt.ylabel('AMI')
# plt.show()
# for q in [0, 0.5, 0.9]:
# plt.plot(csdfd[q].median(axis=1), label = 'Median - %.1f' %q, c = colors[q], ls = ':')
plt.legend(ncol=2)
plt.xscale('log')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.title('Partition Stability')
plt.show()

# %%

np.mean(resrange[-3:-1])
=0.25

# %%
doneHDD = set([x[:-9]
              for x in glob.glob('/Volumes/PGPassport/evtLocal/*/adjNN.h5')])
donelocal = set([x[:-9] for x in glob.glob('active_data/tempevs/*/adjNN.h5')])
donedone = set([x.replace('/Volumes/PGPassport/evtLocal', 'active_data/tempevs')[:-16]
               for x in glob.glob('/Volumes/PGPassport/evtLocal/*/tcweights.h5.h5')])
# %%
edf = evlsn.loc[[x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
                 for x in donelocal-donedone]].sort_values('len')[evlsn['len'] < 2000]
step = 60
quantile = 0
res = 0.25
for n in range(0, len(edf), step):
    print(n)
    try:
        out1 = Parallel(n_jobs=-1, verbose=10)(delayed(cd1)(
            [x, evlsn.loc[x]], res, quantile, fp='/simtgraph5NN.npz') for x in edf.index[n:n+step])
        out2 = Parallel(n_jobs=-1, verbose=10)(delayed(cd2)(*x)
                                               for x in out1 if len(x) == 5)

        names = [x for m, x in enumerate(
            edf.index[n:n+step]) if len(out1[m]) == 5]
        for m, i in enumerate(out2):
            try:
                # i[0].to_hdf('%s/%s' %(names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'membdf.h5'), key='df')
                # for k, v in i[1].items()
                # v.to_hdf('%s/%s.h5' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), k), key='df')
                # i[2].to_hdf('%s/%s' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'tcweights.h5'), key='df')

                i[0].to_hdf('%s/%s' % (names[m].replace('Samsung_T3/evtest',
                            'PGPassport/evtLocal'), 'membdf.h5'), key='df')
                for k, v in i[1].items():
                    v.to_hdf(
                        '%s/%s.h5' % (names[m].replace('Samsung_T3/evtest', 'PGPassport/evtLocal'), k), key='df')
                for k, v in i[2].items():
                    v.to_hdf('%s/%s.h5' % (names[m].replace(
                        'Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key=k)
                    # i[2].to_hdf('%s/%s' %(names[m].replace('Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key='df')
            except Exception as ex:
                # raise
                print(ex)
    except Exception as ex:
        print(ex)
# %%
doneHDD = set([x[:-9]
              for x in glob.glob('/Volumes/PGPassport/evtLocal/*/adjNN.h5')])
donelocal = set([x[:-9] for x in glob.glob('active_data/tempevs/*/adjNN.h5')])
donedone = set([x.replace('/Volumes/PGPassport/evtLocal', 'active_data/tempevs')[:-16]
               for x in glob.glob('/Volumes/PGPassport/evtLocal/*/tcweights.h5.h5')])

edf = evlsn.loc[[x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
                 for x in donelocal-donedone]].sort_values('len')[evlsn['len'] < 3000]
step = 30
quantile = 0
res = 0.25
for n in range(0, len(edf), step):
    print(n)
    try:
        out1 = Parallel(n_jobs=8, verbose=10)(delayed(cd1)(
            [x, evlsn.loc[x]], res, quantile, fp='/simtgraph5NN.npz') for x in edf.index[n:n+step])
        out2 = Parallel(n_jobs=8, verbose=10)(delayed(cd2)(*x)
                                              for x in out1 if len(x) == 5)

        names = [x for m, x in enumerate(
            edf.index[n:n+step]) if len(out1[m]) == 5]
        for m, i in enumerate(out2):
            try:
                # i[0].to_hdf('%s/%s' %(names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'membdf.h5'), key='df')
                # for k, v in i[1].items()
                # v.to_hdf('%s/%s.h5' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), k), key='df')
                # i[2].to_hdf('%s/%s' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'tcweights.h5'), key='df')

                i[0].to_hdf('%s/%s' % (names[m].replace('Samsung_T3/evtest',
                            'PGPassport/evtLocal'), 'membdf.h5'), key='df')
                for k, v in i[1].items():
                    v.to_hdf(
                        '%s/%s.h5' % (names[m].replace('Samsung_T3/evtest', 'PGPassport/evtLocal'), k), key='df')
                for k, v in i[2].items():
                    v.to_hdf('%s/%s.h5' % (names[m].replace(
                        'Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key=k)
                    # i[2].to_hdf('%s/%s' %(names[m].replace('Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key='df')
            except Exception as ex:
                # raise
                print(ex)
    except Exception as ex:
        print(ex)
# %%
doneHDD = set([x[:-9]
              for x in glob.glob('/Volumes/PGPassport/evtLocal/*/adjNN.h5')])
donelocal = set([x[:-9] for x in glob.glob('active_data/tempevs/*/adjNN.h5')])
donedone = set([x.replace('/Volumes/PGPassport/evtLocal', 'active_data/tempevs')[:-16]
               for x in glob.glob('/Volumes/PGPassport/evtLocal/*/tcweights.h5.h5')])
# %%
edf = evlsn.loc[[x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
                 for x in donelocal-donedone]].sort_values('len')
step = 4
quantile = 0
res = 0.25
for n in range(0, len(edf), step):
    print(n)
    try:
        out1 = Parallel(n_jobs=4, verbose=10)(delayed(cd1)(
            [x, evlsn.loc[x]], res, quantile, fp='/simtgraph5NN.npz') for x in edf.index[n:n+step])
        out2 = Parallel(n_jobs=4, verbose=10)(delayed(cd2)(*x)
                                              for x in out1 if len(x) == 5)

        names = [x for m, x in enumerate(
            edf.index[n:n+step]) if len(out1[m]) == 5]
        for m, i in enumerate(out2):
            try:
                # i[0].to_hdf('%s/%s' %(names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'membdf.h5'), key='df')
                # for k, v in i[1].items()
                # v.to_hdf('%s/%s.h5' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), k), key='df')
                # i[2].to_hdf('%s/%s' %((names[m].replace('/Volumes/Samsung_T3/evtest', 'active_data/evtLocal/flat'), 'tcweights.h5'), key='df')

                i[0].to_hdf('%s/%s' % (names[m].replace('Samsung_T3/evtest',
                            'PGPassport/evtLocal'), 'membdf.h5'), key='df')
                for k, v in i[1].items():
                    v.to_hdf('%s/%s.h5' % (names[m].replace(
                        'Samsung_T3/evtest', 'PGPassport/evtLocal'), k.replace('/', ':')), key='df')
                for k, v in i[2].items():
                    v.to_hdf('%s/%s.h5' % (names[m].replace(
                        'Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key=k)
                    # i[2].to_hdf('%s/%s' %(names[m].replace('Samsung_T3/evtest', 'PGPassport/evtLocal'), 'tcweights.h5'), key='df')
            except Exception as ex:
                # raise
                print(ex)
    except Exception as ex:
        print(ex)
        # %%

evsamp = [x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
          for x in random.sample(donedone, 50)]
# %%
out = Parallel(n_jobs=-1, verbose=10)(delayed(cdflatR)
                                      ([x, evlsn.loc[x]], resrange, fp='/simtgraph5NN.npz') for x in evsamp)
# %%
amidf = pd.DataFrame()
for n, x in enumerate(out):
    tpart = pd.read_hdf(evsamp[n].replace(
        'Samsung_T3/evtest', 'PGPassport/evtLocal')+'/membdf.h5')[27]
    ami = pd.Series()
    for r in resrange:
        ct = pd.DataFrame([x[0][r], tpart], index=[0, 1]).T.dropna()
        ami[r] = sklearn.metrics.adjusted_mutual_info_score(ct[0], ct[1])
        # x[0][r]
    amidf[evsamp[n]] = ami
# %%


def jac(x, y):
    return len(set(x) & set(y))/len(set(x) | set(y))


def wjac(x, y):
    # x = (~x.isna()).mean(axis=1)
    # y = (~y.isna()).mean(axis=1)
    df = pd.DataFrame([x, y]).T.fillna(0)
    return (df.min(axis=1)/df.max(axis=1)).mean()


allev = [x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
         for x in donedone]
chout = []
for n in range(0, len(donedone), 200):
    print(n)
    chout.append(Parallel(n_jobs=-1, verbose=10)(delayed(cdflatR)
                 ([x, evlsn.loc[x]], resrange, fp='/simtgraph5NN.npz') for x in evsamp))
# %%

# wjs = pd.Series()
# js = pd.Series()
sf = ['adjNN.h5', 'coltitlesNN.h5', 'membdf.h5',
      'tcweights.h5.h5', 'quantilesNN.npz', 'simtgraph5NN.npz']
allev = [x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
         for x in donedone]
step = 200
# %%

wjdone = set([x[:-8].replace('/Volumes/PGPassport/evtLocal', 'active_data/tempevs')
             for x in glob.glob('/Volumes/PGPassport/evtLocal/*/wjac.h5')])
doneHDD = set([x[:-9].replace('/Volumes/PGPassport/evtLocal', 'active_data/tempevs')
              for x in glob.glob('/Volumes/PGPassport/evtLocal/*/adjNN.h5')])

allev = [x.replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest')
         for x in doneHDD-wjdone]

# %%
core_D = {f[:-9]: [megamap.get(x, x) for x in pd.read_csv(f, sep='\t', header=None)[0]]
          for f in glob.glob('/Volumes/Samsung_T3/evtest/*/core.tsv')}
# %%
flatadj_D = {x[:-9].replace('active_data/tempevs', '/Volumes/Samsung_T3/evtest'): pd.read_hdf(x, key='df') for x in glob.glob('active_data/tempevs/*/adjNN.h5')}
# %%


def cmmatcher(n, x, colm):
    try:
        # colm =
        wjdf = pd.DataFrame()
        jdf = pd.DataFrame()
        for k, v in colm.items():
            for r in resrange:
                maw = []
                ma = []
                for k2, v2 in x[2][r].items():
                    maw.append(wjac(v, v2))
                    ma.append(jac(set(v.index), set(v2.index)))
                wjdf.loc[k, r] = max(maw)
                jdf.loc[k, r] = max(ma)

        wjdf.index = pd.Series(wjdf.index).apply(
            lambda x: allev[m+n].split('evtest/')[-1] + '/'+x+'.h5')
        jdf.index = pd.Series(jdf.index).apply(
            lambda x: allev[m+n].split('evtest/')[-1] + '/'+x+'.h5')

        return wjdf, jdf
    except Exception as ex:
        return ['error', n, x, ex]


    # %%
DD = sorted([x for x in glob.glob(
    '/Volumes/PGPassport/evtLocal/*/tcweights.h5.h5')])
DDC = {}
for n, x in enumerate(DD):
    print(n/len(DD))
    with pd.HDFStore(x, mode='r') as hdf:
        for k in hdf.keys():
            DDC[x[:-16]+k+'.h5'] = pd.read_hdf(x, key=k)

            # %%
DDC2 = {}
for n, (k, v) in enumerate(DDC.items()):
    print(n/len(DDC))
    if k.split('/')[-2] not in DDC2.keys():
        DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
    else:
        DDC2[k.split('/')[-2]][k.split('/')[-1]] = v

# %%
werrs = []
sf = ['adjNN.h5', 'coltitlesNN.h5', 'membdf.h5',
      'tcweights.h5.h5', 'quantilesNN.npz', 'simtgraph5NN.npz']
step = 200
for m in range(0, len(allev), step):
    print(m)
    out = Parallel(n_jobs=-1, verbose=10)(delayed(cdflatR)
                                          ([x, evlsn.loc[x]], resrange, core_D[x], flatadj_D.get(x, pd.DataFrame())) for x in allev[m:m+step])

    print('Getting Js')
    wws = Parallel(n_jobs=-1, verbose=10)(delayed(cmmatcher)(n, x,
                                                             DDC2.get(allev[m+n].split('/')[-1], {})) for n, x in enumerate(out))

    print('Writing Js')
    for n, x in enumerate(wws):
        try:
            x[0].to_hdf('/Volumes/PGPassport/evtLocal/%s/wjac.h5' %
                        allev[m+n].split('evtest/')[-1], key='df')
            x[1].to_hdf('/Volumes/PGPassport/evtLocal/%s/jac.h5' %
                        allev[m+n].split('evtest/')[-1], key='df')
        except Exception as ex:
            print(ex)
            werrs.append([n, x, ex])

            # %%
combos = [(DDC[k1], DDC[k2]) for n, k1 in enumerate(sorted(DDC.keys()))
          for m, k2 in enumerate(sorted(DDC.keys())) if n < m]
print(len(combos))

# %%
# jll = []
# for n in range(int(1E6), len(combos), int(1E6)):
n = 60000000
print(n, len(combos))
jacsw = Parallel(n_jobs=-1, verbose=3)(delayed(wjac)
                                       (x[0], x[1]) for x in combos[n:n+int(1E6)])
# jll.append(jacsw)
with open('active_data/jw/jacsw%.1f.json' % (n/1E6), 'w+') as f:
    json.dump(jacsw, f)


# %%
n = 82000000

for n in range(0, 13300000, 1000000):
    with open('/Users/Patrick/active_data/jw/jacsw%.1f.json' % (n/1E6), 'r') as f:
        j1 = json.load(f)

    with open('/Users/Patrick/active_data/jw copy/jacsw%.1f.json' % (n/1E6), 'r') as f:
        j2 = json.load(f)

    print(n, j1 == j2)
# %%
# jt = pd.DataFrame([j1, j2])
del combos
jll = []
for n in range(0, 132787956, int(1E6)):
    print(n, 132787956)
    with open('active_data/jw/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll.extend(json.load(f))


# %%
arr = np.zeros((len(DDC), len(DDC)))
# %%
# 82 onwards
ixsall = np.array(np.triu_indices(len(DDC), 1))
# %%
ixspre = ixsall[:, :82000000]
ixspost = ixsall[:, 82000000:]
# %%
ixspost = vec_translate(ixspost, imp)
# %%
ixc = np.concatenate([ixspre, ixspost], axis=1)
# %%
ixt = (ixc[0], ixc[1])
# %%
arr[ixt] = jll
# %%
simdf = pd.DataFrame(arr, index=sorted(DDC.keys()), columns=sorted(DDC.keys()))
del arr, ixsall, ixspre, ixspost, ixc, jll
# %%
sds = simdf.stack()
del simdf
sds = sds[sds > 0]
# %%
el = sds.reset_index()
del sds
# %%
el.columns = ['source', 'target', 'weight']
el.source = el.source.str.split('evtLocal/').str[-1]
el.target = el.target.str.split('evtLocal/').str[-1]

tuples = [tuple(x) for x in el.values]
del el
# %%
G = igraph.Graph.TupleList(tuples, directed=False, edge_attrs=['weight'])
print(len(G.vs), len(G.es))
# %%
partitions = {}
for r in np.exp(np.arange(-5, 0.1, 0.1)):
    print(r)
    partitions[r] = la.find_partition(
        G, la.CPMVertexPartition, resolution_parameter=r, weights='weight')

# %%
pi = pd.DataFrame()
for r in np.exp(np.arange(-5, 0.1, 0.1)):
    print(r)
    pi[r] = pd.Series({y: n for n, x in enumerate(partitions[r])
                      for y in x}).sort_index()
pi.index = G.vs['name']
# %%
# data = []#%%#%%
# datc = []
for n in range(47, len(pi.columns)):
    print(n)
    ct = pi[[pi.columns[n-1], pi.columns[n]]]
    ct = ct.dropna()
    c1 = ct[pi.columns[n]]
    c2 = ct[pi.columns[n-1]]

    data.append(sklearn.metrics.adjusted_mutual_info_score(c1, c2))
    c1 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c1).items()})
    c2 = Clustering(elm2clu_dict={k: [v] for k, v in dict(c2).items()})
    datc.append(sim.element_sim(c1, c2, alpha=0.9))

# %%
pi.to_hdf('active_data/H_partition.h5', key='df')
pd.Series(data, index=pi.columns[1:]).to_hdf('active_data/H_ami.h5', key='df')
pd.Series(datc, index=pi.columns[1:]).to_hdf(
    'active_data/H_clusim.h5', key='df')
# %%
pi = pd.read_hdf('active_data/H_partition.h5', key='df')
data = pd.read_hdf('active_data/H_ami.h5', key='df')
datc = pd.read_hdf('active_data/H_clusim.h5', key='df')
# %%
plt.figure(figsize=[12, 8])
pd.Series(data, index=pi.columns[1:]).plot(label='AMI', lw=3)
pd.Series(datc, index=pi.columns[1:]).plot(label='CluSim', lw=3)
plt.legend()
plt.xscale('log')
plt.title('Higher Level Partition Stability')
plt.xlabel('Resolution')
plt.ylabel('Similarity')
plt.show()
# %%
r = 0.12
partition = la.find_partition(
    G, la.CPMVertexPartition, resolution_parameter=r, weights='weight')
pip = pd.Series({y: n for n, x in enumerate(partition)
                for y in x}).sort_index()
# %%
pip.index = G.vs['name']
pip.to_hdf('active_data/H_final_partition.h5', key='df')
# gjdf = pd.DataFrame()
# %%
gjdf = pd.DataFrame()

for n, m in enumerate(DD):
    print(n/len(DD))
    wj = pd.read_hdf(m.replace('tcweights.h5.h5', 'wjac.h5'), key='df')
    gjdf = gjdf.append(wj)

# %%
gjdf['max'] = gjdf.max(axis=1)
gjdf['comm'] = pip
# %%
meanwj = gjdf.groupby('comm').mean()
medianwj = gjdf.groupby('comm').median()
meanwj['count'] = gjdf['comm'].value_counts()
medianwj['count'] = gjdf['comm'].value_counts()

# %%
meanwj = meanwj.sort_values('count', ascending=False)
# %%

for i in meanwj[meanwj['Name'].isna()].index:
    print(i)
    evs = gjdf[gjdf['comm'] == i].index.str.split('/').str[0]
    cores = pd.Series([x for y in gjdf[gjdf['comm'] == i].index.str.split(
        '/').str[-1].str[:-3] for x in y.split('---')]).value_counts()
    allarts = pd.concat([DDC['/Volumes/PGPassport/evtLocal/'+x] for x in gjdf[gjdf['comm'] == i].index]
                        ).reset_index().groupby('index').sum().sort_values(by=0, ascending=False)
    n = 0
    while True:
        print('Stories')
        print('\n\n'.join(storiesdf.loc[sorted(set(evs) & set(
            storiesdf.index)), 'Text'].iloc[n:n+10].values))
        print('\n\n')
        print('Cores')
        print(cores.iloc[n:n+10])
        print('All arts')
        print(allarts.iloc[n:n+10])

        mi = input('More info?\n')

        if mi.lower() == 'n':
            break
        n += 10
    nn = input('Enter name\n')
    meanwj.loc[i, 'Name'] = nn

    print('##'*20)
# %%
ixs = []

for i in gjdf.index:
    if i[-3:] != '.h5':
        ixs.append(i+'.h5')
    else:
        ixs.append(i)
# %%
gjdf.index = ixs
# %%
errs2 = []
for n, i in enumerate(gjdf[gjdf['PROM'].isna()].index):
    print(n/len(gjdf[gjdf['PROM'].isna()]))

    try:
        cm = DDC['/Volumes/PGPassport/evtLocal/'+i]
        arts = set(cm.index)
        date = pd.to_datetime(i[:8])

        start = date-relativedelta(days=30)
        stop = date+relativedelta(days=30)

        # start1 = date-relativedelta(days=int((27-cm.columns[0])+30))
        # end2 = date+relativedelta(days=int((cm.columns[-1]-27)+31))

        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        # months2 = months_range(pd.to_datetime(start1), pd.to_datetime(end2))

        artsr = [y for x in arts for y in redir_arts_map.get(
            x.replace(' ', '_'), [x])]

        cc = []
        for m in months:
            cc.append(tsl['/Volumes/home/Python Projects/Data/PageViews_Rep/hourly/hourly_t_series_%s.h5' % m][sorted({x.replace(
                ' ', '_') for x in artsr} & set(tsl['/Volumes/home/Python Projects/Data/PageViews_Rep/hourly/hourly_t_series_%s.h5' % m].columns))])

        ts = pd.concat(cc, sort=True).fillna(
            0).loc[start:stop-relativedelta(days=1)]
        ts.columns = [megamap.get(x, x) for x in ts.columns]
        ts = ts.T.groupby(ts.T.index).sum().T

        weights = cm

        wts = ts.dot(weights.reindex(ts.columns).fillna(0))

        premed = wts.loc[date-relativedelta(days=30):date].median()
        # postmed = wts.loc[date+relativedelta(days=30):].median()

        peakex = (wts.loc[date-relativedelta(days=1):date +
                  relativedelta(days=1)]-premed).max()
        peakdev = peakex/(wts.loc[date-relativedelta(days=30):date].quantile(.75) -
                          wts.loc[date-relativedelta(days=30):date].quantile(.25))

        gjdf.loc[i, ['PROM', 'MAG', 'DEV']] = [premed, peakex, peakdev]

    except Exception as ex:
        # raise
        print(i, ex)
        errs2.append((i, ex))


# %%
for f in ['count', 'PROM', 'MAG', 'DEV']:
    for i in meanwj.sort_values(f, ascending=False)[meanwj['count'] >= 10].iloc[:20][meanwj['Name'] == '?'].index:
        print(i, len(meanwj.sort_values(f)[
              meanwj['count'] >= 5].iloc[:50][meanwj['Name'].isna()]))
        evs = gjdf[gjdf['comm'] == i].index.str.split('/').str[0]
        cores = pd.Series([x for y in gjdf[gjdf['comm'] == i].index.str.split(
            '/').str[-1].str[:-3] for x in y.split('---')]).value_counts()
        allarts = pd.concat([DDC['/Volumes/PGPassport/evtLocal/'+x] for x in gjdf[gjdf['comm'] == i].index]
                            ).reset_index().groupby('index').sum().sort_values(by=0, ascending=False)
        n = 0
        print(f, meanwj.loc[i, f])
        while True:
            print('Stories')
            print('\n\n'.join(storiesdf.loc[sorted(set(evs) & set(
                storiesdf.index)), 'Text'].iloc[n:n+10].values))
            print('\n\n')
            print('Cores')
            print(cores.iloc[n:n+10])
            print('All arts')
            print(allarts.iloc[n:n+10])

            mi = input('More info?\n')

            if mi.lower() == 'n':
                break
            n += 10
        nn = input('Enter name\n')
        meanwj.loc[i, 'Name'] = nn

        print('##'*20)

# %%
outdf = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    outdf[f] = meanwj.sort_values(f, ascending=False)[
        meanwj['count'] >= 10].iloc[:20]['Name'].reset_index(drop=True)
# %%
UQ = meanwj[0.6065306597126334].quantile(.75)
MM = meanwj[0.6065306597126334].quantile(.5)
LQ = meanwj[0.6065306597126334].quantile(.25)
# %%
UQ = 0.85
# MM = meanwj[0.6065306597126334].quantile(.5)
LQ = 0.4


def colourer(x):
    if x == '?':
        return None
    ou = meanwj[meanwj['Name'] == x]['max']
    if len(ou) != 1:
        print(x)
        raise
    if ou.iloc[0] > UQ:
        return '\cellcolor{green}' + x.replace('&', '\&')
    # elif ou.iloc[0]>MM: return '\cellcolor{yellow}'+ x.replace('&', '\&')
    elif ou.iloc[0] > LQ:
        return '\cellcolor{orange}' + x.replace('&', '\&')
    else:
        return '\cellcolor{red}' + x.replace('&', '\&')


# %%
odfv = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    odfv[f] = outdf[f].apply(colourer)

odfv.columns = ['# Events', 'Prominence', 'Magnitude', 'Deviance']

# %%
plt.figure(figsize=[12, 8])
sns.kdeplot(gjdf['max'], fill=True, lw=3)
plt.title('Structural Similarity Distribution')
plt.xlabel('Community Structural Similarity')
plt.ylabel('Density')
plt.show()
