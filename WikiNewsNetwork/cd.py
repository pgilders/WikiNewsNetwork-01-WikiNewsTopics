#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:23:53 2022

@author: 
"""

import os
import pandas as pd
import igraph
import leidenalg as la
import numpy as np
import sklearn.metrics
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import RobustScaler
from clusim.clustering import Clustering
import clusim.sim as sim
import WikiNewsNetwork.utilities as utilities
import WikiNewsNetwork.processing as processing

# %% Temporal community detection


def read_t_graph(event, core):
    """
    Read temporal network data and generate igraph networks.

    Parameters
    ----------
    event : str
        Event name (folder location).
    core : list
        List of core articles for event.

    Returns
    -------
    tuple
        glist2 : List of igraph networks for each timestep.
        igname : List of dictionaries for vertex names.

    """
    try:
        # print('generating graphs', len(articles))

        # print('loading data')
        pcsel = np.nan_to_num(np.load(event + '/simtgraphelNN.npz')['arr_0'])
        ixs = np.nan_to_num(np.load(event + '/simtgraphixNN.npz')['arr_0'])
        artixdict = pd.read_hdf(event + '/coltitlesNN.h5', key='df').to_dict()

        glist = []
        for n in range(pcsel.shape[0]):
            sel = pd.DataFrame(
                np.append(ixs, pcsel[n].reshape(pcsel.shape[1], 1), axis=1))
            sel.columns = ['source', 'target', 'weight']
            sel['source'] = sel['source'].map(artixdict)
            sel['target'] = sel['target'].map(artixdict)
            tuples = [tuple(x) for x in sel.values]
            glist.append(igraph.Graph.TupleList(tuples, directed=False,
                                                edge_attrs=['weight']))
            glist[n].vs["slice"] = n

        if sum([len(x.vs) for x in glist]) == 0:
            # print('empty')
            return [event, 'error', 'empty']

        # print('generating agg network')
        igname = [{x.index: x['name'] for x in glist[y].vs}
                  for y in range(len(glist))]
        igname_rev = [{v: k for k, v in y.items()} for y in igname]
        del pcsel, ixs, sel

        glist2 = []
        for n, gl in enumerate(glist):
            coreids = [igname_rev[n][x] for x in core if x in igname_rev[n]]
            components = [c for c in gl.components()
                          if any([x in c for x in coreids])]
            vertexes = [y for x in components for y in x]
            glist2.append(gl.subgraph(vertexes))

        igname = [{x.index: x['name'] for x in glist2[y].vs}
                  for y in range(len(glist2))]

        return glist2, igname

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        raise
        return [event, 'error', ex]


def temporal_community_detection(glist, res, interslice_weight=1):
    """
    Take temporal network and run community detection at specified resolution.

    Parameters
    ----------
    glist : list
        List of igraph networks for each timestep for an event.
    res : float
        Community detection resolution.
    interslice_weight : float, optional
        Edge weight between equivalent node in adjacent temporal layers. The
        default is 1.

    Returns
    -------
    DataFrame
        DataFrame with community memberships at each timestep.

    """
    try:
        # print('running community detection')

        membership = la.find_partition_temporal(
            glist, la.CPMVertexPartition, vertex_id_attr='name',
            interslice_weight=interslice_weight, resolution_parameter=res)[0]
        membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
                            for n, x in enumerate(membership)],
                           axis=1, sort=True)

        return membdf

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]

def cd_demo(timeseries, network, similarity_metric=processing.pearsonr_ixs, sim_kwargs={},
            algorithm=la.find_partition_temporal,
            alg_kwargs={'partition_type': la.CPMVertexPartition,
                        'vertex_id_attr': 'name',
                        'interslice_weight': 1, 'resolution_parameter': 1},
            res=None, tau=None,
            mode='igraph'):
    """
    

    Parameters
    ----------
    timeseries : Pandas DataFrame / Numpy Array
        Scaled time series for page views of each article.
    network : Pandas DataFrame / igraph
        Representing the network.
    similarity_metric : Function, optional
        Function to calculate time series similarity.
        The default is processing.pearsonr_ixs.
    sim_kwargs : dict, optional
        Arguments for similarity measure. The default is {}.
    algorithm : Function, optional
        Community detection algorithm to be applied to temporal network.
        The default is la.find_partition_temporal.
    alg_kwargs : dict, optional
        Arguments for community detection algorithm.
        The default is {'partition_type': la.CPMVertexPartition,
                        'vertex_id_attr': 'name', 'interslice_weight': 1,
                        'resolution_parameter': 1}.
    res : float, optional
        Resolution parameter, overrides alg_kwargs. The default is None.
    tau : float, optional
        Interlayer coupling parameter, overrides alg_kwargs. The default is None.
    mode : str, optional
        Future parameter to adapt for networkx & other input.
        The default is 'igraph'.

    Returns
    -------
    cd_output :
        Output of the supplied temporal community detection function.
    nodename_dict : dict
        Mapping of node index to names for each network layer.

    """

    for k, v in {'resolution_parameter': res, 'interslice_weight': tau}.items():
        if v:
            alg_kwargs[k] = v

    if type(network) == pd.DataFrame:
        adj = network.values.copy()
        nodenames = list(network.columns)
    elif mode == 'igraph':
        adj = np.array(network.get_adjacency().data)  # upper/lower/both?
        nodenames = list(network.vs['name'])
    # elif mode == 'networkx': # future: add nx support
    #     adj = nx.to_numpy_array(network)
    #     nodenames = list(network.nodes)
    else:
        raise

    if type(timeseries) == np.ndarray:
        ts_array = timeseries.copy()
    elif type(timeseries) == pd.DataFrame:
        ts_array = np.array(timeseries[nodenames])
    else:
        raise

    nodename_idx_dict = {n: x for n, x in enumerate(nodenames)}

    # future: integrate weights here somehow?
    el, ixs = processing.rolling_sim_ixs(ts_array, adj, sim=similarity_metric,
                              **sim_kwargs)

    if mode == 'igraph':
        processed_net = processing.rolling_sims_to_igraph(el, ixs, nodename_idx_dict)
        nodename_dict = {n: processed_net[n].vs['name'] for n in range(len(processed_net))}
    # elif mode == 'networkx': # future: add nx support
        # processed_net = rolling_sims_to_networkx(el, ixs, nodename_idx_dict)
        # pass
    else:
        raise

    cd_output = algorithm(processed_net, **alg_kwargs)

    return cd_output, nodename_dict

def extract_event_reactions(membdf, core, articles):
    """
    Process community detection output to extract the event reactions.

    Keep only communities which have a core article on day at t=0.

    Parameters
    ----------
    membdf : DataFrame
        DataFrame with community memberships.
    core : list
        Core articles for network.
    articles : list
        List of all articles in network.

    Returns
    -------
    dict
        Dictionary of event reactions with core article(s) as keys and filtered
        DataFrames as values.

    """
    try:

        # print('getting event reactions')
        mid = len(membdf.columns)//2
        evrs = {}
        t0comms = membdf.loc[set(core) & set(articles), mid]
        membdfc = membdf.loc[set(core) & set(articles)].T

        cmd = {x: [] for x in set(t0comms.dropna().values)}
        for x in t0comms.dropna().iteritems():
            cmd[x[1]].append(x[0])

        for k, c in cmd.items():
            segt = ((membdfc[c] == k).sum(axis=1) > .5*len(c))
            tf = segt == segt.shift(1)
            diff = np.append(np.where(~tf.values)[0] - mid,
                             len(membdf.columns) - 1)
            beg = mid + diff[diff <= 0].max()
            end = mid + diff[diff > 0].min() - 1
            t = membdf.T.loc[beg:end][membdf.T.loc[beg:end]
                                      == k].T.dropna(how='all')
            gg = {x: set(t[x].dropna().index) for x in t}
            js = pd.Series({k: utilities.jac(set(gg[mid]), set(v))
                            for k, v in gg.items()})
            t = t[js[js > .5].index].dropna(how='all')
            evrs['---'.join(sorted(c))] = t

        # uniquify sets
        for n, (k, v) in enumerate(evrs.items()):
            for m, (l, u) in enumerate(evrs.items()):
                if m > n:
                    for s in set(v.index) & set(u.index):
                        f1 = len(v.loc[s].dropna())/len(v.loc[s])
                        f2 = len(u.loc[s].dropna())/len(u.loc[s])
                        if f1 > f2:
                            evrs[l] = evrs[l].drop(s)
                        elif f2 > f1:
                            evrs[k] = evrs[k].drop(s)
                        elif len(v) < len(u):
                            evrs[l] = evrs[l].drop(s)
                        else:
                            evrs[k] = evrs[k].drop(s)

        return evrs

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]


def community_centralities(glist, igname, membdf, evrs):
    """
    Calculate centrality in community subgraphs.

    Parameters
    ----------
    glist : list
        List of igraph networks for each timestep for an event.
    igname : dict
        List of dicts with igraph node names.
    membdf : DataFrame
        DataFrame with community memberships.
    evrs : dict
        Event reactions dictionary.

    Returns
    -------
    dict
        Dictionary of node centralities for each event reaction.
    """
    try:

        # print('getting comm data')
        for n in range(len(membdf.columns)):
            cattr = membdf[n].copy()
            cattr.index = cattr.index.map({v: k for k, v in igname[n].items()})
            glist[n].vs['community'] = cattr.sort_index().values
            glist[n].vs["label"] = glist[n].vs["name"]

        tcd = {k: processing.tcentrality(glist, v) for k, v in evrs.items()}

        return tcd

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]


def read_ev_data(e, rdarts_rev):
    """
    Read all event network data.

    Parameters
    ----------
    e : str
        Event name.
    rdarts_rev : dict
        Dictionary of redirects to true article names.

    Returns
    -------
    tuple
        core : List of core articles.
        articles : List of all articles.
        glist : temporal network as list.
        igname : list of dicts of vertex names.

    """
    try:
        core = [rdarts_rev.get(x.replace(' ', '_'), x.replace(' ', '_'))
                for x in pd.read_csv(e+'/core.tsv', sep='\t', header=None)[0]]
        articles = pd.read_hdf(e + '/coltitlesNN.h5', key='df')
        glist, igname = read_t_graph(e, core)

        if len(glist[0].vs):
            return core, articles, glist, igname
        else:
            print('Graph empty ' + e)
            return (core, articles)

    except Exception as ex:
        print('Error reading ' + e)
        return (e, ex)


def ev_reactions(core, articles, glist, igname, res, tcd=False):
    """
    Run community detection and extract event reactions from event network.

    Parameters
    ----------
    core : list
        List of core articles for event.
    articles : list
        List of all articles in network.
    glist : list
        List of igraph networks for each timestep for an event.
    igname : dict
        List of dicts with igraph node names.
    res : float
        Community detection resolution.
    tcd : Boolean, optional
        Whether to return node community centralities. The default is False.

    Returns
    -------
    tuple
        membdf : DataFrame with community memberships.
        evrs : Event reactions dictionary.
        cc : Dict of node centralities for each event reaction (optional).
    """
    membdf = temporal_community_detection(glist, res)
    evrs = extract_event_reactions(membdf, core, articles)
    if tcd:
        cc = community_centralities(glist, igname, membdf, evrs)
        return membdf, evrs, cc
    return membdf, evrs

def ev_reactions_flat(core, articles, graph, igname, res, weights=None):
    """
    Run community detection and extract event reactions from event network.

    Parameters
    ----------
    core : list
        List of core articles for event.
    articles : list
        List of all articles in network.
    glist : list
        List of igraph networks for each timestep for an event.
    igname : dict
        List of dicts with igraph node names.
    res : float
        Community detection resolution.
    tcd : Boolean, optional
        Whether to return node community centralities. The default is False.

    Returns
    -------
    tuple
        membdf : DataFrame with community memberships.
        evrs : Event reactions dictionary.
        cc : Dict of node centralities for each event reaction (optional).
    """

    partition = la.find_partition(graph, la.CPMVertexPartition,
                                  resolution_parameter=res,
                                  weights=weights)
    membdfF = pd.Series({y: n for n, x in enumerate(partition)
                         for y in x}).sort_index()
    membdfF.index = graph.vs['name']
    membdfF = membdfF.sort_index()
    mcore = set(membdfF.index)&set(core)
    evrs = {x: list(membdfF[membdfF==x].index) for x in set(membdfF.loc[mcore])}
    cm = {x:'---'.join(sorted(membdfF.loc[mcore][membdfF.loc[mcore]==x].index))
          for x in set(membdfF.loc[mcore])}
    evrs = {cm[k]:v for k, v in evrs.items()}

    return membdfF, evrs

def server_tcd(e, rdarts_rev, res):
    """
    Read data, run community detection and extract event reactions.

    Parameters
    ----------
    e : str
        Event name.
    rdarts_rev : dict
        Dictionary of redirects to true article names.
    res : float
        Community detection resolution.

    Returns
    -------
    tuple
        membdf : DataFrame with community memberships.
        evrs : Event reactions dictionary.
        cc : Dict of node centralities for each event reaction.

    """
    try:
        core = [rdarts_rev.get(x.replace(' ', '_'), x.replace(' ', '_'))
                for x in pd.read_csv(e+'/core.tsv', sep='\t', header=None)[0]]
        articles = pd.read_hdf(e + '/coltitlesNN.h5', key='df')
        glist, igname = read_t_graph(e, core)
    except Exception as ex:
        print('Error reading ' + e)
        return (e, ex)
    membdf = temporal_community_detection(glist, res)
    evrs = extract_event_reactions(membdf, core, articles)
    tcd = community_centralities(glist, igname, membdf, evrs)
    return membdf, evrs, tcd


# %% Flat community detection & structural similarity

def read_f_graph(event, core):
    """
    Read network data to create flat network.

    Parameters
    ----------
    event : str
        Event name (folder location).
    core : list
        List of core articles for event.

    Returns
    -------
    tuple
        G : igraph network (single layer).
        igname : Dictionaries for vertex names.

    """
    try:
        # print('generating graphs', len(articles))

        # print('loading data')
        ixs = np.nan_to_num(np.load(event + '/simtgraphixNN.npz')['arr_0'])
        artixdict = pd.read_hdf(event + '/coltitlesNN.h5', key='df').to_dict()

        el = pd.DataFrame(ixs, columns=['source', 'target'])
        el['source'] = el['source'].map(artixdict)
        el['target'] = el['target'].map(artixdict)
        el['weight'] = 1

        G = igraph.Graph.TupleList([tuple(x) for x in el.values],
                                   directed=False, edge_attrs=['weight'])

        if len(G.vs) == 0:
            # print('empty')
            return [event, 'error', 'empty']

        # print('generating agg network')
        igname = {x.index: x['name'] for x in G.vs}
        igname_rev = {v: k for k, v in igname.items()}
        del ixs, el

        coreids = [igname_rev[x] for x in core if x in igname_rev]
        components = [c for c in G.components()
                      if any([x in c for x in coreids])]
        vertexes = [y for x in components for y in x]
        G = G.subgraph(vertexes)

        igname = {x.index: x['name'] for x in G.vs}

        return G, igname

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        raise
        return [event, 'error', ex]


def read_f_data(e, rdarts_rev):
    """
    Read all event data to generate flat, single layer network.

    Parameters
    ----------
    e : str
        Event name.
    rdarts_rev : dict
        Dictionary of redirects to true article names.

    Returns
    -------
    tuple
        core : List of core articles for event.
        articles : List of all articles in network.
        G : igraph network (single aggregated layer).
        igname : List of dictionaries for vertex names.

    """
    try:
        core = [rdarts_rev.get(x.replace(' ', '_'), x.replace(' ', '_'))
                for x in pd.read_csv(e+'/core.tsv', sep='\t', header=None)[0]]
        articles = pd.read_hdf(e + '/coltitlesNN.h5', key='df')
        G, igname = read_f_graph(e, core)
        return core, articles, G, igname
    except Exception as ex:
        print('Error reading ' + e)
        return (e, ex)


def flat_CD(G, igname, core, resrange, weights=None):
    """
    Run flat community detection across a range of resolutions.

    Parameters
    ----------
    row : (str, Series)
        Community name and edgelist. Unused?
    resrange : iterable
        Range of resolutions to run community detection with.
    core : list
        List of 'core' articles.
    flatadj : DataFrame
        Adjacency matrix of network.

    Returns
    -------
    tuple
        membdfFD : Dictionary of community memberships at each resolution.
        commsFD : Dictionary of dictionaries with communities.
        tcdD : Dictionary of dictionaries with community centralities.

    """
    try:

        # print('generating graphs', len(articles))
        articles = set(igname.values())
        igname_rev = {v: k for k, v in igname.items()}
        membdfFD = {}
        commsFD = {}
        for res in resrange:
            partition = la.find_partition(G, la.CPMVertexPartition,
                                          resolution_parameter=res,
                                          weights=weights)
            membdfF = pd.Series({y: n for n, x in enumerate(partition)
                                 for y in x}).sort_index()
            membdfF.index = G.vs['name']
            membdfF = membdfF.sort_index()

            # print('getting collmems')
            comms = {}
            for c in set(membdfF.loc[set(core) & articles]):
                cn = '---'.join(sorted(membdfF.loc[set(core) & articles]
                                       [membdfF == c].index))
                comms[cn] = list(membdfF[membdfF == c].index)

            membdfFD[res] = membdfF
            commsFD[res] = comms

        tcdD = {res: {k: processing.centrality(G, v, igname_rev)
                      for k, v in V.items()} for res, V in commsFD.items()}

        return membdfFD, commsFD, tcdD

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return ['error', ex]


def evr_matcher(e, tcdD, tcdt, resrange):
    """
    Match communities from flat network across res range to existing community.

    Parameters
    ----------
    n : int
        Index of parallelisation stage.
    x : tuple
        Flat community detection output.
    colm : dict
        Community obtained from temporal (with centrality values).
    resrange : iterable
        Resolution range to explore.
    m : int
        Index of for loop stage..
    allev : DataFrame
        Dataframe of event names.

    Returns
    -------
    tuple
        wjdf : DataFrame with weighted Jaccard similarities across res range.
        jdf : DataFrame with Jaccard similarities across res range.
    """
    try:
        wjdf = pd.DataFrame()
        jdf = pd.DataFrame()
        for k, v in tcdt.items():
            for r in resrange:
                maw = []
                ma = []
                for k2, v2 in tcdD[r].items():
                    maw.append(utilities.wjac(v.to_dict(), v2.to_dict()))
                    ma.append(utilities.jac(set(v.index), set(v2.index)))
                wjdf.loc[k, r] = max(maw)
                jdf.loc[k, r] = max(ma)

        wjdf.index = pd.Series(wjdf.index).apply(lambda x: e + '/' + x)
        jdf.index = pd.Series(jdf.index).apply(lambda x: e + '/' + x)

        return wjdf, jdf
    except Exception as ex:
        return ['error', e, ex]

# %% Robustness procedures


def load_resresults(resdict, resrange, fpath):
    if not os.path.exists(fpath):
        print('filepath doesn\'t exist:', fpath)
        return
    for res in resrange:
        print(res)
        try:
            resdict[res] = [pd.read_hdf(fpath, key='%.5f_%d' % (res, n))
                               for n in range(100)]
        except KeyError:
            print('No ', res)


def cd_test(evdata, resdict, resrange, cdmethod, fpath, *args):
    for res in resrange:
        print(res)
        if res in resdict:
            continue
        try:   
            out2 = Parallel(n_jobs=-1, verbose=10)(delayed(cdmethod
                                                           )(*x, res, *args)
                                                   for x in evdata if len(x) == 4)
    
            for n, x in enumerate(out2):
                x[0].to_hdf(fpath, key='%.5f_%d' % (res, n), mode='a')
            resdict[res] = [x[0] for x in out2]
        except Exception as ex:
            # raise
            print(ex)
            
def comm_similarities(events, resdict, resrange):
    amis = {}
    clusims = {}
    
    for n, e in enumerate(events):
        if n % 10 == 0:
            print("%.2f %%" % (100*n/len(events)))
        midcom = pd.concat([resdict[r][n] for r in resrange], axis=1)
        midcom.columns = resrange
        mc = midcom.columns
    
        amis[e] = [ami(midcom[[mc[c-1], mc[c]]].dropna()[mc[c-1]],
                       midcom[[mc[c-1], mc[c]]].dropna()[mc[c]])
                   for c in range(1, len(midcom.columns))]
    
        cs = []
        for c in range(1, len(midcom.columns)):
            comms = midcom[[mc[c-1], mc[c]]].dropna()
            c1 = Clustering(elm2clu_dict={k: [v] for k, v in
                                          dict(comms[mc[c]]).items()})
            c2 = Clustering(elm2clu_dict={k: [v] for k, v in
                                          dict(comms[mc[c-1]]).items()})
            cs.append(sim.element_sim(c1, c2, alpha=0.9))
    
        clusims[e] = cs
    return amis, clusims

def cd_range_test(evdata, evsample, resrange, r_fpath, c_fpath, cdmethod, *args):

    resresults = {}   
    load_resresults(resresults, resrange, r_fpath)
    cd_test(evdata, resresults, resrange, cdmethod, r_fpath, args)
    if not (os.path.exists(c_fpath %'amis')&os.path.exists(c_fpath %'clusims')):    
        amis, clusims = comm_similarities(evsample, resresults, resrange)    
        amis_df = pd.DataFrame(amis)
        amis_df.index = resrange[1:]
        clusims_df = pd.DataFrame(clusims) 
        clusims_df.index = resrange[1:]

        amis_df.to_hdf(c_fpath %'amis', key='df')
        clusims_df.to_hdf(c_fpath %'clusims', key='df')
    else:
        amis_df = pd.read_hdf(c_fpath %'amis', key='df')
        clusims_df = pd.read_hdf(c_fpath %'clusims', key='df')   
        
    return resresults, amis_df, clusims_df

#%% Evaluate communities

def get_mdev(cc, ts):
    tsc = ts[set(cc)&set(ts.columns)]
    scaler = RobustScaler()
    tsm = pd.Series(scaler.fit_transform(pd.DataFrame(tsc.mean(axis=1).T)
                                         )[:,0], index = range(-30, 31))
    mdev = tsm.loc[-1:1].max()   
    return mdev, tsc


def captured_excess(c, ts, end=None, thresh=3):
    rt = 0
    for k, v in c.items():
        mdev, tsc = get_mdev(c[k], ts)
        if mdev > thresh:
            rt += (tsc.loc[0:end] - tsc.median()).sum().sum()
    return rt


def pad_to_square(a, pad_value=0):
    m = a.reshape((a.shape[0], -1))
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
    padded[0:m.shape[0], 0:m.shape[1]] = m
    return padded


def create_diag(a):
    t = False
    if a.shape[0] > a.shape[1]:
        t=True
        a = a.T
    a = pad_to_square(a)
    max_rows = a.max(axis=1)
    # max_cols = a.max(axis=0)
    colswaps = {}
    for n, m in enumerate(max_rows):
        if m>0:
            c1, c2 = np.where(a[n:n+1,]==m)
            a[:,[n,c2[0]]] = a[:,[c2[0],n]]
            colswaps[n] = c2[0]
        else:
            colswaps[n] = n

    order = np.argsort(np.diag(a))[::-1]
    if t:
        a = a.T   
   
    return a, colswaps, order


def rearrange_comms(comms1, comms2):
    sims = np.array([[utilities.jac(v1, v2) for v1 in comms1.values()]
                     for v2 in comms2.values()])
    
    mat, cswap, order = create_diag(sims)
    if sims.shape[0] <= sims.shape[1]:      
        comms_l = list(comms1.items())
        for k in range(len(comms1)):
            c1 = comms_l[k]
            c2 = comms_l[cswap[k]]
            comms_l[k] = c2
            comms_l[cswap[k]] = c1
        comms1 = dict(comms_l)
    else:
        commst_l = list(comms2.items())
        for k in range(len(comms2)):
            c1 = commst_l[k]
            c2 = commst_l[cswap[k]]
            commst_l[k] = c2
            commst_l[cswap[k]] = c1
        comms2 = dict(commst_l)
        
    return comms1, comms2


# %% Higher level community detection


def H_sim(ct):
    """
    Get similarity between clusterings at different resolutions.

    Parameters
    ----------
    ct : DataFrame
        DataFrame with two columns, community assignments in each column.

    Returns
    -------
    ami_out : float
        AMI between clusterings.
    clusim_out : float
        CluSim similarity between clusterings.

    """
    c1 = ct[:, 0]
    c2 = ct[:, 1]

    ami_out = sklearn.metrics.adjusted_mutual_info_score(c1, c2)

    c1 = Clustering(elm2clu_dict={n: [v] for n, v in enumerate(c1)})
    c2 = Clustering(elm2clu_dict={n: [v] for n, v in enumerate(c2)})
    clusim_out = sim.element_sim(c1, c2, alpha=0.9)

    return ami_out, clusim_out

# %% Unused


# def cd1(row, res, quantile, megamap={}, fp='/simtgraph5N.npz'):
#     """
#     Prepare temporal event network for community detection.

#     Parameters
#     ----------
#     row : (str, Series)
#         Community name and edgelist.
#     res : float
#         Resolution.
#     quantile : float
#         Quantile to threshold edges on.
#     megamap : dict
#         Map of article redirects.
#     fp : str, optional
#         Filepath to correlation ndarray. The default is '/simtgraph5N.npz'.

#     Returns
#     -------
#     list
#         res : Resolution.
#         glist : List of networks (layers in temporal network).
#         igname : List of dicts with igraph node names.
#         coreu : Core articles (with underscores).
#         articles : Articles in network.

#     """
#     try:
#         core = [megamap.get(x, x) for x in pd.read_csv(
#             row[0]+'/core.tsv', sep='\t', header=None)[0]]
#         coreu = [x.replace(' ', '_') for x in core]
#         flatadj = pd.read_hdf(
#             row[0].replace('/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
#             + '/adjNN.h5', key='df')

#         articles = list(flatadj.columns)

#         # print('generating graphs', len(articles))

#         print('loading data', fp)
#         pcs = np.nan_to_num(np.load(row[0].replace(
#             '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
#             + fp)['arr_0'])
#         q = pd.read_hdf(row[0].replace(
#             '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
#             + '/quantilesNN.npz')
#         pcs = np.where(pcs >= q.loc[quantile], pcs, 0)

#         print('generating graphs', len(articles))

#         glist = []

#         tlen = 55
#         for n in range(tlen):
#             sel = pd.DataFrame(pcs[n, :, :], index=articles,
#                                columns=articles).stack().reset_index()
#             sel = sel[sel[0] != 0]
#             sel.columns = ['source', 'target', 'weight']
#             sel = sel[sel['source'] != 'Main Page']
#             sel = sel[sel['target'] != 'Main Page']
#             tuples = [tuple(x) for x in sel.values]
#             glist.append(igraph.Graph.TupleList(tuples, directed=False,
#                                                 edge_attrs=['weight']).simplify(combine_edges='first'))
#             glist[n].vs["slice"] = n

#         if sum([len(x.vs) for x in glist]) == 0:
#             # print('empty')
#             return [fp, 'error', 'empty']

#         # print('generating agg network')
#         igname = [{x.index: x['name'] for x in glist[y].vs}
#                   for y in range(tlen)]

#         del pcs
#         del sel

#         return [res, glist, igname, coreu, articles]

#     except KeyboardInterrupt:
#         raise
#     except Exception as ex:
#         # raise
#         return [fp, 'error', ex]


# def cd2(res, glist, igname, coreu, articles):
#     """
#     Run Leiden algorithm on temporal network, returning community memberships.

#     Parameters
#     ----------
#     res : float
#         Resolution.
#     glist : list
#         List of networks (layers in temporal network).
#     igname : list
#         List of dicts with igraph node names.
#     coreu : list
#         Core articles (with underscores).
#     articles : list
#         Articles in network.


#     Returns
#     -------
#     tuple
#         membdf : DataFrame with community memberships.
#         collmems : Dictionary of communities.
#         tcd : Centralities of articles within each community.

#     """
#     try:
#         print('running community detection')

#         membership, improvement = la.find_partition_temporal(
#             glist, la.CPMVertexPartition, vertex_id_attr='name',
#             interslice_weight=1, resolution_parameter=res)
#         membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
#                             for n, x in enumerate(membership)],
#                            axis=1, sort=True)

#         print('getting comm data')
#         for n in range(55):
#             cattr = membdf[n].copy()
#             cattr.index = cattr.index.map({v: k for k, v in igname[n].items()})
#             glist[n].vs['community'] = cattr.sort_index().values
#             glist[n].vs["label"] = glist[n].vs["name"]

#         print('getting collmems')
#         collmems = {}
#         t0comms = membdf.loc[set(coreu) & set(articles), 27]
#         membdfc = membdf.loc[set(coreu) & set(articles)].T

#         cmd = {x: [] for x in set(t0comms.dropna().values)}
#         for x in t0comms.dropna().iteritems():
#             cmd[x[1]].append(x[0])

#         for k, c in cmd.items():
#             segt = ((membdfc[c] == k).sum(axis=1) > .5*len(c))
#             tf = segt == segt.shift(1)
#             diff = np.where(tf.values is False)[0]-27
#             beg = 27+diff[diff <= 0].max()
#             try:
#                 end = 26+diff[diff > 0].min()
#             except ValueError:
#                 end = 54
#             tstep = membdfc.loc[beg:end, c].index
#             t = membdf.T.loc[tstep][membdf.T.loc[tstep]
#                                     == k].T.dropna(how='all')
#             gg = {x: set(t[x].dropna().index) for x in t}
#             js = pd.Series({k: utilities.jac(set(gg[27]), set(v))
#                             for k, v in gg.items()})
#             t = t[js[js > .5].index].dropna(how='all')
#             collmems['---'.join(sorted(c))] = t

#         # uniquify sets
#         for k, v in collmems.items():
#             for l, u in collmems.items():
#                 if k != l:
#                     for s in set(v.index) & set(u.index):
#                         f1 = len(v.loc[s].dropna())/len(v.loc[s])
#                         f2 = len(u.loc[s].dropna())/len(u.loc[s])
#                         if f1 > f2:
#                             collmems[l] = collmems[l].drop(s)
#                         elif f2 > f1:
#                             collmems[k] = collmems[k].drop(s)
#                         elif len(v) < len(u):
#                             collmems[l] = collmems[l].drop(s)
#                         elif len(v) >= len(u):
#                             collmems[k] = collmems[k].drop(s)

#         tcd = {k: processing.tcentrality(glist, v)
#                for k, v in collmems.items()}

#         return membdf, collmems, tcd

#     except KeyboardInterrupt:
#         raise
#     except Exception as ex:
#         # raise
#         return ['error', ex]

# def ev_reactions_tcd(core, articles, glist, igname, res):
#     membdf = temporal_community_detection(glist, res)
#     evrs = extract_event_reactions(membdf, igname, core, articles)
#     tcd = community_centralities(glist, igname, membdf, evrs)
#     return membdf, evrs, tcd
