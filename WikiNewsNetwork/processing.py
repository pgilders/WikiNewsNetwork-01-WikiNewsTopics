#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:21:20 2022

@author: Patrick
"""

from calendar import monthrange
import datetime
import pandas as pd
import numpy as np
from WikiNewsNetwork.utilities import months_range


# %% Network functions

def tcentrality(gl, comm):
    """
    Centrality of nodes within specified community in temporal network.

    Parameters
    ----------
    gl : list of graphs
        Layers of temporal network.
    comm : DataFrame
        Community assignments at each layer?

    Returns
    -------
    Series
        Mean centralities.

    """
    cdf = pd.DataFrame(index=comm.index, columns=comm.columns)
    for x in comm.columns:
        sg = gl[x].subgraph(comm[x].dropna().index)
        cdf[x] = pd.Series(sg.pagerank(weights='weight'), index=sg.vs['name'])
    return cdf.mean(axis=1)


def centrality(g, comm, nodenamesr):
    """
    Centrality of nodes within specified community in network.

    Parameters
    ----------
    g : network
        Network of interest.
    comm : iterable
        ?.
    nodenamesr : dict
        Maps redirect node names.

    Returns
    -------
    cdf : Series
        Centralites.

    """
    cdf = pd.Series(index=comm)

    sg = g.subgraph(pd.Series(comm).map(nodenamesr))
    cdf = pd.Series(sg.pagerank(), index=sg.vs['name'])

    return cdf


# %% Creating Event Article Networks


def get_neighbours_quick(articles, csd, corerds):
    """
    Get all 1 hop neighbours of specified articles from edgelist.

    Parameters
    ----------
    articles : iterable
        Articles (egos).
    csd : DataFrame
        Edgelist.
    corerds : dict
        Dict of all redirects for articles.

    Returns
    -------
    set
        Set of all neighbours of 'articles'.

    """
    try:

        # print('reading')
        corerd = {y for x in articles
                  for y in corerds.get(x.replace(' ', '_'),
                                       [x.replace(' ', '_')])}

        # print('csdf')
        ndf = csd[(csd['curr'].isin(corerd)) | (csd['prev'].isin(corerd))]
        # print('ndf')
        ndfarticles = set(ndf['curr']) | set(ndf['prev'])

        return ndfarticles
    except Exception as ex:
        print(ex)
        # raise
        return [False, ex]


def getel(e, csd, redir_arts_map, rdarts_rev, eventsdf):
    """
    Get edgelist of event, correcting for redirects.

    Get all neighbours of core articles, and all edges between all neighbours.
    Calculate edge weights weighted by days in month, keep all above 100.

    Parameters
    ----------
    e : str
        Event name.
    csd : DataFrame
        Edgelist with monthly totals.
    redir_arts_map : dict
        Dictionary of article title keys to list of all redirects as value.
    rdarts_rev : dict
        Dictionary of redirect titles as keys to true title as value.
    eventsdf : DataFrame
        DataFrame of events.

    Returns
    -------
    str, DataFrame
        Event name and associated edgelist.

    """
    try:

        date = pd.to_datetime(e[:8])
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)

        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
              for x in months}

        core = eventsdf.loc[e, 'Articles']
        ndfarticles = get_neighbours_quick(core, csd, redir_arts_map)

        el = csd[(csd['curr'].isin(ndfarticles)) &
                 (csd['prev'].isin(ndfarticles))].copy()

        # map to rdtitle and group sum
        el['prev'] = el['prev'].apply(lambda x: rdarts_rev.get(x, x))
        el['curr'] = el['curr'].apply(lambda x: rdarts_rev.get(x, x))
        el = el.groupby(['prev', 'curr']).sum().reset_index()

        # Weighted sum of edgeweights
        el[el.columns[2]] = el[el.columns[2]] * \
            ((start.replace(day=mr[el.columns[2]])-start).days+1)
        el[el.columns[-2]] = el[el.columns[-2]] * \
            ((stop-stop.replace(day=1)).days+1)
        el[el.columns[3:-2]] = el[el.columns[3:-2]
                                  ].apply(lambda x: mr[x.name]*x)
        el['n'] = el[el.columns[2:-1]].sum(axis=1)

        # Filter to >100 pv
        el = el[el['n'] > 100]

        return e, el
    except Exception as ex:
        print(e, ex)
        # raise
        return e, False, ex


def getevls(i, edf, tsl, rdarts_rev, epath, pvlimit=1000,
            elp='/all_el100NNN.h5'):
    """
    Take event name and return adjacency matrix and page views for articles.

    Reads previously generated edgelist, filters for articles with page views >
    pvlimit, and returns corresponding network and time series, together with
    summary stats for DataFrame.

    Parameters
    ----------
    i : str
        Event name.
    edf : DataFrame
        DataFrame of events with # of associated articles and list of them.
    tsl : dict
        Dictionary of month keys and DataFrames with page views as values.
    rdarts_rev : dict
        Dictionary of redirect titles as keys to true title as value.
    epath : str
        Path to events directory.
    pvlimit : int, optional
        Threshold for minimum page views in period. The default is 1000.
    elp : str, optional
        Filename for edgelist. The default is '/all_el100NNN.h5'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    try:
        if i in edf.index:
            return ({i: 'got'},)

        date = datetime.datetime.strptime(i[:8], '%Y%m%d')
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)

        # print('mc')
        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        el = pd.read_hdf(epath + i + elp)

        articles = set(el['prev']) | set(el['curr'])

        # print('filesread')
        ts = pd.concat([y[sorted(articles & set(y.columns))] for y in
                        [tsl[z] for z in months]],
                       sort=True).fillna(0).loc[start:stop]

        ts = ts[ts.sum()[ts.sum() > pvlimit].index]

        el = el[(el['prev'].isin(ts.columns)) & (el['curr'].isin(ts.columns))]

        articles = sorted(set(el['prev']) | set(el['curr']))
        adj = (~el.pivot(index='prev', columns='curr', values='n').isna()
               ).reindex(columns=articles, index=articles, fill_value=False)
        adj = (adj | adj.T).astype(int)

        return {i: {'len': len(articles),
                    'articles': articles}}, adj, ts[articles]

    except KeyboardInterrupt:
        raise

    except Exception as ex:
        #        raise
        print('ex')
        return ({i: ex},)

# %% Edge correlations


def pearsonr_ixs(x, ixs):
    """
    Calculate Pearson correlation between time series in x for indexes in ixs.

    Only the correlations between indexes in ixs is calculated and returned.
    Order of correlations returned matches order of indexes supplied.

    Parameters
    ----------
    x : NumPy Array
        Time series of page views.
    ixs : NumPy Array
        Array of indexes where edge is present and correlation is calculated.

    Returns
    -------
    NumPy Array
        Pearson correlation between time series at index combinations (1D).

    """
    xm = (x - x.mean(axis=0))
    xmn = xm/np.sqrt((xm*xm).sum(axis=0))

    return (xmn[:, ixs][:, :, 0] * xmn[:, ixs][:, :, 1]).sum(axis=0)


def rolling_pearson_ixs(timeseries, adj):
    """
    Calculate rolling Pearson correlation between time series for indexes in ixs.

    Only the correlations between indexes in ixs is calculated and returned.
    Order of correlations returned at each time step matches order of indexes
    supplied.

    Parameters
    ----------
    timeseries : NumPy Array
        Time series of page views for articles in network.
    adj : NumPy Array
        Adjacency matrix of network.

    Returns
    -------
    NumPy Array
        Rolling (7 day) Pearson correlation between time series at index
        combinations at each time step (2D). An edgelist for each time step.
    ixs : NumPy Array
        Array of indexes where edge is present and correlation is calculated.

    """
    ixs = np.argwhere(adj.values)
    return np.array([pearsonr_ixs(timeseries[x:x+7], ixs)
                     for x in range(0, len(timeseries)-6)]), ixs


# def getadj(el):
#     elt = [(i[1]['prev'], i[1]['curr'])
#            for i in list(el[['prev', 'curr']].iterrows())]
#     + [(i[1]['curr'], i[1]['prev'])
#         for i in list(el[['prev', 'curr']].iterrows())]
#     st = pd.DataFrame(list(elt)).set_index([0, 1])
#     st[0] = 1
#     st = st.loc[~st.index.duplicated()]
#     adj = st.unstack()
#     adj.columns = adj.columns.droplevel()
#     adj = adj.sort_index()[adj.sort_index().index]

#     return adj


# def procts(el, tsl, articles, start, stop, months):
#     cc = []
#     for m in months:
#         cc.append(tsl[m][sorted(articles & set(tsl[m].columns))])
#     ts = pd.concat(cc, sort=True).fillna(0).loc[start:stop]

#     print('gotf')
#     ts = ts[sorted(articles)]

#     print('elcut')
#     elt = [(i[1]['prev'], i[1]['curr'])
#            for i in list(el[['prev', 'curr']].iterrows())] \
#         + [(i[1]['curr'], i[1]['prev'])
#             for i in list(el[['prev', 'curr']].iterrows())]
#     st = pd.DataFrame(list(elt)).set_index([0, 1])
#     st[0] = 1
#     st = st.loc[~st.index.duplicated()]
#     adj = st.unstack()
#     adj.columns = adj.columns.droplevel()
#     adj = adj.reindex(index=ts.columns, columns=ts.columns).fillna(0)
#     adj = adj.sort_index()[adj.sort_index().index]

#     return adj, ts
