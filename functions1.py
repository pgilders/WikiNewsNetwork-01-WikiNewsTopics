#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:36:28 2022

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
import requests
import re
from bs4 import BeautifulSoup as bs


# =============================================================================
# Utilities
# =============================================================================


def months_range(start, stop):
    """
    Take start and stop datetimes and return list of str months in that range.

    Parameters
    ----------
    start : datetime
        Starting date.
    stop : datetime
        End date.

    Returns
    -------
    months : list of strs
        list of yearmonths in YYYYMM format.

    """
    xr = start
    stop2 = stop
    stop2 = stop2.replace(day=monthrange(stop2.year, stop2.month)[1])
    months = []
    while xr <= stop2:
        months.append('%d%02d' % (xr.year, xr.month))
        xr += relativedelta(months=1)
    return months


def getmr(ev):
    date = pd.to_datetime(ev[:8])
    start = date-datetime.timedelta(days=30)
    stop = date+datetime.timedelta(days=30)

    months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
    return {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1] for x in months}


def jac(x, y):
    """
    Jaccard index of two sets.

    Parameters
    ----------
    x : set
        First set.
    y : set
        Second set.

    Returns
    -------
    float
        In range 0-1.

    """
    return len(x & y)/len(x | y)


def jac(x, y):
    """
    Jaccard index of two iterables (no need to be sets).

    Parameters
    ----------
    x : iterable
        Items in first group.
    y : iterable
        Items in second group.

    Returns
    -------
    float
        In range 0-1.

    """
    return len(set(x) & set(y))/len(set(x) | set(y))


def jac(x, y):
    """
    Jaccard index of two iterables (no need to be sets).

    Parameters
    ----------
    x : iterable
        Items in first group.
    y : iterable
        Items in second group.

    Returns
    -------
    float
        In range 0-1.

    """
    if type(x) != set:
        x = set(x)
    if type(y) != set:
        y = set(y)
    return len(x & y)/len(x | y)


def fdiv(a, b):
    if a < b:
        return a/b
    else:
        return b/a


def wjac(x, y):
    """
    Weighted Jaccard index of prealigned sets.

    Parameters
    ----------
    x : dict
        weights of items in set x (0s for items not in original set x).
    y : dict
        weights of items in set y (0s for items not in original set y).

    Returns
    -------
    float
        In range 0-1.

    """
    l = [fdiv(x.get(i, 0), y.get(i, 0))
         for i in set(x.keys()) | set(y.keys())]

    return sum(l)/len(l)


def colourer(x, simdf):
    """
    Colours Latex cell based on similarity quantile.

    Parameters
    ----------
    x : str
        Name of topic.
    simdf : DataFrame
        Dataframe with similarities.

    Returns
    -------
    str
        Latex code with colour.

    """
    if x == '?':
        return None
    ou = simdf[simdf['Name'] == x][0.6065306597126334]
    LQ, MM, UQ = simdf[0.6065306597126334].quantile([0.25, 0.5, 0.75])
    if len(ou) != 1:
        print(x)
        raise
    if ou.iloc[0] > UQ:
        return '\\cellcolor{green}' + x.replace('&', '\\&')
    elif ou.iloc[0] > MM:
        return '\\cellcolor{yellow}' + x.replace('&', '\\&')
    elif ou.iloc[0] > LQ:
        return '\\cellcolor{orange}' + x.replace('&', '\\&')
    else:
        return '\\cellcolor{red}' + x.replace('&', '\\&')

# =============================================================================
# Scrape
# =============================================================================


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def query(request):
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            'https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            print('erroorrrr')
            print(result['error'])
            # raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']


def parse(request):
    request['action'] = 'parse'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            'https://en.wikipedia.org/w/api.php', params=req).json()
        # print(result)
        if 'error' in result:
            print('erroorrrr')
            print(result['error'])
            # raise Error(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'parse' in result:
            yield result['parse']
        if 'continue' not in result:
            break
        lastContinue = result['continue']


def wiki_news_articles(months):

    stories_df = pd.DataFrame(columns=['Date', 'Category', 'Text', 'Articles',
                                       'Ext links', 'HTML'])  # create df

    for m in months:
        print(m)
        params = {'page': 'Portal:Current_events/%s' % m, 'prop': 'text'}
        # Get data from current events page for that month
        scr = list(parse(params))

        doc = bs(scr[0]['text']['*'], 'html.parser')  # BeautifulSoup the html

        # tag for each day ######## NOTE THAT THIS CHANGES BASED ON MONTH - NEED TO GET RIGOROUS ON THIS - Previously "table", {"class" : "vevent"}
        days = doc.findAll("div", {"class": "vevent"})
        for d in days:
            day = d['id']  # date
            # get categories in list
            # cats = d.findAll("div", {'role': 'heading',
            #                          "style": "margin-top:0.3em; font-size:inherit; font-weight:bold;"})
            cats = d.findAll("div", {'role': 'heading',
                                     "class": "current-events-content-heading"})
            for c in cats:
                cat_name = c.text
                # get all stories in cat (note some repeats)
                stories = c.next_sibling.next_sibling.findAll("li")
                for s in stories:
                    txt = ''
                    new_txt = s.text.strip()  # get text description
                    if new_txt in txt:  # if matches previous, skip
                        continue
                    txt = new_txt
                    HTML = s
                    links = s.findAll("a")  # get links in description
                    articles = []
                    ext = []
                    for l in links:
                        try:
                            # add wiki links to list//
                            articles.append(l['title'])
                        except KeyError:
                            # add ext links to another list
                            ext.append(l['href'])

                    # add event data to df
                    stories_df = stories_df.append({'Date': day,
                                                    'Category': cat_name,
                                                    'Text': txt,
                                                    'Articles': articles,
                                                    'Ext links': ext,
                                                    'HTML': str(HTML)},
                                                   ignore_index=True)

    return stories_df


# =============================================================================
# Redirects
# =============================================================================


def fix_redirects(articles, existingmap={}):
    tar_chunks = list(chunks(list(set(articles)-set(existingmap.keys())), 50))
    mapping = {}
    for n, i in enumerate(tar_chunks):
        if n % 100 == 0:
            print('Fixing redirects', round(100*n/len(tar_chunks), 2), '%')
        istr = '|'.join(i)
        params = {'titles': istr, 'redirects': ''}
        dat = list(query(params))
        for j in dat:
            try:
                for k in j['redirects']:
                    mapping[k['from']] = k['to']

            except KeyError:
                # print('No redirects in this chunk')
                pass

    # Apply mapping to self to fix second order redirects
    print('Consolidating mapping')
    titlemap = mapping.copy()
    for k, v in mapping.items():
        for i in mapping.keys():
            if i == v:
                titlemap[k] = mapping[i]
                break

    return {**existingmap, **titlemap}


def get_redirects(articles, existingrds={}):
    tar_chunks = list(chunks(list(set(articles)-set(existingrds.keys())), 50))
    rd_map = {x.replace(' ', '_'): [x.replace(' ', '_')] for x in articles}
    for n, i in enumerate(tar_chunks):
        if n % 100 == 0:
            print('Getting redirects %.2f%%' % (100*n/len(tar_chunks)))
        istr = '|'.join(i)
        params = {'titles': istr, 'prop': 'redirects',
                  'rdlimit': 'max', 'rdnamespace': '0'}
        for j in query(params):
            for k in j['pages'].values():
                try:
                    rd_map[k['title'].replace(' ', '_')].extend(
                        [x['title'].replace(' ', '_') for x in k['redirects']])
                except KeyError:
                    pass
    return {**existingrds, **rd_map}

# =============================================================================
# Pageviews
# =============================================================================


def text_to_tseries(text, year, month):
    lastday = calendar.monthrange(year, month)[1]
    ts = text.split(',')[:-1]
    pvd = {ord(x[0])-64: x[1:] for x in ts}
    pvds = {datetime.datetime.strptime('-'.join(['%d%02d' % (year, month),
                                                 str(k), str(ord(x[0])-65)]),
                                       '%Y%m-%d-%H'):
            int(x[1].replace('?', '0')) if x[1] else 0 for k, v in pvd.items()
            for x in re.findall('([A-Z])([\?\d]*)', v)}

    return pd.Series(pvds, index=pd.date_range('%d%02d01' % (year, month),
                                               '%d%02d%d2300' % (year, month,
                                                                 lastday),
                                               freq='H')).fillna(0)

# =============================================================================
# Network functions
# =============================================================================


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


# =============================================================================
# Processing
# =============================================================================


def get_neighbours_quick_old(e, csd, corerds, eventsdf, mrk=None):
    try:
        if not mrk:
            date = pd.to_datetime(e[:8])
            start = date-datetime.timedelta(days=30)
            stop = date+datetime.timedelta(days=30)

            months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
            mr = {'n_%s-%s' % (x[:4], x[4:]):
                  monthrange(int(x[:4]), int(x[4:]))[1] for x in months}
            mrk = frozenset(mr.keys())

        # print('reading')
        core = eventsdf.loc[e, 'Articles']
        # coremm = {corerds.get(x, x).replace(' ', '_') for x in core}
        corerd = {y for x in core for y in corerds.get(x.replace(' ', '_'),
                                                       [x.replace(' ', '_')])}
        # return 0

        # print('csdf')
        ndf = csd[mrk][(csd[mrk]['curr'].isin(corerd)) |
                       (csd[mrk]['prev'].isin(corerd))]
        # print('ndf')
        ndfarticles = set(ndf['curr']) | set(ndf['prev'])

        # # print('el')
        # el = ndf[(ndf['curr'].isin(ndfarticles)) &
        #          (ndf['prev'].isin(ndfarticles))].copy()

        # el[el.columns[2]] = el[el.columns[2]] * \
        #     ((start.replace(day=mr[el.columns[2]])-start).days+1)

        # el[el.columns[-2]] = el[el.columns[-2]] * \
        #     ((stop-stop.replace(day=1)).days+1)

        # el[el.columns[3:-2]] = el[el.columns[3:-2]
        #                           ].apply(lambda x: mr[x.name]*x)

        # el['n'] = el[el.columns[2:-1]].sum(axis=1)
        # el = el[el['n'] > 100]

        # return [e, set(el['prev']) | set(el['curr'])]
        return [e, ndfarticles]
    except Exception as ex:
        print(e, ex)
        # raise
        return [e, False, ex]


def get_neighbours_quick(e, csd, corerds, eventsdf):
    try:

        # print('reading')
        core = eventsdf.loc[e, 'Articles']
        corerd = {y for x in core for y in corerds.get(x.replace(' ', '_'),
                                                       [x.replace(' ', '_')])}

        # print('csdf')
        ndf = csd[(csd['curr'].isin(corerd)) | (csd['prev'].isin(corerd))]
        # print('ndf')
        ndfarticles = set(ndf['curr']) | set(ndf['prev'])

        return [e, ndfarticles]
    except Exception as ex:
        print(e, ex)
        # raise
        return [e, False, ex]


def get_neighbours_quick_2(articles, csd, corerds, eventsdf):
    try:

        # print('reading')
        corerd = {y for x in articles for y in corerds.get(x.replace(' ', '_'),
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


def getel(e, csd, megamap, redir_arts_map, rdarts_rev, eventsdf):
    try:

        #         date = pd.to_datetime(e[:8])
        #         start = date-datetime.timedelta(days=30)
        #         stop = date+datetime.timedelta(days=30)

        #         months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        #         mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
        #               for x in months}
        #         mrk = frozenset(mr.keys())

        # #        print('reading')
        #         core = eventsdf.loc[e, 'Articles']

        #         coremm = {megamap.get(x, x).replace(' ', '_') for x in core}
        #         corerd = {y for x in coremm for y in redir_arts_map.get(x, [x])}
        #         # return 0

        # #        print('csdf')
        #         ndf = csd[mrk][(csd[mrk]['curr'].isin(corerd)) |
        #                        (csd[mrk]['prev'].isin(corerd))].copy()
        #        print('ndf')

        date = pd.to_datetime(e[:8])
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)

        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
              for x in months}
        mrk = frozenset(mr.keys())

        ndfarticles = get_neighbours_quick(e, csd, redir_arts_map, eventsdf,
                                           mrk)[1]

#        print('el')
        el = csd[mrk][(csd[mrk]['curr'].isin(ndfarticles)) &
                      (csd[mrk]['prev'].isin(ndfarticles))].copy()

        # map to rdtitle and group sum
        el['prev'] = el['prev'].apply(lambda x: rdarts_rev.get(x, x)).dropna()
        el['curr'] = el['curr'].apply(lambda x: rdarts_rev.get(x, x)).dropna()
        el = el.groupby(['prev', 'curr']).sum().reset_index()

        el[el.columns[2]] = el[el.columns[2]] * \
            ((start.replace(day=mr[el.columns[2]])-start).days+1)
        el[el.columns[-2]] = el[el.columns[-2]] * \
            ((stop-stop.replace(day=1)).days+1)
        el[el.columns[3:-2]] = el[el.columns[3:-2]
                                  ].apply(lambda x: mr[x.name]*x)
        el['n'] = el[el.columns[2:-1]].sum(axis=1)
        el = el[el['n'] > 100]

        return [e, el]
    except Exception as ex:
        print(e, ex)
        # raise
        return [e, False, ex]


def getevls(i, edf, tsl, megamap, day_offset=0, elp='/all_el100.h5'):

    try:
        if i in edf.index:
            return {i: 'got'}

        date = datetime.datetime.strptime(
            i.split('/evtest/')[-1][:8], '%Y%m%d')-datetime.timedelta(days=day_offset)
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)

        # print('mc')
        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
              for x in months}

        core = [megamap.get(x, x) for x in eventsdf.loc[e, 'Articles']]

        el = pd.read_hdf(i+elp)
        el['prev'] = el['prev'].str.replace('_', ' ').map(megamap)
        el['curr'] = el['curr'].str.replace('_', ' ').map(megamap)
        el = el[el['prev'] != 'Main Page']
        el = el[el['curr'] != 'Main Page']
        articles = set(el['prev']) | set(el['curr'])

        # print('elops')

        el[el.columns[2]] = el[el.columns[2]] * \
            ((start.replace(day=mr[el.columns[2]])-start).days+1)  # hmmmmmmmmm
        el[el.columns[-2]] = el[el.columns[-2]] * \
            ((stop-stop.replace(day=1)).days+1)
        el[el.columns[3:-2]] = el[el.columns[3:-2]].apply(lambda x:
                                                          mr[x.name]*x)
        el['n'] = el[el.columns[2:-1]].sum(axis=1)

        el = el[el['n'] > 100]

        articles = set(el['prev'][el['curr'].isin(core)]) | set(
            el['curr'][el['prev'].isin(core)]) | core

        el = el[el['prev'].isin(articles) & el['curr'].isin(articles)]

        # print('filesread')
        ts = pd.concat([y[sorted({x.replace(' ', '_') for x in articles}
                                 & set(y.columns))]
                        for y in [tsl[z] for z in months]],
                       sort=True).fillna(0).loc[start:stop]

        ts = ts[ts.sum()[ts.sum() > 1000].index]

        el = el[el['prev'].isin([x.replace('_', ' ') for x in ts.columns])]
        el = el[el['curr'].isin([x.replace('_', ' ') for x in ts.columns])]

        articles = set(el['prev']) | set(el['curr'])
        return {i: {'len': len(articles), 'articles': articles}}

    except KeyboardInterrupt:
        raise

    except Exception as ex:
        #        raise
        return {i: ex}


def csgraph(i, articles, megamap, start, stop, elf='/all_el100.h5',
            day_offset=0):

    months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
    mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
          for x in months}

    el = pd.read_hdf(i+elf)
    el['prev'] = el['prev'].str.replace('_', ' ').map(megamap)
    el['curr'] = el['curr'].str.replace('_', ' ').map(megamap)
    el = el[el['prev'].isin(articles)]
    el = el[el['curr'].isin(articles)]

    el[el.columns[2]] = el[el.columns[2]] * \
        ((start.replace(day=mr[el.columns[2]])-start).days+1)
    el[el.columns[-2]] = el[el.columns[-2]] * \
        ((stop-stop.replace(day=1)).days+1)
    el[el.columns[3:-2]] = el[el.columns[3:-2]
                              ].apply(lambda x: mr[x.name]*x)
    el['n'] = el[el.columns[3:]].sum(axis=1)
    el = el[el['n'] > 100]

    return el


def scoreretlist(n, tsx):
    x = tsx.iloc[n:n+7].values
    score = pd.DataFrame(x).corr().values
    return score


def procts(el, tsl, articles, start, stop, months):
    cc = []
    for m in months:
        cc.append(tsl[m][sorted({x.replace(' ', '_') for x in articles}
                                & set(tsl[m].columns))])
    ts = pd.concat(cc, sort=True).fillna(0).loc[start:stop]

    print('gotf')
    ts = ts[sorted([x.replace(' ', '_') for x in articles])]

    print('elcut')
    elt = [(i[1]['prev'], i[1]['curr'])
           for i in list(el[['prev', 'curr']].iterrows())] \
        + [(i[1]['curr'], i[1]['prev'])
            for i in list(el[['prev', 'curr']].iterrows())]
    st = pd.DataFrame(list(elt)).set_index([0, 1])
    st[0] = 1
    st = st.loc[~st.index.duplicated()]
    adj = st.unstack()
    adj.columns = adj.columns.droplevel()
    adj = adj.reindex(index=[x.replace('_', ' ') for x in ts.columns],
                      columns=[x.replace('_', ' ') for x in ts.columns]
                      ).fillna(0)
    adj = adj.sort_index()[adj.sort_index().index]
    adj.index = [x.replace(' ', '_') for x in adj.index]
    adj.columns = [x.replace(' ', '_') for x in adj.columns]
    print('gotadj')
    scaler = RobustScaler()
    tsx = pd.DataFrame(scaler.fit_transform(ts), index=ts.index,
                       columns=ts.columns)
    return adj, tsx

# =============================================================================
# Community Detection
# =============================================================================


def cd1(row, res, quantile, megamap={}, fp='/simtgraph5N.npz'):
    """
    Prepare temporal event network for community detection.

    Parameters
    ----------
    row : (str, Series)
        Community name and edgelist.
    res : float
        Resolution.
    quantile : float
        Quantile to threshold edges on.
    megamap : dict
        Map of article redirects.
    fp : str, optional
        Filepath to correlation ndarray. The default is '/simtgraph5N.npz'.

    Returns
    -------
    list
        res : Resolution.
        glist : List of networks (layers in temporal network).
        igname : List of dicts with igraph node names.
        coreu : Core articles (with underscores).
        articles : Articles in network.

    """
    try:
        core = [megamap.get(x, x) for x in pd.read_csv(
            row[0]+'/core.tsv', sep='\t', header=None)[0]]
        coreu = [x.replace(' ', '_') for x in core]
        flatadj = pd.read_hdf(
            row[0].replace('/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
            + '/adjNN.h5', key='df')

        articles = list(flatadj.columns)

        # print('generating graphs', len(articles))

        print('loading data', fp)
        pcs = np.nan_to_num(np.load(row[0].replace(
            '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
            + fp)['arr_0'])
        q = pd.read_hdf(row[0].replace(
            '/Volumes/Samsung_T3/evtest', 'active_data/tempevs')
            + '/quantilesNN.npz')
        pcs = np.where(pcs >= q.loc[quantile], pcs, 0)

        print('generating graphs', len(articles))

        glist = []

        tlen = 55
        for n in range(tlen):
            sel = pd.DataFrame(pcs[n, :, :], index=articles,
                               columns=articles).stack().reset_index()
            sel = sel[sel[0] != 0]
            sel.columns = ['source', 'target', 'weight']
            sel = sel[sel['source'] != 'Main Page']
            sel = sel[sel['target'] != 'Main Page']
            tuples = [tuple(x) for x in sel.values]
            glist.append(igraph.Graph.TupleList(tuples, directed=False,
                                                edge_attrs=['weight']).simplify(combine_edges='first'))
            glist[n].vs["slice"] = n

        if sum([len(x.vs) for x in glist]) == 0:
            # print('empty')
            return [fp, 'error', 'empty']

        # print('generating agg network')
        igname = [{x.index: x['name'] for x in glist[y].vs}
                  for y in range(tlen)]

        del pcs
        del sel

        return [res, glist, igname, coreu, articles]

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        # raise
        return [fp, 'error', ex]


def cd2(res, glist, igname, coreu, articles):
    """
    Run Leiden algorithm on temporal network, returning community memberships.

    Parameters
    ----------
    res : float
        Resolution.
    glist : list
        List of networks (layers in temporal network).
    igname : list
        List of dicts with igraph node names.
    coreu : list
        Core articles (with underscores).
    articles : list
        Articles in network.


    Returns
    -------
    tuple
        membdf : DataFrame with community memberships.
        collmems : Dictionary of communities.
        tcd : Centralities of articles within each community.

    """
    try:
        print('running community detection')

        membership, improvement = la.find_partition_temporal(
            glist, la.CPMVertexPartition, vertex_id_attr='name',
            interslice_weight=1, resolution_parameter=res)
        membdf = pd.concat([pd.Series(x, index=glist[n].vs['name'])
                            for n, x in enumerate(membership)],
                           axis=1, sort=True)

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
            diff = np.where(tf.values is False)[0]-27
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

# =============================================================================
# Flat community detection comparison
# =============================================================================


def cdflatR(row, resrange, core, flatadj):
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
        collmemsFD : Dictionary of dictionaries with communities.
        tcdD : Dictionary of dictionaries with community centralities.

    """
    try:
        coreu = [x.replace(' ', '_') for x in core]
        articles = list(flatadj.columns)

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

        membdfFD = {}
        collmemsFD = {}
        for res in resrange:
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


def cmmatcher(n, x, colm, resrange, m, allev):
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
