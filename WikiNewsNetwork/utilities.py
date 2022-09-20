#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:18:39 2022

@author: Patrick
"""

from calendar import monthrange
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


# %% Date functions


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
    """
    Take event name and return dict of associated month lengths.

    Parameters
    ----------
    ev : str
        Event folder name.

    Returns
    -------
    dict
        Dictionary of month keys ±30 days with their length in days as values.

    """
    date = pd.to_datetime(ev[:8])
    start = date-datetime.timedelta(days=30)
    stop = date+datetime.timedelta(days=30)

    months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
    return {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
            for x in months}

# %% Set functions


def fdiv(a, b):
    """
    Divide smaller number by larger number.

    Parameters
    ----------
    a : number
    b : number

    Returns
    -------
    float
        Quotient of a and b.

    """
    if a < b:
        return a/b
    return b/a


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
    if not isinstance(x, set):
        x = set(x)
    if not isinstance(y, set):
        y = set(y)
    return len(x & y)/len(x | y)


def wjac(x, y):
    """
    Weighted Jaccard index of sets.

    Parameters
    ----------
    x : dict
        weights of items in set x.
    y : dict
        weights of items in set y.

    Returns
    -------
    float
        In range 0-1.

    """
    li = [fdiv(x.get(i, 0), y.get(i, 0))
          for i in set(x.keys()) | set(y.keys())]

    return sum(li)/len(li)

# %% Latex output


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
    ou = simdf[simdf['outlabel'] == x]['SS_score']
    lq, mm, uq = simdf['SS_score'].quantile([0.25, 0.5, 0.75])

    assert len(ou) == 1

    cdict = {uq: 'green', mm: 'yellow', lq: 'orange', 0: 'red'}

    for q in [uq, mm, lq, 0]:
        if ou.iloc[0] > q:
            return '\\cellcolor{%s}%s' % (cdict[q], x.replace('&', '\\&'))
