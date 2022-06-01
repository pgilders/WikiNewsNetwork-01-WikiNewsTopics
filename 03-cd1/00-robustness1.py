#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:27:01 2022

@author: Patrick
"""


def wjac(x, y):
    """
    Weighted Jaccard index of prealigned sets.

    Parameters
    ----------
    x : iterable, numeric
        weights of items in set x (0s for items not in original set x).
    y : iterable, numeric
        weights of items in set y (0s for items not in original set y).

    Returns
    -------
    float
        In range 0-1.

    """
    df = pd.DataFrame([x, y]).T.fillna(0)
    return (df.min(axis=1)/df.max(axis=1)).mean()


def wjac2(x, y):
    l = [min(x.get(i, 0), y.get(i, 0))/max(x.get(i, 0), y.get(i, 0))
         for i in set(x.keys()) | set(y.keys())]

    return sum(l)/len(l)


def fdiv(a, b):
    if a < b:
        return a/b
    else:
        return b/a


def wjac3(x, y):
    l = [fdiv(x.get(i, 0), y.get(i, 0))
         for i in set(x.keys()) | set(y.keys())]

    return sum(l)/len(l)


# %%
combos = [(DDC[k1], DDC[k2]) for n, k1 in enumerate(sorted(DDC.keys())[:500])
          for m, k2 in enumerate(sorted(DDC.keys())[:500]) if n < m]


method 1
# %%
t = time.time()
m1 = [wjac(x[0], x[1]) for x in combos[:10000]]
print(time.time()-t)
# %%
t = time.time()
m2 = [wjac2(dict(x[0]), dict(x[1])) for x in combos[:100000]]
print(time.time()-t)
# %%
t = time.time()
m3 = [wjac3(x[0], x[1]) for x in combos[37000000:37000000+301580]]
print(time.time()-t)
# %%
