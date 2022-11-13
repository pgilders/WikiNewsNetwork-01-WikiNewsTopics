#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:20:05 2022

@author: 
"""

import calendar
import datetime
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

# %% Basic query functions


def chunks(li, n):
    """
    Split list li into list of lists of length n.

    Parameters
    ----------
    li : list
        Initial list.
    n : int
        Desired sublist size.

    Yields
    ------
    list
        Subsequent sublists of length n.

    """
    # For item i in a range that is a length of li,
    for i in range(0, len(li), n):
        # Create an index range for li of n items:
        yield li[i:i+n]


def query(request):
    """
    Query Wikipedia API with specified parameters.

    Parameters
    ----------
    request : dict
        API call parameters.

    Raises
    ------
    ValueError
        Raises error if returned by API.

    Yields
    ------
    dict
        Subsequent dicts of json API response.

    """
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify with values from the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            'https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            print('ERROR')
            # print(result['error'])
            raise ValueError(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']
        if 'continue' not in result:
            break
        lastContinue = result['continue']


def parse(request):
    """
    Query Wikipedia API with specified parameters to parse data.

    Parameters
    ----------
    request : dict
        API call parameters.

    Raises
    ------
    ValueError
        Raises error if returned by API.

    Yields
    ------
    dict
        Subsequent dicts of json API response.

    """
    request['action'] = 'parse'
    request['format'] = 'json'
    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify with values from the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            'https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            print('ERROR')
            # print(result['error'])
            raise ValueError(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'parse' in result:
            yield result['parse']
        if 'continue' not in result:
            break
        lastContinue = result['continue']

# %% Current events portal


def wiki_news_articles(months):
    """
    Generate dataframe of events from the Wikipedia Current Events Portal.

    Parameters
    ----------
    months : iterable
        List of months to get events for.

    Returns
    -------
    DataFrame
        Events DF with Date, Category, Text, Articles, Ext links, HTML columns

    """
    stories_df = pd.DataFrame(columns=['Date', 'Category', 'Text', 'Articles',
                                       'Ext links', 'HTML'])  # create df

    for m in months:
        print(m)
        params = {'page': 'Portal:Current_events/%s' % m, 'prop': 'text'}
        # Get data from current events page for that month
        scr = list(parse(params))

        doc = bs(scr[0]['text']['*'], 'html.parser')  # BeautifulSoup the html

        # tag for each day
        # ##### NOTE THAT THIS CHANGES BASED ON MONTH
        # Previously "table", {"class" : "vevent"}
        days = doc.findAll("div", {"class": "vevent"})
        for d in days:
            day = d['id']  # date
            # get categories in list
            # cats = d.findAll("div", {'role': 'heading',
            #                          "style": "margin-top:0.3em; font-size:inherit; font-weight:bold;"})
            cats = d.findAll("div", {'role': 'heading',
                                     "class":
                                         "current-events-content-heading"})
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
                    html = s
                    links = s.findAll("a")  # get links in description
                    articles = []
                    ext = []
                    for li in links:
                        try:
                            # add wiki links to list//
                            articles.append(li['title'])
                        except KeyError:
                            # add ext links to another list
                            ext.append(li['href'])

                    # add event data to df
                    stories_df = stories_df.append({'Date': day,
                                                    'Category': cat_name,
                                                    'Text': txt,
                                                    'Articles': articles,
                                                    'Ext links': ext,
                                                    'HTML': str(html)},
                                                   ignore_index=True)

    return stories_df


# %% Redirects

def fix_redirects(articles, existingmap={}):
    """
    Map redirect name to true Wikipedia article name.

    Parameters
    ----------
    articles : iterable
        Wikipedia article names.
    existingmap : dict, optional
        If a (partial) existing map exists, combine. The default is {}.

    Returns
    -------
    dict
        Map with redirect name keys and true article name values.

    """
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
    """
    Get all redirects to a true Wikipedia article name.

    Parameters
    ----------
    articles : iterable
        Wikipedia article names.
    existingrds : dict, optional
        If a (partial) existing map exists, combine. The default is {}.

    Returns
    -------
    dict
        Map with true article name keys and all names that redirect to it
        in a list as values.
    """
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

# %% Pageviews


def text_to_tseries(text, year, month):
    """
    Convert compressed Wiki file line of text to page view time series.

    Parameters
    ----------
    text : str
        String with article pageviews.
    year : int
        Year.
    month : int
        Month.

    Returns
    -------
    Series
        Series with hourly page views.

    """
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
