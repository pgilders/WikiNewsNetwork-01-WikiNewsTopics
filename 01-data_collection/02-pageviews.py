#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:20:03 2022

@author: Patrick
"""

import datetime
import shutil
import requests
import functions1 as pgc

# %% download pageview files
# (originally done manually but have added script for completeness)

BPATH = '/Volumes/PGPassport/DPhil redo data/'

for ym in pgc.months_range(datetime.datetime(2017, 11, 1),
                           datetime.datetime(2018, 12, 1)):
    print(ym)
    url = 'https://dumps.wikimedia.org/other/pagecounts-ez/merged/pagecounts-%s-%s-views-ge-5.bz2' % (
        ym[:4], ym[4:])

    with requests.get(url, stream=True) as r:
        with open(BPATH + 'zips/' + 'pagecounts-%s-%s-views-ge-5.bz2'
                  % (ym[:4], ym[4:]), 'wb') as f:
            shutil.copyfileobj(r.raw, f)

# %% Unzip files later
