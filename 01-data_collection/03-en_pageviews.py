#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:05:08 2022

@author: Patrick
"""

# filter to en pageviews


import glob
BPATH = '/Volumes/PGPassport/DPhil redo data/'

summfile = open(BPATH+'pageviews/raw/summ.txt', "w+")
pvfiles = sorted(glob.glob(BPATH+'pageviews/raw/pagecounts-*-views-ge-5'))

for inpath in pvfiles:
    print(inpath)
    outpath = inpath.replace('views-ge-5', 'viewen')

    filein = open(inpath, "r", encoding="utf-8")
    fileout = open(outpath, "w+", encoding="utf-8")

    summfile.write('i/o files opened\n')
    print('i/o files opened\n')

    enzone = False
    for line in filein:
        if line[0] == '#':
            continue
        if line[:line.index(' ')] == 'en.m':
            fileout.write(line)
        elif line[:line.index(' ')] in ['en.z', 'en.zero']:
            if enzone is False:
                print('settingenzone')
                enzone = True
            fileout.write(line)
        elif enzone is True:
            print('breaking')
            break

    filein.close()
    fileout.close()

    summfile.write('o file written\n')
    print('o file written\n')
