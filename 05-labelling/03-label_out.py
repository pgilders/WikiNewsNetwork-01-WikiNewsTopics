#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:20:32 2022

@author: 
"""

import pandas as pd
# import functions1 as pgc
import WikiNewsNetwork as wnn

# %% Import data
meanwj = pd.read_hdf('support_data/mean_topic_props.h5', key='df')
labelling = pd.read_excel('support_data/label agreement.xlsx',
                          engine="openpyxl", sheet_name=0, index_col=0)

labelling = pd.merge(labelling, meanwj, how='left',
                     left_index=True, right_index=True)

labelling['outlabel'] = labelling['Label 1'].str.strip() \
    + labelling.apply(lambda x: (int(x['Agreement 1'] == 'P') +
                                 int(x['Agreement 2'] == 'P'))*'*' +
                      int(x['Agreement 1'] == 'N')*'^', axis=1)

# %% Calculate % agreement
pc = 100/len(labelling)
un_strong = sum((labelling['Agreement 1'] == 'S')
                & (labelling['Agreement 2'] == 'S'))*pc
strong_partial = sum((labelling['Agreement 1'] == 'S')
                     | (labelling['Agreement 2'] == 'S'))*pc - un_strong
un_partial = sum((labelling['Agreement 1'] == 'P')
                 & (labelling['Agreement 2'] == 'P'))*pc
un_no = sum((labelling['Agreement 1'] == 'N')
            & (labelling['Agreement 2'] == 'N'))*pc
partial_no = sum((labelling['Agreement 1'] == 'N')
                 | (labelling['Agreement 2'] == 'N'))*pc - un_no


# %% Create latex table

outdf = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    outdf[f] = labelling.sort_values(f, ascending=False
                                     )['outlabel'].reset_index(drop=True)

odfv = pd.DataFrame()
for f in ['count', 'PROM', 'MAG', 'DEV']:
    odfv[f] = outdf[f].apply(lambda x: wnn.utilities.colourer(x, labelling))

odfv.columns = ['# Event Reactions', 'Prominence', 'Magnitude', 'Deviance']
