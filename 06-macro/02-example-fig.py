#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:31:11 2023

@author: Patrick
"""

import pandas as pd
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import RobustScaler
from matplotlib.lines import Line2D
import WikiNewsNetwork as wnn

plt.style.use('seaborn-darkgrid')
with open('figures/figurestyle.json', 'r') as f:
    params = json.load(f)
params['legend.handlelength'] = 1
params['legend.title_fontsize'] = 20
plt.rcParams.update(params)


# %% Load data

BPATH = ''

with open('support_data/redir_arts_map.json') as json_data:
    redir_arts_map = json.load(json_data)
    json_data.close()

rdarts_rev = {x: k for k, v in redir_arts_map.items() for x in v}

evlsn = pd.read_hdf(BPATH + 'aux_data/evlsN.h5', key='df')
eventsdf = pd.read_hdf('support_data/eventsdf.h5')


#%% Load data for shortlist events       

evs = ['20181130_2018 Anchorage earthquake--Moment magnitude scale--Anchorage, Alaska_CS']

core_D = {}
G_D = {}
igname_D = {}
articles = {}
DDC = {}
DDC2 = {}
comms_D = {}
commsw_D = {}
tsd = {}
excessdf = pd.read_hdf('support_data/excessdf.h5')

for n, e in enumerate(evs):
    with pd.HDFStore(BPATH + 'events/' + e +'/tcweights.h5', mode='r'
                     ) as hdf:
        for k in hdf.keys():
            DDC[e + '/' + k[1:].replace('/', ':')
                ] = pd.read_hdf(BPATH + 'events/' + e +'/tcweights.h5',
                                key=k)
            
    for n, (k, v) in enumerate(DDC.items()):
        if k.split('/')[-2] not in DDC2.keys():
            DDC2[k.split('/')[-2]] = {k.split('/')[-1]: v}
        else:
            DDC2[k.split('/')[-2]][k.split('/')[-1]] = v  
    
    with open(BPATH + 'events/' + e + '/f_comms.pkl', 'rb') as f:
        comms_D[e] = pickle.load(f)
        
    with open(BPATH + 'events/' + e + '/w_comms.pkl', 'rb') as f:
        commsw_D[e] = pickle.load(f)   

    core_D[e], articles[e], G_D[e], igname_D[e] = wnn.cd.read_f_data(
        BPATH + 'events/' + e, rdarts_rev)
    
    tsd[e] = pd.read_hdf(BPATH + 'events/' + e + '/tsNN.h5')[G_D[e].vs['name']]
    tsd[e].index = range(-30, 31)

#%% Plotting functions

def share_axes(axs, columns):
    for c in range(columns):
        axs[3,c].get_shared_x_axes().join(axs[3,c], axs[4,c])
        axs[3,c].set_xticklabels([])            
    axs[3,0].get_shared_y_axes().join(*[axs[3,c]
                                        for c in range(columns)])
    
def subtitles_legends(fig, axs, com, palette):

    # create network title & legend
    gs = axs[0, 1].get_gridspec()
    for ax in axs[0,:]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :])
    axbig.axis('off')     
    axbig.text(0.5, 0.65, 'Article Hyperlink Network Structure',
               ha='center', va='center', fontsize=30)
    l_elements1 = [Line2D([0], [0], color='k', marker='o', lw=0,
                               ms=5, label='Regular Article'),
                        Line2D([0], [0], color='k', markeredgecolor='w',
                               marker='*', lw=0, ms=15,
                               label='Core Article')]
    l_elements2 = [Line2D([0], [0], color=palette[n], marker='*', lw=0, 
                              markeredgecolor='w', ms=15, label=k)
                        for n, k in enumerate(com)]
    l1 = axbig.legend(handles=l_elements1, loc=8, ncol=len(l_elements1),
                      bbox_to_anchor=(0.5, 0))

    l2 = axbig.legend(handles=l_elements2, handlelength=0, loc=8,
                   ncol=len(l_elements2), bbox_to_anchor=(0.5, -0.25))
    axbig.add_artist(l1)
    axbig.add_artist(l2)


    # create page view title and legend
    gs = axs[2, 1].get_gridspec()
    for ax in axs[2,:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, :])
    axbig.axis('off')
    legend_elements1 = [Line2D([0], [0], color='k', lw=1,
                               label='Individual Article'),
                        Line2D([0], [0], color='k', lw=4,
                               label='Community Mean'),
                        Line2D([0], [0], color='k', lw=4, ls = ':',
                               label='Core Article')]
    axbig.legend(handles=legend_elements1, loc=8, ncol=3,
                       bbox_to_anchor=(0.5, -0.25))
    axbig.text(0.5, 0.65, 'Page Views to Articles', ha='center',
             va='center', fontsize=30)   

def gen_labels(partitions, ts, tsxv):
    az = list(map(chr, range(97, 123)))

    subp = ([((3, n), com, 'Page Views')
            for n, com in enumerate(partitions)] +
            [((4, n), com, 'Scaled Page Views')
                    for n, com in enumerate(partitions)])
    td = {3:ts, 4:tsxv}
    sd = {3:'log', 4:'symlog'}
    ld = {(row,c):az[(row//2)*len(partitions)+c] for row in [3,4]
          for c in range(len(partitions))}

    subn = [((1,n), com, az[n]) for n, com in enumerate(partitions)]
             
    return subn, subp, td, sd, ld

def plot_pageviews(ax, cdt, com, ylab, tsk, palette, yscale, lab):
         
    for m, (k, v) in enumerate(com.items()):
        ax.plot(tsk[set(tsk.columns)&set(v)], c=palette[m], lw=1,
                alpha=0.25)
    for m, k in enumerate(com.keys()):
        ax.plot(tsk[k.split('---')], c=palette[m], lw=4, ls=':')
    for m, (k, v) in enumerate(com.items()):
        ax.plot(tsk[set(tsk.columns)&set(v)].mean(axis=1), c=palette[m],
                lw=4)
    if cdt[1]==0:
        ax.set_ylabel(ylab)
    if cdt[0]==4:
        ax.set_xlabel('Days from Event')
    ax.set_yscale(yscale)
    ax.text(.01, .99, lab, ha='left', va='top', fontsize=30,
                  transform=ax.transAxes)

    return ax

def plot_network(net, pos, ax, com, lab, palette, core_D):
    cd_articles = {y for x in com.values() for y in x}
    comm_id = {y:m for m, (k, v) in enumerate(com.items()) for y in v}
    node_color = {x:palette[comm_id[x]] if x in comm_id else 'grey'
                  for x in list(net.nodes())}
    co_nodes = [x for x in net.nodes()
                if (x in core_D[e])&(x in cd_articles)]
    c_nodes = [x for x in net.nodes()
               if (x not in core_D[e])&(x in cd_articles)]
    o_nodes = [x for x in net.nodes()
               if (x not in core_D[e])&(x not in cd_articles)]      

    nx.draw_networkx_edges(net, pos, ax=ax, width=0.1,
                           edge_color='grey', alpha=0.8,
                           edgelist = [x for x in net.edges
                                       if (x[0] not in cd_articles)|
                                       (x[1] not in cd_articles)])
    nx.draw_networkx_nodes(net, pos, ax=ax, node_size=5,
                           nodelist=o_nodes,
                           node_color='grey',
                           alpha=0.5, node_shape='o')
    nx.draw_networkx_edges(net, pos, ax=ax, width=1,
                           edge_color='k', alpha=1,
                           edgelist =  [x for x in net.edges
                                       if (x[0] in cd_articles)&
                                       (x[1] in cd_articles)])
    nx.draw_networkx_nodes(net, pos, ax=ax, node_size=15,
                            nodelist=c_nodes,
                            node_color=[node_color[x]
                                        for x in c_nodes],
                            node_shape='o')
    core_n = nx.draw_networkx_nodes(net, pos, ax=ax, node_size=200,
                           nodelist=co_nodes, node_shape='*',
                           node_color=[node_color[x]
                                       for x in co_nodes])
    core_n.set_edgecolor('w')
    
    l_elements1 = ([Line2D([0], [0], color=palette[n], marker='o',
                                lw=0, ms=10, label='%d (%d)'
                                %(n+1, len(list(com.values())[n])))
                         for n in range(len(com))]
                       + [Line2D([0], [0], color='grey', alpha=0.5,
                                 marker='o', lw=0, ms=10,
                                 label='Other (%d)'
                                 %len(set(net.nodes)-cd_articles))])
    # l_elements2 = [Line2D([0], [0], color=palette[n], marker='*', lw=0, 
                          # markeredgecolor='w', ms=15, label=k)
                        # for n, k in enumerate(com)]
 
    l1 = ax.legend(handles=l_elements1, handlelength=0, loc=8,
                   title='Community (size)', ncol=len(l_elements1))
    l1._legend_box.align='left'
    # l2 = ax.legend(handles=l_elements2, handlelength=0, loc=9, fontsize=12,
                   # ncol=len(l_elements2))
    # l2._legend_box.align='left'
    ax.add_artist(l1)
    # ax.add_artist(l2)
    ax.set_ylim([ax.get_ylim()[0]*1.05, ax.get_ylim()[1]])
    ax.grid(False)
    ax.text(.01, .99, lab, ha='left', va='top', fontsize=30,
                  transform=ax.transAxes)
    
    return ax
    
def fig_adjust(fig, title, xdf=pd.DataFrame()):
    
    fig.suptitle(title,
                 fontsize=32)
    if not xdf.empty:
        fig.text(0.1, 0.94, e, ha='left',
                 va='center', fontsize=16)
        xdftext = '  '.join(xdf.loc[e, ['X_f', 'X_w', 'X_t',
                                        'X7_f', 'X7_w', 'X7_t']
                                    ].apply(round).astype(str))
        fig.text(0.1, 0.89, xdftext, ha='left', va='center',
                 fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    
def plot_megafig(net, ts, tsxv, partitions, title, pos=None,
                 xdf=pd.DataFrame(), savefigpath=None):
    
    palette = sns.color_palette('colorblind', max([len(x)
                                                   for x in partitions]))
    if not pos:       
        pos = nx.spring_layout(net, k=10/len(net.nodes)**0.5)
     
    fig, axs = plt.subplots(5, len(partitions),
                            figsize=(len(partitions)*8, 24), sharey='row',
                            gridspec_kw={'height_ratios':
                                         [0.2,1,0.15,1,1]})
           
    share_axes(axs, len(partitions))
    
    subn, subp, td, sd, ld = gen_labels(partitions, ts, tsxv)
    subtitles_legends(fig, axs, subn[0][1], palette)          


    # draw networks
    for cdt, com, lab in subn:
        axs[cdt] = plot_network(net, pos, axs[cdt], com, lab, palette, core_D)
        
    #draw page views       
    for cdt, com, ylab in subp:
        axs[cdt] = plot_pageviews(axs[cdt], cdt, com, ylab, td[cdt[0]], palette,
                       sd[cdt[0]], ld[cdt])
 
    # final adjustments    
    fig_adjust(fig, title, xdf)
    if savefigpath:
        for ext in ['svg', 'eps', 'pdf', 'png']:
            plt.savefig(savefigpath %ext)        
    
    plt.show()
    


#%% Double & Triple plots with flat, weighted, and temp
  
for n, e in enumerate(evs):
    try:
            
        # Flat
        g = G_D[e]
        net = nx.from_edgelist([(names[x[0]], names[x[1]])
                          for names in [g.vs['name']] # simply a let
                          for x in g.get_edgelist()], nx.Graph())
        net.remove_edges_from(nx.selfloop_edges(net))
        ts = tsd[e].copy()
        scaler = RobustScaler()
        tsxv = pd.DataFrame(scaler.fit_transform(ts), index=ts.index,
                            columns=ts.columns)

        commst = DDC2[e].copy()
        comms = comms_D[e]
        commsw = commsw_D[e]
        commst = {k:list(v.index) for k, v in commst.items()}
    
        comms, commsw = wnn.cd.rearrange_comms(comms, commsw)
        comms, commst = wnn.cd.rearrange_comms(comms, commst)
        commsw, commst = wnn.cd.rearrange_comms(commsw, commst)
        
        pos = nx.spring_layout(net, k=10/len(net.nodes)**0.5)        
        plot_megafig(net, ts, tsxv, [comms, commsw, commst],
                     'Comparison of Communities from Static (Left) vs Static Weighted (Centre) vs Temporal (Right)\nCommunity Detection Approaches',
                     pos, savefigpath='figures/cdcompare_fwt.%s')
        plot_megafig(net, ts, tsxv, [comms, commst],
                     'Comparison of Communities from Static (Left) vs Temporal (Right)\nCommunity Detection Approaches',
                     pos, savefigpath='figures/cdcompare_ft.%s')
        plot_megafig(net, ts, tsxv, [commsw, commst],
                     'Comparison of Communities from Static Weighted (Left) vs Temporal (Right)\nCommunity Detection Approaches',
                     pos, savefigpath='figures/cdcompare_wt.%s')
        print(e, eventsdf.loc[e, 'Text'], sep='\n')

    except KeyboardInterrupt:
        break
    except Exception as ex:
        raise
        print(e, ex)
        
        


  