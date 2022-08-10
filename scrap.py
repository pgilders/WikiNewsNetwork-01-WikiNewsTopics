#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:35:52 2022

@author: Patrick
"""
# compare jlls

jll = []
for n in range(0, 132787956, int(1E6)):
    print(n, 132787956)
    with open('/Users/Patrick/active_data/jw/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll.extend(json.load(f))


jll2 = []
for n in range(0, 132787956, int(1E6)):
    print(n, 132787956)
    with open('/Users/Patrick/active_data/jw2/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll2.extend(json.load(f))
# %%

diffs = []
for n in range(132787956):
    if n % 10000000 == 0:
        print(n/132787956)
    if abs(jll[n]-jll2[n]) > 0.000001:
        print(n, jll[n], jll2[n])
        diffs.append((n, jll[n], jll2[n]))

# %%


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


arr = np.zeros((len(DDC), len(DDC)))
ixsall = np.array(np.triu_indices(len(DDC), 1))
ixc = ixsall
# ixspre = ixsall[:, :82000000] ####?????????
# ixspost = ixsall[:, 82000000:]
# ixspost = vec_translate(ixspost, imp)
# ixc = np.concatenate([ixspre, ixspost], axis=1)
ixt = (ixc[0], ixc[1])
arr[ixt] = jll


# %%

arr2 = np.zeros((len(DDC), len(DDC)))
ixsall = np.array(np.triu_indices(len(DDC), 1))
ixc = ixsall
# ixspre = ixsall[:, :82000000] ####?????????
# ixspost = ixsall[:, 82000000:]
# ixspost = vec_translate(ixspost, imp)
# ixc = np.concatenate([ixspre, ixspost], axis=1)
ixt = (ixc[0], ixc[1])
arr2[ixt] = jll2

# %%

arrT = np.zeros((len(DDC), len(DDC)))
ixsall = np.array(np.triu_indices(len(DDC), 1))
ixc = ixsall
ixspre = ixsall[:, :82000000]  # ?????????
ixspost = ixsall[:, 82000000:]
ixspost = vec_translate(ixspost, imp)
ixc = np.concatenate([ixspre, ixspost], axis=1)
ixt = (ixc[0], ixc[1])
arrT[ixt] = jll
# %%

with open('/Users/Patrick/Downloads/Downloads Folder Data/ddck.json', 'r') as f:
    ddc2 = json.load(f)

dcdf = pd.DataFrame([sorted(DDC.keys()), ddc2]).T
dcdf[0] = dcdf[0].str.split(
    '/').str[-2:].apply(lambda x: unicodedata.normalize('NFKD', '/'.join(x)))
dcdf[1] = dcdf[1].str.split(
    '/').str[-2:].apply(lambda x: unicodedata.normalize('NFKD', '/'.join(x)))


def sortfix(x):
    try:
        return dcdf[(dcdf[0] == x[1])].index[0]
    except:

        return x.name


dmap = dcdf[(dcdf[0] != dcdf[1])][[1]].apply(lambda x: sortfix(x), axis=1)
dmd = dmap.to_dict()
ixmap = dcdf[[1]].apply(lambda x: dmd.get(x.name, x.name), axis=1)
imp = ixmap.to_dict()

# %%

e = '/Volumes/Samsung_T3/evtest/20180625_BHP--Brazil_CS'

el = pgc.getel(e, csd, megamap, redir_arts_map)


# %%


date = pd.to_datetime(e.split('evtest/')[1][:8])
start = date-datetime.timedelta(days=30)
stop = date+datetime.timedelta(days=30)

months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))
mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]),
                                             int(x[4:]))[1] for x in months}
mrk = frozenset(mr.keys())

#        print('reading')
core = set(pd.read_csv(e+'/core.tsv', sep='\t', header=None)[0])

coremm = {megamap.get(x, x).replace(' ', '_') for x in core}
corerd = {y for x in coremm for y in redir_arts_map.get(x, [x])}
# return 0

#        print('csdf')
ndf = csd[mrk][(csd[mrk]['curr'].isin(corerd)) |
               (csd[mrk]['prev'].isin(corerd))].copy()
#        print('ndf')
ndfarticles = set(ndf['curr']) | set(ndf['prev'])

#        print('el')
el = csd[mrk][(csd[mrk]['curr'].isin(ndfarticles)) &
              (csd[mrk]['prev'].isin(ndfarticles))].copy()
el[el.columns[2]] = el[el.columns[2]] * \
    ((start.replace(day=mr[el.columns[2]])-start).days+1)
el[el.columns[-2]] = el[el.columns[-2]] * \
    ((stop-stop.replace(day=1)).days+1)
el[el.columns[3:-2]] = el[el.columns[3:-2]
                          ].apply(lambda x: mr[x.name]*x)
el['n'] = el[el.columns[2:-1]].sum(axis=1)
el = el[el['n'] > 100]
# %%
el0 = pd.read_hdf(i+elp)
el0d = pd.read_hdf(i+'/all_el100d.h5')

# %%

set(el['prev']+'____'+el['curr'])
set(el0['prev']+'____'+el0['curr'])

# %%

date = datetime.datetime.strptime(
    i.split('/evtest/')[-1][:8], '%Y%m%d')-datetime.timedelta(days=day_offset)
start = date-datetime.timedelta(days=30)
stop = date+datetime.timedelta(days=30)

# print('mc')
months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))
mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
      for x in months}

core = [megamap.get(x, x).replace(' ', '_') for x in pd.read_csv(
    i+'/core.tsv', sep='\t', header=None)[0]]
coreu = {x.replace('_', ' ') for x in core}

el = el[1]
el['prev'] = el['prev'].str.replace('_', ' ').map(megamap)
el['curr'] = el['curr'].str.replace('_', ' ').map(megamap)
el = el[el['prev'] != 'Main Page']
el = el[el['curr'] != 'Main Page']
articles = set(el['prev']) | set(el['curr'])

# print('elops')

el = el[el['n'] > 100]

articles = set(el['prev'][el['curr'].isin(coreu)]) | set(
    el['curr'][el['prev'].isin(coreu)]) | coreu

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


# %%

mm1 = '/Volumes/PGPassport/active_data_2019/megamap.json'
mm2 = '/Users/Patrick/Downloads/Downloads Folder Data/megamap1308.json'
mm3 = '/Users/Patrick/Downloads/Downloads Folder Data/megamap.json'
mm4 = '/Users/Patrick/Downloads/Downloads Folder Data/servevs2/megamap.json'

mml = [mm1, mm2, mm3, mm4]
# %%

rdm1 = '/Users/Patrick/Downloads/Downloads Folder Data/servevs2/redir_arts_map.json'
rdm2 = '/Volumes/PGPassport/active_data_2019/redir_arts_map.json'
rdm3 = '/Users/Patrick/Downloads/Downloads Folder Data/redir_arts_map.json'

rdl = [rdm1, rdm2, rdm3]

# %%

rt = []
mt = []
for n in range(4):

    with open(mml[n]) as json_data:
        mt.append(json.load(json_data))
        json_data.close()

    with open(rdl[n]) as json_data:
        rt.append(json.load(json_data))
        json_data.close()


# %%
pcs = np.nan_to_num(np.load(
    '/Volumes/Samsung_T3/evtest/20180625_BHP--Brazil_CS/simtgraph5NN.npz')['arr_0'])
# %%

getevls(el, pd.DataFrame(), tsl, megamap, day_offset=0, elp='/all_el100.h5')
# %%


el = pd.read_csv(e+'/all_el.tsv', sep='\t', index_col=0)
# %%
el[el.columns[2]] = el[el.columns[2]] * \
    ((start.replace(day=mr[el.columns[2]])-start).days+1)
el[el.columns[-2]] = el[el.columns[-2]] * \
    ((stop-stop.replace(day=1)).days+1)
el[el.columns[3:-2]] = el[el.columns[3:-2]
                          ].apply(lambda x: mr[x.name]*x)
el['n'] = el[el.columns[2:-1]].sum(axis=1)
el = el[el['n'] > 100]
# %%


# %%
stt = {x.split('est/')[-1] for x in glob.glob("/Volumes/Samsung_T3/evtest/*")}
# %%
stt0 = {x[:8] + '_' +
        '--'.join(sorted(x[9:-3].split('--')))+'_CS' for x in stt}
stt0rd = {x[:8] + '_' + '--'.join(sorted([megamap.get(z, z)
                                  for z in x[9:-3].split('--')]))+'_CS' for x in stt}
stta0 = {x[:8] + '_' +
         '--'.join(sorted(x[9:-3].split('--')))+'_CS' for x in stta}
sttrd0 = {x[:8] + '_' +
          '--'.join(sorted(x[9:-3].split('--')))+'_CS' for x in sttrd}

# %%

stt = {x.split('est/')[-1] for x in glob.glob("/Volumes/Samsung_T3/evtest/*")}
si = set(storiesdf.index)

# %%
sfiles = '20180608_44th G7 summit--Group of Seven--La Malbaie--Quebec--Canada--United States--European Union--President of the United States--Donald Trump--Russia--Group of Eight--Annexation of Crimea by the Russian Federation--Giuseppe Conte_CS',
sindex = '20180608_44th G7 summit--Group of Seven--La Malbaie--Quebec--Canada--United States--European Union--U.S. President--Donald Trump--Russia--Group of Eight--Annexation of Crimea by the Russian Federation--Giuseppe Conte_CS'
stt = '20180608_44th G7 summit--Group of Seven--La Malbaie--Quebec--Canada--United States--European Union--U.S. President--Donald Trump--Russia--Group of Eight--Annexation of Crimea by the Russian Federation--Giuseppe Conte_CS   '
strdt = '20180608_44th G7 summit--Group of Seven--La Malbaie--Quebec--Canada--United States--European Union--U.S. President--Donald Trump--Russia--Group of Eight--Annexation of Crimea by the Russian Federation--Giuseppe Conte_CS   '

# %%

foldernamemap = {}

storiesdfindex = pd.Series(storiesdf.index, index=storiesdf.index)
storiesdfartjoin = storiesdf.apply(lambda x: x['Date'].strftime(
    '%Y%m%d')+'_'+'--'.join(x['Articles'])+'_CS', axis=1)
storiesdfrdartjoin = storiesdf.apply(lambda x: x['Date'].strftime(
    '%Y%m%d')+'_'+'--'.join(x['rdArticles'])+'_CS', axis=1)
storiesdfrdartsortjoin = storiesdf.apply(lambda x: x['Date'].strftime(
    '%Y%m%d')+'_'+'--'.join(sorted(x['rdArticles']))+'_CS', axis=1)
# %%

fmapd = {}
nn = 0
for foldername in [x.split('est/')[-1] for x in glob.glob("/Volumes/Samsung_T3/evtest/*")]:

    for cat in [storiesdfindex, storiesdfartjoin, storiesdfrdartjoin,
                storiesdfrdartsortjoin]:
        if foldername in cat.str.lower().values:
            ix = cat[cat.str.lower() == foldername.lower()].index[0]
            foldernamemap[foldername] = [storiesdfindex[ix],
                                         storiesdfartjoin[ix],
                                         storiesdfrdartjoin[ix],
                                         storiesdfrdartsortjoin[ix]]
            break

        date = foldername[:8]
        arts = foldername[9:-3].split('--')
        rdarts = {megamap.get(x, x).lower() for x in arts}
        dsets = storiesdf[storiesdf['Date'] == date]['rdArticles'].apply(set)
        dsj = dsets.apply(lambda x: len(
            {y.lower() for y in x} & rdarts)/len({y.lower() for y in x} | rdarts))
        if len(dsj[dsj >= 0.5] == 1):
            # print(nn)
            nn += 1
            ix = storiesdfindex[dsj[dsj >= 0.5].index[0]]
            foldernamemap[foldername] = [storiesdfindex[ix],
                                         storiesdfartjoin[ix],
                                         storiesdfrdartjoin[ix],
                                         storiesdfrdartsortjoin[ix]]
        else:
            fmapd[foldername] = dsj.sort_values()[-1:]
# %%
foldernamemap2 = {}

for k, v in fmapd.items():
    if v.iloc[0] > 0:
        print(k, '\n', v.index[0])
        io = input()
        if io == 'y':
            ix = v.index[0]
            foldernamemap2[k] = [storiesdfindex[ix],
                                 storiesdfartjoin[ix],
                                 storiesdfrdartjoin[ix],
                                 storiesdfrdartsortjoin[ix]]
# %%


sii = {x[:8]+'_' + '--'.join(sorted(x[9:-3].split('--')))
       for x in storiesdf.index}
levi = {x[:8]+'_' + '--'.join(sorted(x[9:-3].split('--'))) for x in levl}

etl = glob.glob('/Volumes/PGPassport/evtLocal/*')
trli = {x.split('al/')[-1] for x in etl}
trli = {x[:8]+'_' + '--'.join(sorted(x[9:-3].split('--'))) for x in trli}
# %%
lll
for fp in glob.glob('/Volumes/PGPassport/DPhil redo data/pageviews/en/pagecounts-*-viewen'):
    sset = set()
    with open(fp, 'r') as f:
        for l in f:
            sset.add(l.split(' ')[0])
    print(fp, sset)

# %%

lastday = calendar.monthrange(year, month)[1]
ts = text.split(',')[:-1]
pvd = {ord(x[0])-64: x[1:] for x in ts}
pvds = {datetime.datetime.strptime('-'.join(['%d%02d' % (year, month),
                                             str(k), str(ord(x[0])-65)]),
                                   '%Y%m-%d-%H'):
        int(x[1].replace('?', '0')) if x[1] else 0 for k, v in pvd.items()
        for x in re.findall('([A-Z])([\?\d]*)', v)}

pd.Series(pvds, index=pd.date_range('%d%02d01' % (year, month),
                                    '%d%02d%d2300' % (year, month,
                                                      lastday),
                                    freq='H')).fillna(0)


def text_to_tseries2(text, year, month):
    lastday = calendar.monthrange(year, month)[1]
    ts = text.split(',')[:-1]
    pvd = {ord(x[0])-64: x[1:] for x in ts}
    pvds = {datetime.datetime.strptime('-'.join(['%d%02d' % (year, month),
                                                 str(k), str(ord(x[0])-65)]),
                                       '%Y%m-%d-%H'):
            int(x[1].replace('?', '0')) if x[1] else 0 for k, v in pvd.items()
            for x in re.findall('([A-Z])([\?\d]*)', v)}

    return pvds
# %%


def getel(e, csd, megamap, redir_arts_map, eventsdf):
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
        ndfarticles = get_neighbours_quick(e, csd, redir_arts_map, eventsdf)[1]

#        print('el')
        el = csd[mrk][(csd[mrk]['curr'].isin(ndfarticles)) &
                      (csd[mrk]['prev'].isin(ndfarticles))].copy()
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


def get_neighbours_quick(e, csd, corerds, eventsdf):
    try:

        date = pd.to_datetime(e[:8])
        start = date-datetime.timedelta(days=30)
        stop = date+datetime.timedelta(days=30)

        months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
        mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
              for x in months}
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
# %%


# mk = set(m_map.keys())
# mv = set(m_map.values())
# an = allneighbours_set
# anm = {m_map.get(x.replace('_', ' '), x).replace(' ', '_') for x in an}
mmk = set(megamap.keys())
mmk = set(megamap.values())

# rk = set(rd_arts_map.keys())
rrk = set(redir_arts_map.keys())


# %%

an-rk
an-rrk

rk-an
rrk-an


# %%

date = pd.to_datetime(e[:8])
start = date-datetime.timedelta(days=30)
stop = date+datetime.timedelta(days=30)

months = pgc.months_range(pd.to_datetime(start), pd.to_datetime(stop))
mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
      for x in months}
mrk = frozenset(mr.keys())

ndfarticles = get_neighbours_quick(e, csd, redir_arts_map, eventsdf)[1]

#        print('el')
el = csd[mrk][(csd[mrk]['curr'].isin(ndfarticles)) &
              (csd[mrk]['prev'].isin(ndfarticles))].copy()

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
# %%

errors = []
eventsdf['MR'] = eventsdf.apply(lambda x: frozenset(pgc.getmr(x.name).keys()),
                                axis=1)
mrset = set(eventsdf['MR'])

for mn, mrk in enumerate(mrset):
    print('\nMonth keys: %.2f %%\n==================' % (100*mn/len(mrset)))
    mr = {x: monthrange(int(x[2:6]), int(x[7:9]))[1] for x in mrk}
    csd = mcsdf[['prev', 'curr']+sorted(mr)].copy().fillna(0)
    for k, v in mr.items():
        csd[k] = csd[k]*v
    csd['n'] = csd[csd.columns[3:]].sum(axis=1)
    csd = csd[csd['n'] > 0]
    # el = el[el['n'] > 100]
    for k, v in mr.items():
        csd[k] = csd[k]/v

    events = list(eventsdf[(eventsdf['MR'] == mrk) &
                           (eventsdf['Articles'].str.len() > 0)].index)

    # att = pgc.getel_2(events[0], csd, redir_arts_map, rdarts_rev, eventsdf)
    # if len(att)==3:
    #     print(att, mrk)

    print('%d Events' % len(events))
    for x in events:
        e, elz = pgc.getel_2(x, csd, redir_arts_map, rdarts_rev, eventsdf)
        elarts = set(elz['prev']) | set(elz['curr'])
        if len(elarts-rrk) > 0:
            errors.append(e)
            raise

    del csd

# %%

date = pd.to_datetime(e[:8])
start = date-datetime.timedelta(days=30)
stop = date+datetime.timedelta(days=30)

months = months_range(pd.to_datetime(start), pd.to_datetime(stop))
mr = {'n_%s-%s' % (x[:4], x[4:]): monthrange(int(x[:4]), int(x[4:]))[1]
      for x in months}


el2 = pgc.get_neighbours_quick(x, csd, redir_arts_map, eventsdf)
el2a = pgc.get_neighbours_quick_2(
    eventsdf.loc[x, 'Articles'], csd, redir_arts_map, eventsdf)
el2c = pgc.get_neighbours_quick_2(
    eventsdf.loc[x, 'Articles'], csd, crd, eventsdf)


ndfarticles = el2a
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
el = el[el['n'] > 100]


# %%

mset = {y for x in montharticles.values() for y in x}
aset = allneighbours_set

asetmm = {megamap.get(x.replace('_', ' '), x).replace(' ', '_') for x in aset}
asetrd = {rdarts_rev.get(x, x) for x in aset}
# %%

rd_arts_map_topup = pgc.get_redirects({megamap.get(x.replace('_', ' '),
                                                   x.replace('_', ' '))
                                       for x in aset - set(rdarts_rev2.keys())})
# %%

with open('support_data/megamap3a.json', 'w+') as f:
    json.dump(megamap, f)

with open('support_data/redir_arts_map3a.json', 'w+') as f:
    json.dump(rd_arts_map2, f)

    # %%

adjs = glob.glob('/Volumes/PGPassport/DPhil redo data/events/*/adjNN.h5')
# %%
redo '/Volumes/PGPassport/DPhil redo data/events/20180201_United States Secretary of State--Rex Tillerson--Mexico--President of Mexico--Enrique Peña Nieto--Luis Videgaray Caso_CS/adjNN.h5'

# %%
for n, a in enumerate(adjs):
    if n < 1248:
        continue
    print(n)
    adj = pd.read_hdf(a)
    adj = (adj | adj.T).astype(int)
    adj.to_hdf(a, key='df')


# %%
ats = []
for n, a in enumerate(adjs):
    if n < 2611:
        continue
    print(n)
    adj = pd.read_hdf(a)
    ts = pd.read_hdf(a.replace('adjNN', 'tsNN'))
    if set(adj.columns)-set(ts.columns):
        ats.append(a)
        print('error', a)
    elif set(ts.columns)-set(adj.columns):
        print('subset')
        ts[adj.columns].to_hdf(a.replace('adjNN', 'tsNN'), key='df')

# %%


adj = np.array([[0, 1, 1],
                [1, 0, 0],
                [1, 0, 0]])


tse = np.array([[1, 2, 5, 3, 2],
                [1, 3, 4, 2, 2],
                [3, 1, 0, 1, 2]])
# %%

t0 = time.time()
pcs = rolling_pearson_ixs(tsxv, adj)
print(time.time()-t0)

t0 = time.time()
pcs = np.triu(adj.values)*np.array([pgc.scoreretlist2(x, tsxv)
                                    for x in range(0, len(tsxv)-6)])
print(time.time()-t0)

# %%

# %%
core = [rdarts_rev.get(x.replace(' ', '_'), x.replace(' ', '_')) for x in
        pd.read_csv(BPATH+'events/'+e+'/core.tsv', sep='\t', header=None)[0]]
articles = pd.read_hdf(BPATH+'events/' + e + '/coltitlesNN.h5', key='df')
glist, igname = read_t_graph(BPATH+'events/'+e)
membdf = temporal_community_detection(glist, 0.1)
evrs = extract_event_reactions(membdf, igname, core, articles)
tcd = community_centralities(glist, igname, membdf, evrs)


def read_ev_data(e):
    core = [rdarts_rev.get(x.replace(' ', '_'), x.replace(' ', '_')) for x in
            pd.read_csv(BPATH+'events/'+e+'/core.tsv', sep='\t', header=None)[0]]
    articles = pd.read_hdf(BPATH+'events/' + e + '/coltitlesNN.h5', key='df')
    glist, igname = read_t_graph(BPATH+'events/'+e)
    return core, articles, glist, igname


def event_reactions(core, articles, glist, igname, res):
    membdf = temporal_community_detection(glist, res)
    evrs = extract_event_reactions(membdf, igname, core, articles)
    return membdf, evrs


def event_reactions_tcds(core, articles, glist, igname, res):
    membdf = temporal_community_detection(glist, res)
    evrs = extract_event_reactions(membdf, igname, core, articles)
    tcd = community_centralities(glist, igname, membdf, evrs)
    return membdf, evrs, tcd


# %%
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
    t = membdf.T.loc[beg:end][membdf.T.loc[beg:end] == k].T.dropna(how='all')
    gg = {x: set(t[x].dropna().index) for x in t}
    js = pd.Series({k: jac(set(gg[mid]), set(v))
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
# %%

ready = set([x.split('/')[-2] for x in
             glob.glob(BPATH + 'events/*/simtgraphixNN.npz')])
finished = set([x.split('/')[-2] for x in
                glob.glob(BPATH+'events/*/tcweights.h5')])
copied = set([x.split('/')[-2] for x in
              glob.glob(BPATH+'events2/*/coltitlesNN.h5')])
err = []
for n, e in enumerate((ready-finished)-copied):
    print(n/len((ready-finished)-copied))
    try:
        try:
            os.mkdir(BPATH+'events2/'+e)
        except FileExistsError:
            print('folder exists')
        for f in ['/core.tsv', '/simtgraphelNN.npz', '/simtgraphixNN.npz', '/coltitlesNN.h5']:
            shutil.copy(BPATH+'events/'+e+f, BPATH+'events2/'+e+f)
    except Exception as ex:
        print(ex)
        err.append((n, e, ex))
# %%

comp_folders = glob.glob('/Users/Patrick/Downloads/ev out/*')
dupes = []
cerror = []
for n, f in enumerate(comp_folders):
    print(n/len(comp_folders))
    dest = f.replace('/Users/Patrick/Downloads/ev out/',
                     '/Volumes/PGPassport/DPhil redo data/events/')
    if any([dest+x in glob.glob(dest+'/*') for x in ['/evrs.h5', '/membdf.h5',
                                                     '/tcweights.h5']]):
        dupes.append(f)
    else:
        for x in ['/evrs.h5', '/membdf.h5', '/tcweights.h5']:
            try:
                shutil.copy(f+x, dest+x)
            except:
                print(f+x)
                cerror.append(f+x)
# %%
comp_folders = glob.glob('/Users/Patrick/Downloads/ev out/*')
e2f = glob.glob('/Volumes/PGPassport/DPhil redo data/events2/*')
e2c = [f.replace('/Users/Patrick/Downloads/ev out/',
                 '/Volumes/PGPassport/DPhil redo data/events2/') for f in comp_folders]
# %%

namedict = {n: x for n, x in enumerate(sorted(set(e2f)-set(e2c)))}
with open('support_data/rem_folders.json', 'w+') as f:
    json.dump(namedict, f)
# %%
namedictrev = {v: k for k, v in namedict.items()}

for k, v in namedictrev.items():
    os.rename(k, '/'.join(k.split('/')[:-1])+'/'+str(v))


# %%

nf = glob.glob('/Users/Patrick/Downloads/events_out_N/*')

dupes = []
cerror = []
for n, f in enumerate(nf):
    print(n/len(nf))
    dest = namedict[f.split('/')[-1]].replace('/events2/', '/events/')

    if any([dest+x in glob.glob(dest+'/*') for x in ['/evrs.h5', '/membdf.h5',
                                                     '/tcweights.h5']]):
        dupes.append(f)
    else:
        for x in ['/evrs.h5', '/membdf.h5', '/tcweights.h5']:
            try:
                shutil.copy(f+x, dest+x)
            except:
                print(f+x)
                cerror.append(f+x)
# %%

eventsdfs = sorted([unicodedata.normalize('NFD', x) for x in
                    pd.read_hdf('support_data/eventsdf2.h5', key='df').index.str.replace('/', ':')])
edfs3 = sorted(pd.read_hdf('support_data/eventsdf3.h5',
               key='df').index.str.replace('/', ':'))
folders = sorted([x.split('/')[-1] for x in glob.glob(BPATH + 'events/*')])
evlsns = sorted([x for x in
                 pd.read_hdf('support_data/evlsN2.h5', key='df').index])
simtgraphels = sorted([x.split('/')[-2]
                      for x in glob.glob(BPATH + 'events/*/simtgraphelNN.npz')])
membdfs = sorted([x.split('/')[-2]
                 for x in glob.glob(BPATH + 'events/*/membdf.h5')])
tcweights = sorted([x.split('/')[-2]
                   for x in glob.glob(BPATH + 'events/*/tcweights.h5')])
wjacs = sorted([x.split('/')[-2]
               for x in glob.glob(BPATH + 'events/*/wjac.h5')])

# %%

df_no_fol = sorted(set(edfs3) - set(evlsns))

fol_no_stg = sorted(set(folders)-set(simtgraphels)
                    )  # no nodes at filter stage 1
stg_no_mdf = sorted(set(simtgraphels)-set(membdfs))
membdf_no_tcw = sorted(set(membdfs)-set(tcweights))  # all empty graphs


# %%


edf = evlsn.loc[membdf_no_tcw].sort_values('len')

print(len(edf))
res = 0.25
try:

    out1 = [pgc.read_ev_data(BPATH + 'events/' + x, rdarts_rev)
            for x in edf.index]
    out2 = [pgc.ev_reactions_tcd(*x, res) for x in out1 if len(x) == 4]

    names = [BPATH+'events/'+x for m, x in
             enumerate(edf.index) if len(out1[m]) == 4]
    for m, i in enumerate(out2):
        try:
            i[0].to_hdf('%s/membdf.h5' % names[m], key='df')
            for k, v in i[1].items():
                v.to_hdf('%s/evrs.h5' % names[m], key=k)
            for k, v in i[2].items():
                v.to_hdf('%s/tcweights.h5' % names[m], key=k)
        except Exception as ex:
            errors.append((n, m, ex))
            print(ex)
except Exception as ex:
    errors.append((n, ex))
    print(ex)

# %%

with open('support_data/rem_folders.json') as json_data:
    remfold = json.load(json_data)
    json_data.close()
# %%

rfv = sorted([x.split('/')[-1] for x in remfold.values()])

# %%
ldists = pd.DataFrame(index=sorted(set(evlsns) - set(eventsdfs)),
                      columns=sorted(set(eventsdfs) - set(evlsns)))

for i in sorted(set(evlsns) - set(eventsdfs)):
    for j in sorted(set(eventsdfs) - set(evlsns)):
        ldists.loc[i, j] = levenshtein_distance(i, j)/max(len(i), len(j))
# %%

ddl = len(DD)
ddcl = len(DDC)
pil = len(pi)
pipl = len(pip)
Gl = len(G.vs)
wjsl = len(wjs)

# %%

ddc_keys = {x.replace(BPATH+'events/', '') for x in DDC.keys()}
wjs_keys = set(wjs.keys())
# %%
for f in sorted(ddc_keys-wjs_keys)[1:]:
    os.remove(BPATH + 'events/' + f.split('_CS')[0]+'_CS' + '/wjac.h5')
    os.remove(BPATH + 'events/' + f.split('_CS')[0]+'_CS' + '/jac.h5')
# %%

jll2 = []
for n in range(2000000, 11000000, int(1E6)):
    if n % 10000000 == 0:
        print('%.2f %%' % (100*n/11000000))
    with open(BPATH + 'evr_similarities/jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll2.extend(json.load(f))

# %%

jll0 = []
for n in range(2000000, 11000000, int(1E6)):
    if n % 10000000 == 0:
        print('%.2f %%' % (100*n/11000000))
    with open(BPATH + 'jacsw%.1f.json' % (n/1E6), 'r') as f:
        jll0.extend(json.load(f))

# %%

(np.array(jll0)-np.array(jll2)).max()
# %%
o0 = []
o2 = []
for n, x in enumerate(DD):
    if n % 500 == 0:
        print('%.2f %%' % (100*n/len(DD)))
    with pd.HDFStore(x+'/tcweights.h5', mode='r') as hdf:
        for k in hdf.keys():
            o0.append(x + k)
            o2.append(x + '/' + k[1:].replace('/', ':'))

# %%
so0 = sorted(o0)
so2 = sorted(o2)
[(so0[n], so2[n]) for n in range(len(o0))
 if so0[n].replace(':', '/') != so2[n].replace(':', '/')]

# %%
n1 = pd.read_hdf('support_data/topic_labels_1.h5')
n2 = pd.read_hdf('support_data/topic_labels_2.h5')

# %%


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
    ou = simdf[simdf['outlabel'].str.replace('*', '').str.replace('^', '')
               == x]['SS_score']
    lq, mm, uq = simdf['SS_score'].quantile([0.25, 0.5, 0.75])
    if len(ou) != 1:
        print(x)
        raise
    if ou.iloc[0] > uq:
        return '\\cellcolor{green}' + x.replace('&', '\\&')
    elif ou.iloc[0] > mm:
        return '\\cellcolor{yellow}' + x.replace('&', '\\&')
    elif ou.iloc[0] > lq:
        return '\\cellcolor{orange}' + x.replace('&', '\\&')
    else:
        return '\\cellcolor{red}' + x.replace('&', '\\&')
