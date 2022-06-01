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
