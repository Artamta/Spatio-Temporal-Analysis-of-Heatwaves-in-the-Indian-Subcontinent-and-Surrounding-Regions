### Imports ###
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import deepgraph as dg
import plotting as plot
import con_sep as cs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-u", "--upgma_clusters", help="Give the number of upgma clusters",
                        type=int)
    parser.add_argument("-i", "--family", help="Give the number of the family",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
no_clusters = args.upgma_clusters
i = args.family
gv['time']=pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)
# create supernodes from deep graph by partitioning the nodes by cp
# feature functions applied to the supernodes
feature_funcs = {'time': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean], 
                 't2m': [np.max],'ytime':[np.mean],}
# partition graph
cpv, ggv = g.partition_nodes('cp', feature_funcs, return_gv=True)
# append neccessary stuff
# append geographical id sets
cpv['g_ids'] = ggv['g_id'].apply(set)
# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
# append time spans
cpv['timespan'] = cpv.dt.dt.days+1
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

gvv = dg.DeepGraph(gv)
cpg = dg.DeepGraph(cpv)

# create edges
cpg.create_edges(connectors=[cs.cp_node_intersection, 
                             cs.cp_intersection_strength],
                             no_transfer_rs=['intsec_card'],
                             logfile='create_cpe',
                             step_size=1e7)

# create condensed distance matrix
dv = 1 - cpg.e.intsec_strength.values
del cpg.e

# create linkage matrix
lm = linkage(dv, method='average', metric='euclidean')
del dv
    
# form flat clusters and append their labels to cpv
cpv['F_upgma'] = fcluster(lm, no_clusters, criterion='maxclust')
# relabel families by size
f = cpv['F_upgma'].value_counts().index.values
fdic = {j: i for i, j in enumerate(f)}
cpv['F_upgma'] = cpv['F_upgma'].apply(lambda x: fdic[x])
# create F col
gv['F_upgma'] = np.ones(len(gv), dtype=int) * -1
gcpv = cpv.groupby('F_upgma')
it = gcpv.apply(lambda x: x.index.values)

for F in range(len(it)):
    cp_index = gvv.v.cp.isin(it.iloc[F])
    gvv.v.loc[cp_index, 'F_upgma'] = F
    
# feature funcs
def n_cp_nodes(cp):
  return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}

# create family-g_id intersection graph
fgv = gvv.partition_nodes(['F_upgma', 'g_id'], feature_funcs=feature_funcs)
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin':'longitude','latitude_amin':'latitude'}, inplace=True)
# save updated cpv and gv table
cpv.to_csv(path_or_buf = "/Users/ayush/Desktop/Final_Report/results/cpv_fam%s.csv" % i, index=False)
gv.to_csv(path_or_buf = "/Users/ayush/Desktop/Final_Report/clustering2/gv_fam%s.csv" % i, index=False)

plot.plot_families(no_clusters,fgv,gv,'Family_%s' % i)
# uncomment if you want to plot the number of hits and not only the number of heat waves at every grid cell
plot.plot_hits(no_clusters,fgv,gv,'Family_%s' % i)