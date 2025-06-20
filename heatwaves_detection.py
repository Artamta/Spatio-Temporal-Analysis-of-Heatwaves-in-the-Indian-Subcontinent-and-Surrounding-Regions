# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer based coordinates
# calculates threshold for days of extreme heat
# computes days of extreme heat and heat waves
# saves pandas dataframe under Results

### Imports ###
import xarray
import argparse
# the usual
import numpy as np
import pandas as pd
import extr as ex
import matplotlib
import deepgraph as dg
import cppv
import gc

### Functions ###
def perc25(a_list):
    threshold = np.percentile(a_list, 25)
    return threshold

def perc75(a_list):
    threshold = np.percentile(a_list, 75)
    return threshold

# filepath: /Users/ayush/Desktop/Final_Report/code/heatwaves_detection.py
def calc_mag(data):
    if data.t2m > data.t2m_amax_perc25:
        mag = (data.t2m - data.t2m_amax_perc25) / (data.t2m_amax_perc75 - data.t2m_amax_perc25)
    else:
        mag = 0
    return mag

# calculate the 95th percentile of a list
def calc_percentile(a_list):
    threshold = np.percentile(a_list, 95)
    return threshold

# creates the list the 95th percentile should be calculated on and then calls calc_percentile
def calc_perc(lst):
    a_list = []
    for l in lst:
        for i in range(len(l)):
            a_list.append(l[i])
    return calc_percentile(a_list)

# convert temperature from kelvin to degrees celcius
def conv_to_degreescelcius(data):
    data.t2m = data.t2m - 273.15  

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-do", "--original_data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-g", "--g_ids", help="Give the number how many unique g_ids a heatwave needs to be considered a heatwave",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
d = xarray.open_dataset(args.original_data)
b = args.g_ids
print('read data')

# create integer based (x,y) coordinates
d['x'] = (('longitude'), np.arange(len(d.longitude)))
d['y'] = (('latitude'), np.arange(len(d.latitude)))
print('created integer based coordinates')

# convert to dataframe
vt = d.to_dataframe()
print(vt)

# reset index
vt.reset_index(inplace=True)
print('reset index')
del d
gc.collect()

# append dayofyear 
vt['ytime'] = vt.time.apply(lambda x: x.dayofyear)
print('ytime')

# convert temperature from kelvin to degrees celcius
ex.conv_to_degreescelcius(vt)
print('degrees')

first = np.arange(350, 366)
second = np.arange(1, 366)
third = np.arange(1, 16)
time = np.concatenate((first, second, third), axis=None)
g_t = dg.DeepGraph(vt)

# remove 366th day
ytime = np.arange(366)
g_t.filter_by_values_v('ytime', ytime)
print('366') 

# calculate threshold
# partition the node table
cpv_t, gv_t = g_t.partition_nodes(['x', 'y', 'ytime'], return_gv=True)
cpv_t['t2m'] = gv_t['t2m'].apply(list)
cpv_t.reset_index(inplace=True)
tmp2 = pd.DataFrame(columns=['x', 'y', 'ytime', 'thresh'])
for i in range(366):
    g = dg.DeepGraph(cpv_t)
    k = time[i:i+31]
    g.filter_by_values_v('ytime', k)
    tmp, tmp_p = g.partition_nodes(['x', 'y'], return_gv=True)
    tmp['t2m'] = tmp_p['t2m'].apply(list)
    tmp.reset_index(inplace=True)
    tmp['thresh'] = tmp['t2m'].apply(lambda x: np.percentile(x, 99))
    tmp.drop(['t2m'], axis=1, inplace=True)
    tmp['ytime'] = i + 1
    tmp2 = pd.concat([tmp2, tmp])
result = pd.merge(vt, tmp2, on=["ytime", "x", 'y'])
result.drop(columns=['n_nodes'], inplace=True)

# append some necessary stuff to the dataset
# append a column indicating geographical locations (i.e., supernode labels)
result['g_id'] = result.groupby(['longitude', 'latitude']).grouper.group_info[0]
result['g_id'] = result['g_id'].astype(np.uint32)  

# save threshold dataset
result.to_csv(path_or_buf="/Users/ayush/Desktop/Final_Report/results/thresh_new.csv", index=False)

# calculate extreme dataset
# filepath: /Users/ayush/Desktop/Final_Report/code/heatwaves_detection.py
result["keep"] = np.where(result["t2m"] >= result["thresh"], True, False)
extr = result.loc[result['keep'] == True]
extr.drop(columns=['keep'], inplace=True) 

# append integer-based time
times = pd.date_range(extr.time.min(), extr.time.max(), freq='D')
tdic = {time: itime for itime, time in enumerate(times)}
extr['itime'] = extr.time.apply(lambda x: tdic[x])
extr['itime'] = extr['itime'].astype(np.uint16)

# sort by time
extr.sort_values('time', inplace=True)

# assign your new columns
datetimes = pd.to_datetime(vt['time'])
vt['day'] = datetimes.dt.day
vt['month'] = datetimes.dt.month
vt['year'] = datetimes.dt.year

# calculate daily magnitude of extreme events
f_funcs = {'t2m': [np.max]}
gg = dg.DeepGraph(vt)

del vt
gc.collect()

gg_t = gg.partition_nodes(['x', 'y', 'year'], f_funcs)
gg_t.reset_index(inplace=True)
feature_funcs = {'t2m_amax': [perc75, perc25]}
ggt = dg.DeepGraph(gg_t)
ggg = ggt.partition_nodes(['x', 'y'], feature_funcs)
rex = pd.merge(extr, ggg, on=['x', 'y'])

del extr
gc.collect()

rex.drop(columns=['n_nodes'], inplace=True)
rex['magnitude'] = rex.apply(calc_mag, axis=1)
rex.drop(columns=['t2m_amax_perc25', 't2m_amax_perc75', 'thresh'], inplace=True)

# save the extreme dataset
rex.to_csv(path_or_buf="/Users/ayush/Desktop/Final_Report/results/extr_new.csv", index=False)

# create heatwaves from the extreme dataset
rex.sort_values('time', inplace=True)
g, cpg, cpv = cppv.create_cpv(rex, b)

# save heatwaves
cpv.to_csv(path_or_buf=f"/Users/ayush/Desktop/Final_Report/results/cpv_new{b}.csv", index=False)
g.v.to_csv(path_or_buf=f"/Users/ayush/Desktop/Final_Report/results/gv_new{b}.csv", index=False)