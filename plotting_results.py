# takes net_cdf file and converts it into a pandas dataframe with xarray
# creates integer-based coordinates
# saves pandas dataframe under Results

# data i/o
import argparse
import numpy as np
import pandas as pd
import deepgraph as dg
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

### Argparser ###
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the dataset.", type=str)
    parser.add_argument("-n", "--number", help="Heat wave number to be plotted.", type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
no = args.number
gv['time'] = pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)

# Create supernodes from deep graph by partitioning the nodes by cp
feature_funcs = {'time': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean],
                 't2m': [np.max], 'ytime': [np.mean]}
cpv, ggv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# Append necessary columns
cpv['g_ids'] = ggv['g_id'].apply(set)
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
cpv['timespan'] = cpv.dt.dt.days + 1
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

# Plot largest heat wave using pcolormesh
first = gv[gv.cp == no]
first_gv = dg.DeepGraph(first)

# Feature functions
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

# Create g_id intersection graph
fgv = first_gv.partition_nodes('g_id', feature_funcs=feature_funcs)
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin': 'longitude', 'latitude_amin': 'latitude'}, inplace=True)

# Configure map projection
kwds_basemap = {'llcrnrlon': g.v.longitude.min() - 1,
                'urcrnrlon': g.v.longitude.max() + 1,
                'llcrnrlat': g.v.latitude.min() - 1,
                'urcrnrlat': g.v.latitude.max() + 1}

# Create grid for pcolormesh
lon = fgv['longitude'].values
lat = fgv['latitude'].values
data = fgv['n_cp_nodes'].values
lon_grid, lat_grid = np.meshgrid(np.unique(lon), np.unique(lat))
data_grid = np.full((len(np.unique(lat)), len(np.unique(lon))), 0.0)

for _, row in fgv.iterrows():
    lat_idx = np.where(np.unique(lat) == row['latitude'])[0][0]
    lon_idx = np.where(np.unique(lon) == row['longitude'])[0][0]
    data_grid[lat_idx, lon_idx] = row['n_cp_nodes']

# Fill blank regions with zeroes
data_grid = np.nan_to_num(data_grid)

# Plot using pcolormesh
fig, ax = plt.subplots(figsize=(12, 10))
m = Basemap(projection='cyl', **kwds_basemap, resolution='l', ax=ax)
m.drawcoastlines(linewidth=0.8, color='black')
m.drawparallels(np.arange(lat.min(), lat.max(), 10), labels=[1, 0, 0, 0], linewidth=0.2, color='gray')
m.drawmeridians(np.arange(lon.min(), lon.max(), 10), labels=[0, 0, 0, 1], linewidth=0.2, color='gray')

pcm = m.pcolormesh(lon_grid, lat_grid, data_grid, cmap='viridis_r', shading='auto', latlon=True)
cbar = m.colorbar(pcm, location='right', pad="5%")
cbar.set_label('Number of Heat Wave Days', fontsize=15)
ax.set_title(f'{no}. Largest Heat Wave', fontsize=18)

# Save the plot
fig.savefig('/Users/ayush/Desktop/Final_Report/some_otherplots__/largest_heatwave_pcolormesh.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot progression of largest heat wave using pcolormesh
times = np.arange(first.itime.min(), first.itime.max() + 1)
tdic = {time: itime for itime, time in enumerate(times)}
first['dai'] = first.itime.apply(lambda x: tdic[x])
first['dai'] = first['dai'].astype(np.uint16)

# Create a consistent grid for the entire region
lon = first['longitude'].values
lat = first['latitude'].values
data = first['dai'].values

# Define the grid resolution
lon_min, lon_max = lon.min() - 1, lon.max() + 1
lat_min, lat_max = lat.min() - 1, lat.max() + 1
lon_grid, lat_grid = np.meshgrid(
    np.linspace(lon_min, lon_max, 200),  # 200 points for better resolution
    np.linspace(lat_min, lat_max, 200)
)

# Interpolate data onto the grid
data_grid = np.full(lon_grid.shape, 0.0)  # Initialize with zeroes
for _, row in first.iterrows():
    lat_idx = np.abs(lat_grid[:, 0] - row['latitude']).argmin()
    lon_idx = np.abs(lon_grid[0, :] - row['longitude']).argmin()
    data_grid[lat_idx, lon_idx] = row['dai']

# Plot progression using pcolormesh
fig, ax = plt.subplots(figsize=(12, 10))
m = Basemap(projection='cyl', llcrnrlon=lon_min, urcrnrlon=lon_max,
            llcrnrlat=lat_min, urcrnrlat=lat_max, resolution='l', ax=ax)
m.drawcoastlines(linewidth=0.8, color='black')
m.drawparallels(np.arange(lat_min, lat_max, 10), labels=[1, 0, 0, 0], linewidth=0.2, color='gray')
m.drawmeridians(np.arange(lon_min, lon_max, 10), labels=[0, 0, 0, 1], linewidth=0.2, color='gray')

pcm = m.pcolormesh(lon_grid, lat_grid, data_grid, cmap='rainbow', shading='auto', latlon=True)
cbar = m.colorbar(pcm, location='right', pad="5%")
cbar.set_label('Days After Initiation', fontsize=15)
ax.set_title('Progression of Largest Heat Wave', fontsize=18)

# Save the plot
fig.savefig('/Users/ayush/Desktop/Final_Report/some_otherplots__/largest_heatwave_progression_pcolormesh.png', dpi=300, bbox_inches='tight')
plt.close()