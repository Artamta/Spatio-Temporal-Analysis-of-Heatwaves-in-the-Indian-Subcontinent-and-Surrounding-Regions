import argparse
import numpy as np
import pandas as pd
import deepgraph as dg
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the nodes dataset to be worked on.", type=str)
    parser.add_argument("-cpv", "--cpv", help="Give the path to the supernodes tables", type=str)
    parser.add_argument("-n", "--number", help="Give the number of heatwaves to be plotted", type=int)
    parser.add_argument("-b", "--by", help="Give the column name by which the heatwaves should be sorted", type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
cpv = pd.read_csv(args.cpv)
gv = pd.read_csv(args.data)
n = args.number
b = args.by

cpv.sort_values(by=b, inplace=True, ascending=False)

for i in range(1, n + 1):
    cp = cpv.cp.iloc[i - 1]
    ggg = dg.DeepGraph(gv[gv.cp == cp])
    start = cpv.time_amin.iloc[i - 1]
    end = cpv.time_amax.iloc[i - 1]
    duration = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    magnitude = cpv.HWMId_magnitude.iloc[i - 1]

    def n_cp_nodes(cp):
        return len(cp.unique())

    feature_funcs = {'magnitude': [np.sum],
                     'latitude': np.min,
                     'longitude': np.min,
                     'cp': n_cp_nodes}
    fgv = ggg.partition_nodes(['g_id'], feature_funcs=feature_funcs)
    fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin': 'longitude', 'latitude_amin': 'latitude'}, inplace=True)

    gt = dg.DeepGraph(fgv)

    fig, ax = plt.subplots(figsize=(16, 12))

    unique_latitudes = np.sort(gt.v.latitude.unique())
    unique_longitudes = np.sort(gt.v.longitude.unique())

    grid_data = pd.DataFrame({
        'latitude': gt.v.latitude,
        'longitude': gt.v.longitude,
        'magnitude_sum': gt.v.magnitude_sum
    })

    grid_data = grid_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['latitude', 'longitude'])

    print(f"Longitude bounds: {unique_longitudes.min()} to {unique_longitudes.max()}")
    print(f"Latitude bounds: {unique_latitudes.min()} to {unique_latitudes.max()}")

    pivot_table = grid_data.pivot_table(
        index='latitude', columns='longitude', values='magnitude_sum', fill_value=0
    )

    pivot_table = pivot_table.reindex(index=unique_latitudes, columns=unique_longitudes, fill_value=0)

    # Add a buffer to the bounds
    buffer = 1
    llcrnrlon = unique_longitudes.min() - buffer
    urcrnrlon = unique_longitudes.max() + buffer
    llcrnrlat = unique_latitudes.min() - buffer
    urcrnrlat = unique_latitudes.max() + buffer

    # Ensure bounds are valid
    if np.isnan(llcrnrlon) or np.isnan(urcrnrlon) or np.isnan(llcrnrlat) or np.isnan(urcrnrlat):
        raise ValueError("Latitude or Longitude bounds contain NaN values.")

    m = Basemap(
        llcrnrlon=llcrnrlon,
        urcrnrlon=urcrnrlon,
        llcrnrlat=llcrnrlat,
        urcrnrlat=urcrnrlat,
        projection='merc',
        resolution='i',
        ax=ax
    )

    m.drawcoastlines(linewidth=0.8, color='black', zorder=10)
    m.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0], linewidth=0.5, color='gray', fontsize=12)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], linewidth=0.5, color='gray', fontsize=12)

    lon, lat = np.meshgrid(pivot_table.columns.values, pivot_table.index.values)
    x, y = m(lon, lat)
    sc = m.pcolormesh(x, y, pivot_table.values, cmap='viridis', shading='nearest')

    cbar = m.colorbar(sc, location='right', pad=0.02)
    cbar.set_label('Magnitude (Sum)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # filepath: /Users/ayush/Desktop/Final_Report/code/plot_heatwaves.py
    ax.set_title(f'Global Heat Wave {i}: {start} to {end}', fontsize=20, fontweight='bold', pad=20)

    plt.savefig(f'/Users/ayush/Desktop/Final_Report/plots/HWMID_global_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()