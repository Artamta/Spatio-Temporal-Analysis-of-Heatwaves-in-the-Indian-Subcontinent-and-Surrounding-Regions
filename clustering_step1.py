### Imports ###
import argparse
import numpy as np
import pandas as pd
import deepgraph as dg
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import con_sep as cs

### Functions ###

def conv_sin(doy):
    sin_doy = np.sin((doy / 365) * 2 * np.pi)
    return sin_doy

def conv_cos(doy):
    cos_doy = np.cos((doy * 2 * np.pi / 365))
    return cos_doy

# plots heat wave families or clusters on a map using pcolormesh and Basemap
def plot_families(number_families, fgv, v, plot_title):
    families = np.arange(number_families)
    for F in families:
        print(f"\nProcessing Family {F}")
        family_data = fgv[fgv['F_kmeans'] == F]

        if family_data.empty:
            print(f"No data found for Family {F}. Skipping...")
            continue

        # Debugging: Check family data
        print(f"Family {F} data:\n{family_data}")

        # Extract data for plotting
        lon = family_data['longitude'].values
        lat = family_data['latitude'].values
        data = family_data['n_cp_nodes'].values

        # Debugging: Check extracted data
        print(f"Family {F} Longitude range: {lon.min()} to {lon.max()}")
        print(f"Family {F} Latitude range: {lat.min()} to {lat.max()}")
        print(f"Family {F} Data values: {data}")

        # Aggregate data to ensure grid consistency
        grid_data = pd.DataFrame({'latitude': lat, 'longitude': lon, 'data': data})
        aggregated_data = grid_data.groupby(['latitude', 'longitude']).mean().reset_index()

        # Debugging: Check aggregated data
        print(f"Aggregated data for Family {F}:\n{aggregated_data}")

        # Extract the aggregated values
        lat = aggregated_data['latitude'].values
        lon = aggregated_data['longitude'].values
        data = aggregated_data['data'].values

        # Create the grid
        lon_grid, lat_grid = np.meshgrid(np.unique(lon), np.unique(lat))

        # Fill missing grid points with zeros
        data_grid = np.full((len(np.unique(lat)), len(np.unique(lon))), 0.0)
        for _, row in aggregated_data.iterrows():
            lat_idx = np.where(np.unique(lat) == row['latitude'])[0][0]
            lon_idx = np.where(np.unique(lon) == row['longitude'])[0][0]
            data_grid[lat_idx, lon_idx] = row['data']

        # Debugging: Check data grid
        print(f"Data grid for Family {F}:\n{data_grid}")
        print(f"Number of NaN values in data grid: {np.isnan(data_grid).sum()}")

        # Configure map projection using Basemap
        fig, ax = plt.subplots(figsize=(12, 10))
        m = Basemap(projection='cyl', llcrnrlat=lat.min() - 1, urcrnrlat=lat.max() + 1,
                    llcrnrlon=lon.min() - 1, urcrnrlon=lon.max() + 1, resolution='l', ax=ax)

        # Draw coastlines and gridlines
        m.drawcoastlines(linewidth=0.8, color='black')  # Coastlines only
        m.drawparallels(np.arange(lat.min(), lat.max(), 10), labels=[1, 0, 0, 0], linewidth=0.2, color='gray')
        m.drawmeridians(np.arange(lon.min(), lon.max(), 10), labels=[0, 0, 0, 1], linewidth=0.2, color='gray')

        # Plot data using pcolormesh with reversed color scheme
        pcm = m.pcolormesh(lon_grid, lat_grid, data_grid, cmap='viridis_r', shading='auto', latlon=True)

        # Add colorbar
        cbar = m.colorbar(pcm, location='right', pad="5%")
        cbar.set_label('Number of Heatwaves', fontsize=14)

        # Add title
        ax.set_title(f'Family {F}', fontsize=18)

        # Save the plot
        save_path = f'/Users/ayush/Desktop/Final_Report/clustering_Pclr/{plot_title}_Cluster_{F}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close(fig)

### Argparser ###

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the original dataset to be worked on.",
                        type=str)
    parser.add_argument("-k", "--cluster_number", help="Give the number of clusters for the k-means clustering",
                        type=int)
    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
k = args.cluster_number
gv['time'] = pd.to_datetime(gv['time'])
g = dg.DeepGraph(gv)

# create supernodes from deep graph by partitioning the nodes by cp
# feature functions applied to the supernodes
feature_funcs = {'time': [np.min, np.max],
                 't2m': [np.mean],
                 'magnitude': [np.sum],
                 'latitude': [np.mean],
                 'longitude': [np.mean],
                 't2m': [np.max], 'ytime': [np.mean]}
# partition graph
cpv, ggv = g.partition_nodes('cp', feature_funcs, return_gv=True)

# append necessary stuff
# append geographical id sets
cpv['g_ids'] = ggv['g_id'].apply(set)
# append cardinality of g_id sets
cpv['n_unique_g_ids'] = cpv['g_ids'].apply(len)
# append time spans
cpv['dt'] = cpv['time_amax'] - cpv['time_amin']
cpv['timespan'] = cpv.dt.dt.days + 1
# rename magnitude_sum column
cpv.rename(columns={'magnitude_sum': 'HWMId_magnitude'}, inplace=True)

# transform day of year value
cpv['doy_cos'] = cpv.ytime_mean.apply(conv_cos)
cpv['doy_sin'] = cpv.ytime_mean.apply(conv_sin)

# perform k means clustering
clusterer = KMeans(n_clusters=k, random_state=100)
cluster_labels = clusterer.fit_predict(cpv[['doy_sin', 'doy_cos']])
cpv['kmeans_clust'] = cluster_labels

# plot the k means clustering
fig, ax = plt.subplots()
fig.set_size_inches(18, 7)
colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
xs = cpv['doy_cos']
ys = cpv['doy_sin']
ax.scatter(xs, ys, marker=".", s=50, lw=0, alpha=0.7, c=colors, edgecolor="k")
ax.set_title('k=%s' % k)
ax.set_xlabel('doy_cos')
ax.set_ylabel('doy_sin')
fig.savefig('/Users/ayush/Desktop/Final_Report/clustering_Pclr/k-means_clustering')

# create F_kmeans col
gv['F_kmeans'] = np.ones(len(gv), dtype=int) * -1
gcpv = cpv.groupby('kmeans_clust')
it = gcpv.apply(lambda x: x.index.values)

for F in range(len(it)):
    cp_index = g.v.cp.isin(it.iloc[F])
    g.v.loc[cp_index, 'F_kmeans'] = F

# plot the day of year distribution of the clusters
for f in range(k):
    tmp = dg.DeepGraph(g.v)
    tmp.filter_by_values_v('F_kmeans', f)
    plt.hist(tmp.v.ytime, bins=175, label=f'Family %s' % f, alpha=0.5)
    plt.title("Day of Year Distribution of the Heat Wave Families")
    plt.xlabel('Day of year')
    plt.ylabel('Occurrences')
    plt.legend()
plt.savefig('/Users/ayush/Desktop/Final_Report/clustering_Pclr/day_of_year_distribution')

# plot the families on a map
# feature funcs
def n_cp_nodes(cp):
    return len(cp.unique())

feature_funcs = {'magnitude': [np.sum],
                 'latitude': np.min,
                 'longitude': np.min,
                 'cp': n_cp_nodes}

fgv = g.partition_nodes(['F_kmeans', 'g_id'], feature_funcs=feature_funcs)

# Rename columns for clarity
fgv.rename(columns={'cp_n_cp_nodes': 'n_cp_nodes', 'longitude_amin': 'longitude', 'latitude_amin': 'latitude'}, inplace=True)

# Ensure 'F_kmeans' is added to fgv
fgv['F_kmeans'] = fgv.index.get_level_values('F_kmeans')

# Debugging: Check if 'F_kmeans' exists in fgv
print("Columns in fgv:", fgv.columns)
print("First few rows of fgv:\n", fgv.head())

# Call the updated plot_families function
plot_families(k, fgv, gv, 'Family')