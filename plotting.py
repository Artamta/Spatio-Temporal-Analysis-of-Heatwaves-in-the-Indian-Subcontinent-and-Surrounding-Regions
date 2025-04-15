# This module contains all functions for different plotting styles of the results

### Imports ###
import matplotlib.pyplot as plt
import numpy as np
import deepgraph as dg
import os
from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap

# Ensure the output directory exists
OUTPUT_DIR = "/Users/ayush/Desktop/Final_Report/clustering2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to plot heat wave hits on a map using Basemap
def plot_hits(number_families, fgv, v, plot_title):
    families = np.arange(number_families)
    for F in families:
        # Create a DeepGraph instance for each component
        gt = dg.DeepGraph(fgv.loc[F])

        # Extract data for plotting
        lon = gt.v.longitude.values
        lat = gt.v.latitude.values
        data = gt.v.n_nodes.values  # Replace with the appropriate data column

        # Skip if there is insufficient data
        if len(lon) < 3 or len(lat) < 3:
            print(f"Skipping cluster {F + 1} due to insufficient data.")
            continue

        # Create a regular grid
        grid_lon = np.linspace(v.longitude.min(), v.longitude.max(), 100)
        grid_lat = np.linspace(v.latitude.min(), v.latitude.max(), 100)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        # Interpolate data onto the grid
        data_grid = griddata((lon, lat), data, (grid_x, grid_y), method='linear')

        # Fill missing values using nearest-neighbor interpolation
        if np.any(np.isnan(data_grid)):
            data_grid = griddata((lon, lat), data, (grid_x, grid_y), method='nearest')

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlon=grid_lon.min(), urcrnrlon=grid_lon.max(),
                    llcrnrlat=grid_lat.min(), urcrnrlat=grid_lat.max(), resolution='l', ax=ax)
        m.drawcoastlines(linewidth=0.8)
        m.drawparallels(range(-50, 50, 20), linewidth=0.2)
        m.drawmeridians(range(0, 360, 20), linewidth=0.2)

        # Plot the data using pcolormesh
        pcm = m.pcolormesh(grid_lon, grid_lat, data_grid, cmap='viridis_r', shading='auto', latlon=True)

        # Add colorbar
        cbar = m.colorbar(pcm, location='right', pad=0.05)
        cbar.set_label('Number of Heatwave Days', fontsize=15)

        # Add title
        ax.set_title(f'{plot_title} Cluster {F + 1}', fontsize=16)

        # Save the plot
        save_path = os.path.join(OUTPUT_DIR, f'Heatwavedays_{plot_title}_Cluster_{F + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close(fig)

# Function to plot heat wave families on a map using Basemap
def plot_families(number_families, fgv, v, plot_title):
    families = np.arange(number_families)
    for F in families:
        # Create a DeepGraph instance for each component
        gt = dg.DeepGraph(fgv.loc[F])

        # Extract data for plotting
        lon = gt.v.longitude.values
        lat = gt.v.latitude.values
        data = gt.v.n_cp_nodes.values  # Replace with the appropriate data column

        # Skip if there is insufficient data
        if len(lon) < 3 or len(lat) < 3:
            print(f"Skipping cluster {F + 1} due to insufficient data.")
            continue

        # Create a regular grid
        grid_lon = np.linspace(v.longitude.min(), v.longitude.max(), 100)
        grid_lat = np.linspace(v.latitude.min(), v.latitude.max(), 100)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        # Interpolate data onto the grid
        data_grid = griddata((lon, lat), data, (grid_x, grid_y), method='linear')

        # Fill missing values using nearest-neighbor interpolation
        if np.any(np.isnan(data_grid)):
            data_grid = griddata((lon, lat), data, (grid_x, grid_y), method='nearest')

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlon=grid_lon.min(), urcrnrlon=grid_lon.max(),
                    llcrnrlat=grid_lat.min(), urcrnrlat=grid_lat.max(), resolution='l', ax=ax)
        m.drawcoastlines(linewidth=0.8)
        m.drawparallels(range(-50, 50, 20), linewidth=0.2)
        m.drawmeridians(range(0, 360, 20), linewidth=0.2)

        # Plot the data using pcolormesh
        pcm = m.pcolormesh(grid_lon, grid_lat, data_grid, cmap='viridis_r', shading='auto', latlon=True)

        # Add colorbar
        cbar = m.colorbar(pcm, location='right', pad=0.05)
        cbar.set_label('Number of Heatwaves', fontsize=15)

        # Add title
        ax.set_title(f'{plot_title} Cluster {F + 1}', fontsize=16)

        # Save the plot
        save_path = os.path.join(OUTPUT_DIR, f'{plot_title}_Cluster_{F + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close(fig)