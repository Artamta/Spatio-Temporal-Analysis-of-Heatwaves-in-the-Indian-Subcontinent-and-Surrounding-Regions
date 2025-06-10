import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm
import os

outdir = "/Users/ayush/Desktop/Final_Report/analysis"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv("/Users/ayush/Desktop/Final_Report/results/extr_new.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.dropna(subset=['time'])

sns.set(style="whitegrid", font_scale=1.3)

region = [35, 135, -15, 45]  # Adjust as needed for your study area

# 1. KDE Spatial Density with Map (Cartopy)
from scipy.stats import gaussian_kde

# Prepare grid for KDE
lon = df['longitude'].values
lat = df['latitude'].values
kde = gaussian_kde(np.vstack([lon, lat]))
lon_grid, lat_grid = np.meshgrid(
    np.linspace(region[0], region[1], 300),
    np.linspace(region[2], region[3], 300)
)
kde_values = kde(np.vstack([lon_grid.ravel(), lat_grid.ravel()])).reshape(lon_grid.shape)

plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(region)
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.7)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke', alpha=0.3)
cf = ax.contourf(lon_grid, lat_grid, kde_values, 20, cmap='Reds', alpha=0.7, transform=ccrs.PlateCarree())
plt.colorbar(cf, label='KDE Density')
plt.title('KDE Spatial Density of Heatwave Events (with Map)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(f"{outdir}/kde_spatial_density_map.png", dpi=300)
plt.close()

# 2. Monthly Count of Heatwave Events
df['month'] = df['time'].dt.month
plt.figure(figsize=(10, 6))
sns.countplot(x='month', data=df, color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Events')
plt.title('Monthly Count of Heatwave Events')
plt.tight_layout()
plt.savefig(f"{outdir}/monthly_event_count.png", dpi=300)
plt.close()


# 4. Magnitude vs. Month Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='magnitude', data=df, palette='viridis')
plt.xlabel('Month')
plt.ylabel('Magnitude')
plt.title('Monthly Distribution of Heatwave Magnitude')
plt.tight_layout()
plt.savefig(f"{outdir}/boxplot_magnitude_by_month.png", dpi=300)
plt.close()


# 6. Temperature Trend with LOWESS
plt.figure(figsize=(12, 6))
sns.scatterplot(x='time', y='t2m', data=df, color='grey', alpha=0.3, s=10, label='Events')
df_sorted = df.sort_values('time')
lowess = sm.nonparametric.lowess(df_sorted['t2m'], df_sorted['time'], frac=0.1)
plt.plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=2, label='LOWESS Trend')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Trend of Heatwave Temperatures Over Time')
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/trend_temperature_time_lowess.png", dpi=300)
plt.close()

# 7. Yearly Count of Heatwave Events
plt.figure(figsize=(12, 6))
sns.countplot(x='year', data=df, color='royalblue')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.title('Yearly Count of Heatwave Events')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{outdir}/yearly_event_count.png", dpi=300)
plt.close()