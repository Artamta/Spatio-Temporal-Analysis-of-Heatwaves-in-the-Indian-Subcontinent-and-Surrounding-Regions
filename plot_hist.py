import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm

outdir = "/Users/ayush/Desktop/Final_Report/analysis"

df = pd.read_csv("/Users/ayush/Desktop/Final_Report/results/extr_new.csv")
df['time'] = pd.to_datetime(df['time'])
sns.set(style="whitegrid", font_scale=1.3)

# 1. Histogram of Temperature
plt.figure(figsize=(8, 5))
sns.histplot(df['t2m'], bins=30, kde=True, color='tomato')
plt.xlabel('Temperature (°C)')
plt.ylabel('Number of Events')
plt.title('Distribution of Heatwave Temperatures')
plt.tight_layout()
plt.savefig(f"{outdir}/hist_temperature.png", dpi=300)
plt.close()

# 2. Histogram of Event Dates
plt.figure(figsize=(10, 5))
sns.histplot(df['time'], bins=30, kde=False, color='royalblue')
plt.xlabel('Date')
plt.ylabel('Number of Events')
plt.title('Distribution of Heatwave Event Dates')
plt.tight_layout()
plt.savefig(f"{outdir}/hist_event_dates.png", dpi=300)
plt.close()

# 3. Scatter Distribution of Heat Events (Simple, Blue-Red)
plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
region = [-10, 45, df['latitude'].min()-2, df['latitude'].max()+2]
ax.set_extent(region)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
sc = ax.scatter(
    df['longitude'], df['latitude'],
    c=df['t2m'], cmap='coolwarm', s=18, alpha=0.7,
    transform=ccrs.PlateCarree(), edgecolor='none'
)
plt.colorbar(sc, label='Temperature (°C)')
plt.title('Spatial Distribution of Heatwave Events')
plt.tight_layout()
plt.savefig(f"{outdir}/scatter_locations_simple.png", dpi=300)
plt.close()

# 4. Heatwave Event Density (Hexbin) with Simple Map
plt.figure(figsize=(12, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(region)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
hb = ax.hexbin(
    df['longitude'], df['latitude'],
    gridsize=40, cmap='YlOrRd', alpha=0.85, mincnt=1,
    transform=ccrs.PlateCarree()
)
plt.colorbar(hb, label='Number of Events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatwave Event Density')
plt.tight_layout()
plt.savefig(f"{outdir}/heatmap_event_density_simple.png", dpi=300)
plt.close()

# 5. KDE Spatial Density (No Map, as before)
plt.figure(figsize=(12, 7))
sns.kdeplot(
    x=df['longitude'], y=df['latitude'],
    cmap="Reds", fill=True, thresh=0.05, levels=100, alpha=0.7
)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('KDE Spatial Density of Heatwave Events')
plt.tight_layout()
plt.savefig(f"{outdir}/kde_spatial_density.png", dpi=300)
plt.close()

# 6. Trend of Temperature Over Time (with rolling mean and LOWESS)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='time', y='t2m', data=df, color='grey', alpha=0.3, s=10, label='Events')
df_sorted = df.sort_values('time')
rolling = df_sorted['t2m'].rolling(window=30, center=True).mean()
plt.plot(df_sorted['time'], rolling, color='red', linewidth=2, label='30-event Rolling Mean')
lowess = sm.nonparametric.lowess(df_sorted['t2m'], df_sorted['time'], frac=0.1)
plt.plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=2, label='LOWESS Trend')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Trend of Heatwave Temperatures Over Time')
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/trend_temperature_time.png", dpi=300)
plt.close()

# 7. Yearly Count of Heatwave Events (fixed x labels)
df['year'] = df['time'].dt.year
plt.figure(figsize=(12, 6))
sns.countplot(x='year', data=df, color='royalblue')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.title('Yearly Count of Heatwave Events')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{outdir}/yearly_event_count.png", dpi=300)
plt.close()