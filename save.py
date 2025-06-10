import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your main heatwave events CSV
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
plt.savefig("hist_temperature.png", dpi=300)
plt.show()

# 2. Histogram of Event Dates
plt.figure(figsize=(10, 5))
sns.histplot(df['time'], bins=30, kde=False, color='royalblue')
plt.xlabel('Date')
plt.ylabel('Number of Events')
plt.title('Distribution of Heatwave Event Dates')
plt.tight_layout()
plt.savefig("hist_event_dates.png", dpi=300)
plt.show()

# 3. Histogram of Magnitude
plt.figure(figsize=(8, 5))
sns.histplot(df['magnitude'], bins=30, kde=True, color='seagreen')
plt.xlabel('Magnitude')
plt.ylabel('Number of Events')
plt.title('Distribution of Heatwave Magnitude')
plt.tight_layout()
plt.savefig("hist_magnitude.png", dpi=300)
plt.show()

# 4. Improved Spatial Distribution (Scatter Plot)
plt.figure(figsize=(12, 7))
sc = plt.scatter(
    df['longitude'], df['latitude'],
    c=df['t2m'], cmap='hot', s=25, alpha=0.7, edgecolor='k', linewidth=0.2
)
plt.colorbar(sc, label='Temperature (°C)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Heatwave Events (Colored by Temperature)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("scatter_locations_improved.png", dpi=300)
plt.show()

# 5. Boxplot of Temperature by Month
df['month'] = df['time'].dt.month
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='t2m', data=df, palette='coolwarm')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.title('Monthly Distribution of Heatwave Temperatures')
plt.tight_layout()
plt.savefig("boxplot_temp_by_month.png", dpi=300)
plt.show()

# 6. Improved Heatmap of Event Density (2D Histogram)
plt.figure(figsize=(12, 7))
h = plt.hist2d(
    df['longitude'], df['latitude'],
    bins=[40, 40], cmap='YlOrRd', alpha=0.85
)
plt.colorbar(h[3], label='Number of Events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatmap of Heatwave Event Density')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("heatmap_event_density_improved.png", dpi=300)
plt.show()

# 7. KDE Plot for Spatial Density
plt.figure(figsize=(12, 7))
sns.kdeplot(
    x=df['longitude'], y=df['latitude'],
    cmap="Reds", fill=True, thresh=0.05, levels=100, alpha=0.7
)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('KDE Spatial Density of Heatwave Events')
plt.tight_layout()
plt.savefig("kde_spatial_density.png", dpi=300)
plt.show()

# 8. Trend of Temperature Over Time (with rolling mean)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='time', y='t2m', data=df, color='grey', alpha=0.3, s=10, label='Events')
df_sorted = df.sort_values('time')
rolling = df_sorted['t2m'].rolling(window=30, center=True).mean()
plt.plot(df_sorted['time'], rolling, color='red', linewidth=2, label='30-event Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Trend of Heatwave Temperatures Over Time')
plt.legend()
plt.tight_layout()
plt.savefig("trend_temperature_time.png", dpi=300)
plt.show()

# 9. Yearly Count of Heatwave Events
df['year'] = df['time'].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=df, color='royalblue')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.title('Yearly Count of Heatwave Events')
plt.tight_layout()
plt.savefig("yearly_event_count.png", dpi=300)
plt.show()