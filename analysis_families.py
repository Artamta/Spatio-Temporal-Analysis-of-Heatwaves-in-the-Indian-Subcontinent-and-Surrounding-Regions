import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import statsmodels.api as sm

sns.set(style="whitegrid", font_scale=1.3)

base_outdir = "/Users/ayush/Desktop/Final_Report/analysis/families"

# Find all family files
family_files = sorted(glob.glob("/Users/ayush/Desktop/Final_Report/clustering_Pclr/cpv_fam*.csv"))

for fam_idx, fam_file in enumerate(family_files, 1):
    cpv = pd.read_csv(fam_file)
    print(f"Processing {fam_file} with columns: {cpv.columns.tolist()}")
    cpv['time_amin'] = pd.to_datetime(cpv['time_amin'])
    cpv['time_amax'] = pd.to_datetime(cpv['time_amax'])
    cpv['duration'] = (cpv['time_amax'] - cpv['time_amin']).dt.days + 1
    cpv['start_month'] = cpv['time_amin'].dt.month
    cpv['start_year'] = cpv['time_amin'].dt.year
    cpv['start_doy'] = cpv['time_amin'].dt.dayofyear
    cpv['start_doy_bin'] = pd.cut(cpv['start_doy'], bins=12, labels=False)

    outdir = os.path.join(base_outdir, f"family{fam_idx}")
    os.makedirs(outdir, exist_ok=True)

    fam_title = f"Family{fam_idx}"

    # 1. Histogram of Family Durations
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['duration'], bins=30, color='tomato', kde=True)
    plt.xlabel('Duration (days)')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Family Durations')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_duration.png", dpi=300)
    plt.close()

    # 2. Histogram of Family Start Dates (by month)
    plt.figure(figsize=(8, 5))
    sns.countplot(x='start_month', data=cpv, color='royalblue')
    plt.xlabel('Start Month')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Start Months')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_start_month.png", dpi=300)
    plt.close()

    # 3. Histogram of Family Magnitude
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['HWMId_magnitude'], bins=30, color='seagreen', kde=True)
    plt.xlabel('Magnitude')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Family Magnitude')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_magnitude.png", dpi=300)
    plt.close()

    # 4. Histogram of Family Start Day-of-Year
    plt.figure(figsize=(10, 5))
    sns.histplot(cpv['start_doy'], bins=36, color='darkorange', kde=True)
    plt.xlabel('Start Day of Year')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Start Days (Day of Year)')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_start_doy.png", dpi=300)
    plt.close()

    # 5. Boxplot: Duration by Start Month
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='start_month', y='duration', data=cpv, palette='coolwarm')
    plt.xlabel('Start Month')
    plt.ylabel('Duration (days)')
    plt.title(f'{fam_title}: Duration by Start Month')
    plt.tight_layout()
    plt.savefig(f"{outdir}/boxplot_duration_by_month.png", dpi=300)
    plt.close()

    # 6. Violin Plot: Magnitude by Start Month
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='start_month', y='HWMId_magnitude', data=cpv, palette='viridis')
    plt.xlabel('Start Month')
    plt.ylabel('Magnitude')
    plt.title(f'{fam_title}: Magnitude by Start Month')
    plt.tight_layout()
    plt.savefig(f"{outdir}/violin_magnitude_by_month.png", dpi=300)
    plt.close()

    # 7. Boxplot: Duration by Start Day-of-Year (Binned)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='start_doy_bin', y='duration', data=cpv, palette='coolwarm')
    plt.xlabel('Start Day of Year (Binned)')
    plt.ylabel('Duration (days)')
    plt.title(f'{fam_title}: Duration by Start Day of Year')
    plt.tight_layout()
    plt.savefig(f"{outdir}/boxplot_duration_by_start_doy.png", dpi=300)
    plt.close()

    # 8. Violin Plot: Magnitude by Start Day-of-Year (Binned)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='start_doy_bin', y='HWMId_magnitude', data=cpv, palette='viridis')
    plt.xlabel('Start Day of Year (Binned)')
    plt.ylabel('Magnitude')
    plt.title(f'{fam_title}: Magnitude by Start Day of Year')
    plt.tight_layout()
    plt.savefig(f"{outdir}/violin_magnitude_by_start_doy.png", dpi=300)
    plt.close()

    # 9. Scatter: Duration vs Magnitude (color by start month)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='duration', y='HWMId_magnitude', hue='start_month', palette='Spectral', data=cpv, alpha=0.7)
    plt.xlabel('Duration (days)')
    plt.ylabel('Magnitude')
    plt.title(f'{fam_title}: Duration vs Magnitude')
    plt.legend(title='Start Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{outdir}/scatter_duration_vs_magnitude.png", dpi=300)
    plt.close()

    # 10. Histogram of Number of Nodes in Each Family
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['n_nodes'], bins=20, color='slateblue', kde=True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Number of Nodes')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_n_nodes.png", dpi=300)
    plt.close()

    # 11. Histogram of Mean Latitude of Families
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['latitude_mean'], bins=20, color='darkgreen', kde=True)
    plt.xlabel('Mean Latitude')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Mean Latitude')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_latitude_mean.png", dpi=300)
    plt.close()

    # 12. Histogram of Mean Longitude of Families
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['longitude_mean'], bins=20, color='darkred', kde=True)
    plt.xlabel('Mean Longitude')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Mean Longitude')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_longitude_mean.png", dpi=300)
    plt.close()

    # 13. Histogram of Maximum Temperature (t2m_amax)
    plt.figure(figsize=(8, 5))
    sns.histplot(cpv['t2m_amax'], bins=20, color='orange', kde=True)
    plt.xlabel('Maximum Temperature (°C)')
    plt.ylabel('Number of Heatwave Families')
    plt.title(f'{fam_title}: Distribution of Maximum Temperature')
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_family_t2m_amax.png", dpi=300)
    plt.close()

    # 14. Histogram of Timespan (if present)
    if 'timespan' in cpv.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(cpv['timespan'], bins=20, color='teal', kde=True)
        plt.xlabel('Timespan')
        plt.ylabel('Number of Heatwave Families')
        plt.title(f'{fam_title}: Distribution of Timespan')
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_family_timespan.png", dpi=300)
        plt.close()

    # 15. Duration trend over years
    plt.figure(figsize=(10,6))
    sns.lineplot(x='start_year', y='duration', data=cpv, marker='o', ci=None)
    plt.xlabel('Year')
    plt.ylabel('Mean Duration (days)')
    plt.title(f'{fam_title}: Mean Duration Over Years')
    plt.tight_layout()
    plt.savefig(f"{outdir}/trend_duration_over_years.png", dpi=300)
    plt.close()

    # 16. Magnitude trend over years
    plt.figure(figsize=(10,6))
    sns.lineplot(x='start_year', y='HWMId_magnitude', data=cpv, marker='o', ci=None)
    plt.xlabel('Year')
    plt.ylabel('Mean Magnitude')
    plt.title(f'{fam_title}: Mean Magnitude Over Years')
    plt.tight_layout()
    plt.savefig(f"{outdir}/trend_magnitude_over_years.png", dpi=300)
    plt.close()

    # 17. Spatial Distribution of Family Centers
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(cpv['longitude_mean'], cpv['latitude_mean'], c=cpv['start_year'], cmap='viridis', s=60, alpha=0.8)
    plt.colorbar(sc, label='Start Year')
    plt.xlabel('Mean Longitude')
    plt.ylabel('Mean Latitude')
    plt.title(f'{fam_title}: Spatial Distribution of Family Centers')
    plt.tight_layout()
    plt.savefig(f"{outdir}/spatial_family_centers.png", dpi=300)
    plt.close()

    # 18. Heatmap: Month vs. Duration
    plt.figure(figsize=(10, 6))
    pivot = cpv.pivot_table(index='start_month', values='duration', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='YlOrRd')
    plt.xlabel('Month')
    plt.ylabel('')
    plt.title(f'{fam_title}: Mean Duration by Start Month')
    plt.tight_layout()
    plt.savefig(f"{outdir}/heatmap_duration_by_month.png", dpi=300)
    plt.close()

    # 19. Outlier scatter: Duration vs Magnitude, highlight top 5%
    threshold = cpv['HWMId_magnitude'].quantile(0.95)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='duration', y='HWMId_magnitude', data=cpv, alpha=0.5)
    outliers = cpv[cpv['HWMId_magnitude'] > threshold]
    plt.scatter(outliers['duration'], outliers['HWMId_magnitude'], color='red', label='Top 5% Magnitude')
    plt.xlabel('Duration (days)')
    plt.ylabel('Magnitude')
    plt.title(f'{fam_title}: Outlier Families (Top 5% Magnitude)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/outlier_scatter_duration_vs_magnitude.png", dpi=300)
    plt.close()

    # 20. LOWESS trend for duration over years
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='start_year', y='duration', data=cpv, alpha=0.4)
    lowess = sm.nonparametric.lowess(cpv['duration'], cpv['start_year'], frac=0.5)
    plt.plot(lowess[:,0], lowess[:,1], color='red', linewidth=2, label='LOWESS Trend')
    plt.xlabel('Year')
    plt.ylabel('Duration (days)')
    plt.title(f'{fam_title}: Duration Trend Over Years')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/lowess_duration_trend.png", dpi=300)
    plt.close()

    # 21. Composite: Start Day vs. Magnitude
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='start_doy', y='HWMId_magnitude', data=cpv, alpha=0.6)
    plt.xlabel('Start Day of Year')
    plt.ylabel('Magnitude')
    plt.title(f'{fam_title}: Start Day vs. Magnitude')
    plt.tight_layout()
    plt.savefig(f"{outdir}/scatter_startday_vs_magnitude.png", dpi=300)
    plt.close()

print("All family analysis plots saved in:", base_outdir)