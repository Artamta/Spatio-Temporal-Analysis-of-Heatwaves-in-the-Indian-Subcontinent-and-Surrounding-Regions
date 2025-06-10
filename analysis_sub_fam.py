import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import statsmodels.api as sm

sns.set(style="whitegrid", font_scale=1.3)

# Path to all family CSVs
family_files = sorted(glob.glob("/Users/ayush/Desktop/Final_Report/clustering2/cpv_fam*.csv"))
base_outdir = "/Users/ayush/Desktop/Final_Report/analysis/Sub_families"

for fam_idx, fam_file in enumerate(family_files, 1):
    df = pd.read_csv(fam_file)
    df['time_amin'] = pd.to_datetime(df['time_amin'])
    df['time_amax'] = pd.to_datetime(df['time_amax'])
    df['duration'] = (df['time_amax'] - df['time_amin']).dt.days + 1
    df['start_month'] = df['time_amin'].dt.month
    df['start_year'] = df['time_amin'].dt.year
    df['start_doy'] = df['time_amin'].dt.dayofyear
    df['start_doy_bin'] = pd.cut(df['start_doy'], bins=12, labels=False)

    fam_outdir = os.path.join(base_outdir, f"fam{fam_idx}")
    os.makedirs(fam_outdir, exist_ok=True)

    # Loop over all subfamilies (F_upgma)
    for subfam in sorted(df['F_upgma'].unique()):
        subdf = df[df['F_upgma'] == subfam]
        outdir = os.path.join(fam_outdir, f"subfamily{subfam}")
        os.makedirs(outdir, exist_ok=True)
        subfam_title = f"Fam{fam_idx} Subfamily{subfam}"

        # 1. Histogram of Duration
        plt.figure(figsize=(8, 5))
        sns.histplot(subdf['duration'], bins=20, color='tomato', kde=True)
        plt.xlabel('Duration (days)')
        plt.ylabel('Number of Events')
        plt.title(f'{subfam_title}: Duration Distribution')
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_duration.png", dpi=300)
        plt.close()

        # 2. Histogram of Magnitude
        plt.figure(figsize=(8, 5))
        sns.histplot(subdf['HWMId_magnitude'], bins=20, color='seagreen', kde=True)
        plt.xlabel('Magnitude')
        plt.ylabel('Number of Events')
        plt.title(f'{subfam_title}: Magnitude Distribution')
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_magnitude.png", dpi=300)
        plt.close()

        # 3. Histogram of Start Day of Year
        plt.figure(figsize=(8, 5))
        sns.histplot(subdf['start_doy'], bins=20, color='darkorange', kde=True)
        plt.xlabel('Start Day of Year')
        plt.ylabel('Number of Events')
        plt.title(f'{subfam_title}: Start Day of Year')
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_start_doy.png", dpi=300)
        plt.close()

        # 4. Scatter: Duration vs Magnitude
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='duration', y='HWMId_magnitude', data=subdf, alpha=0.7)
        plt.xlabel('Duration (days)')
        plt.ylabel('Magnitude')
        plt.title(f'{subfam_title}: Duration vs Magnitude')
        plt.tight_layout()
        plt.savefig(f"{outdir}/scatter_duration_vs_magnitude.png", dpi=300)
        plt.close()

        # 5. Timeline (Gantt) for subfamily
        plt.figure(figsize=(10, 6))
        for i, row in subdf.iterrows():
            plt.plot([row['time_amin'], row['time_amax']], [i, i], color='red', alpha=0.6)
        plt.xlabel('Date')
        plt.ylabel('Event Index')
        plt.title(f'{subfam_title}: Timeline')
        plt.tight_layout()
        plt.savefig(f"{outdir}/gantt_timeline.png", dpi=300)
        plt.close()

        # 6. Spatial distribution (if columns exist)
        if 'latitude_mean' in subdf.columns and 'longitude_mean' in subdf.columns:
            plt.figure(figsize=(8, 7))
            plt.scatter(subdf['longitude_mean'], subdf['latitude_mean'], c=subdf['start_year'], cmap='viridis', s=60, alpha=0.8)
            plt.colorbar(label='Start Year')
            plt.xlabel('Mean Longitude')
            plt.ylabel('Mean Latitude')
            plt.title(f'{subfam_title}: Spatial Distribution')
            plt.tight_layout()
            plt.savefig(f"{outdir}/spatial_distribution.png", dpi=300)
            plt.close()

        # 7. Boxplot: Duration by Start Month
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='start_month', y='duration', data=subdf, palette='coolwarm')
        plt.xlabel('Start Month')
        plt.ylabel('Duration (days)')
        plt.title(f'{subfam_title}: Duration by Start Month')
        plt.tight_layout()
        plt.savefig(f"{outdir}/boxplot_duration_by_month.png", dpi=300)
        plt.close()

        # 8. Violin Plot: Magnitude by Start Month
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='start_month', y='HWMId_magnitude', data=subdf, palette='viridis')
        plt.xlabel('Start Month')
        plt.ylabel('Magnitude')
        plt.title(f'{subfam_title}: Magnitude by Start Month')
        plt.tight_layout()
        plt.savefig(f"{outdir}/violin_magnitude_by_month.png", dpi=300)
        plt.close()

        # 9. LOWESS trend for duration over years
        if subdf['start_year'].nunique() > 1:
            plt.figure(figsize=(10,6))
            sns.scatterplot(x='start_year', y='duration', data=subdf, alpha=0.4)
            lowess = sm.nonparametric.lowess(subdf['duration'], subdf['start_year'], frac=0.5)
            plt.plot(lowess[:,0], lowess[:,1], color='red', linewidth=2, label='LOWESS Trend')
            plt.xlabel('Year')
            plt.ylabel('Duration (days)')
            plt.title(f'{subfam_title}: Duration Trend Over Years')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{outdir}/lowess_duration_trend.png", dpi=300)
            plt.close()

print("All subfamily analysis plots saved in:", base_outdir)