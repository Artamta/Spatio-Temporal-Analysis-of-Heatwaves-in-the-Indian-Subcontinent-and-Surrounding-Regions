import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob

# Load first 4 families for demonstration
family_files = sorted(glob.glob("/Users/ayush/Desktop/Final_Report/clustering2/cpv_fam*.csv"))
dfs = []
for i, fam_file in enumerate(family_files[:4], 1):
    df = pd.read_csv(fam_file)
    df['family'] = f'fam{i}'
    dfs.append(df)
all_families = pd.concat(dfs, ignore_index=True)

# Ensure duration exists
all_families['time_amin'] = pd.to_datetime(all_families['time_amin'])
all_families['time_amax'] = pd.to_datetime(all_families['time_amax'])
all_families['duration'] = (all_families['time_amax'] - all_families['time_amin']).dt.days + 1

# 1. Scatterplot: Duration vs. Magnitude, color by family and style by subfamily
plt.figure(figsize=(10,7))
sns.scatterplot(
    x='duration', y='HWMId_magnitude',
    hue='family',
    style='F_upgma' if 'F_upgma' in all_families.columns else None,
    data=all_families, palette='Set1', alpha=0.7, edgecolor='k', s=70
)
sns.regplot(
    x='duration', y='HWMId_magnitude',
    data=all_families, scatter=False, color='gray', line_kws={'linestyle':'dashed'}
)
plt.xlabel('Duration (days)')
plt.ylabel('Magnitude')
plt.title('Duration vs. Magnitude by Family and Subfamily')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Heatmap: Family vs. Month (mean duration and event count)
all_families['start_month'] = all_families['time_amin'].dt.month
fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)
# Mean duration
pivot_mean = all_families.pivot_table(index='family', columns='start_month', values='duration', aggfunc='mean')
sns.heatmap(pivot_mean, annot=True, cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Mean Duration')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Family')
# Count
pivot_count = all_families.pivot_table(index='family', columns='start_month', values='duration', aggfunc='count')
sns.heatmap(pivot_count, annot=True, cmap='Blues', ax=axes[1])
axes[1].set_title('Event Count')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('')
plt.suptitle('Family vs. Month: Mean Duration and Event Count', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. PCA plot: Each point is a subfamily, colored by family, size by subfamily size, label by subfamily
if 'F_upgma' in all_families.columns:
    subfam_agg = all_families.groupby(['family', 'F_upgma']).agg({
        'duration': 'mean',
        'HWMId_magnitude': 'mean',
        'latitude_mean': 'mean',
        'longitude_mean': 'mean',
        'family': 'count'
    }).rename(columns={'family': 'subfam_size'}).reset_index()
    features = ['duration', 'HWMId_magnitude', 'latitude_mean', 'longitude_mean']
    X = StandardScaler().fit_transform(subfam_agg[features])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    subfam_agg['PC1'] = pcs[:,0]
    subfam_agg['PC2'] = pcs[:,1]
    plt.figure(figsize=(10,7))
    sns.scatterplot(
        x='PC1', y='PC2', hue='family', size='subfam_size', data=subfam_agg,
        palette='Set2', sizes=(40, 300), alpha=0.8, edgecolor='k', legend='brief'
    )
    for _, row in subfam_agg.iterrows():
        plt.text(row['PC1'], row['PC2'], str(int(row['F_upgma'])), fontsize=8, ha='center', va='center')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Subfamilies (colored by Family, size=subfamily size)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No 'F_upgma' column found for subfamily PCA plot.")