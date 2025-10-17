# -*- coding: utf-8 -*-
"""MedEduChat.ipynb
"""

from google.colab import drive
drive.mount('/content/drive')
df.to_csv('/content/drive/MyDrive/NPJ_DigMed/new_dataframe.csv', index=False)


"""# Analyze Clinician Grading Study

"""
grading = "/content/drive/MyDrive/LLM_Chatbot/MedEduChat/NPJ_Review/Supplementary Material/MedEduChat_Clinical_Grading.csv"

import pandas as pd
import scipy.stats as st, iqr, variation
import numpy as np
import krippendorff  # pip install krippendorff
import pingouin as pg # !pip install pingouin
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a pandas DataFrame
grading = pd.read_csv(grading)
grading.columns

cols_to_check = [
    "Answer's Correctness",
    "Patient Question's Difficulty",
    "Answer's Completeness",
    "Answer is Patient-Ready?",
    "Safety and Harm Potential?",
    "Patient-Centered Response?"
]

# Calculate Krippendorff's alpha for each column
for col in cols_to_check:
    pivot = grading_filtered.pivot_table(index='Problem ID', columns='Rater_ID', values=col)
    matrix = pivot.T.to_numpy()
    try:
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
        print(f"Krippendorff's Alpha for '{col}' (Origin: ID080): {round(alpha, 3)}")
    except Exception as e:
        print(f"Could not calculate Krippendorff's Alpha for '{col}' (Origin: ID080): {e}")

grouped = grading.groupby('Origin')[['Answer\'s Correctness', 'Patient Question\'s Difficulty',
       'Answer\'s Completeness', 'Answer is Patient-Ready?',
       'Safety and Harm Potential?', 'Patient-Centered Response?']].mean()

grouped

# Calculate Cronbach's alpha for the grouped data
cronbach_alpha = pg.cronbach_alpha(data=grouped)

print("Cronbach's Alpha (Inter-rater Reliability):")
print(cronbach_alpha)

stats_columns = ['Answer\'s Correctness', 'Patient Question\'s Difficulty',
                 'Answer\'s Completeness', 'Answer is Patient-Ready?',
                 'Safety and Harm Potential?', 'Patient-Centered Response?']

print("Descriptive Statistics for Grading Metrics:")
display(grading[stats_columns].describe())

# Calculate 95% confidence interval for each column
confidence_intervals = {}
for col in stats_columns:
    data = grading[col].dropna()
    if len(data) > 1: 
      interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
      confidence_intervals[col] = interval
    else:
      confidence_intervals[col] = (np.nan, np.nan)

print("\n95% Confidence Intervals for Grading Metrics:")
for col, ci in confidence_intervals.items():
    print(f"{col}: ({ci[0]:.3f}, {ci[1]:.3f})")

# Generate all unique pairs of columns
pairs_to_compare = list(combinations(cols_to_check, 2))


print("Paired Differences and Effect Sizes:")

for col1, col2 in pairs_to_compare:
    print(f"\nComparing '{col1}' and '{col2}':")

    # Perform paired t-test
    # Ensure there are no NaNs in either column for the paired test
    temp_df = grading[[col1, col2]].dropna()
    if len(temp_df) > 1:
        ttest_results = pg.ttest(x=temp_df[col1], y=temp_df[col2], paired=True)
        display(ttest_results)

        # Calculate Cohen's d (effect size)
        effect_size = pg.compute_effsize(x=temp_df[col1], y=temp_df[col2], paired=True, eftype='cohen')
        print(f"Cohen's d: {effect_size:.3f}")
    else:
        print("Not enough data to perform paired test and calculate effect size.")

# Prepare data for descriptive statistics table
stats_summary = grading[stats_columns].describe().T[['mean', 'std', 'min', 'max']]
stats_summary['95% CI Lower'] = [confidence_intervals[col][0] for col in stats_summary.index]
stats_summary['95% CI Upper'] = [confidence_intervals[col][1] for col in stats_summary.index]
print("Summary Table of Grading Metrics:")
display(stats_summary)

# Prepare data for paired differences and effect sizes table
paired_results_list = []
for col1, col2 in pairs_to_compare:
    temp_df = grading[[col1, col2]].dropna()
    if len(temp_df) > 1:
        ttest_results = pg.ttest(x=temp_df[col1], y=temp_df[col2], paired=True)
        effect_size = pg.compute_effsize(x=temp_df[col1], y=temp_df[col2], paired=True, eftype='cohen')
        paired_results_list.append({
            'Comparison': f'{col1} vs {col2}',
            'T-statistic': ttest_results['T'].iloc[0],
            'Degrees of Freedom': ttest_results['dof'].iloc[0],
            'P-value': ttest_results['p-val'].iloc[0],
            'Cohen\'s d': effect_size
        })
    else:
         paired_results_list.append({
            'Comparison': f'{col1} vs {col2}',
            'T-statistic': np.nan,
            'Degrees of Freedom': np.nan,
            'P-value': np.nan,
            'Cohen\'s d': np.nan
        })


paired_summary_df = pd.DataFrame(paired_results_list)

print("\nSummary Table of Paired Differences and Effect Sizes:")
display(paired_summary_df)


# Create box plots 
plt.figure(figsize=(14, 10))
sns.boxplot(x='Metric', y='Score', data=grouped,
            linewidth=5, color='darkorange',
            flierprops={'marker':'o', 'markersize':30})

# plt.title('Box plots for different metrics by Origin', fontsize=28) 
plt.ylabel('Mean Score', fontsize=26) 
plt.xlabel('Metric', fontsize=26)
plt.xticks(ha='right', fontsize=24) 
plt.yticks(fontsize=24) 
plt.grid(True, linestyle='-', alpha=0.6) 

plt.tight_layout()

with PdfPages(pdf_filename) as pdf:
    pdf.savefig(plt.gcf()) 

plt.show()


def krippendorff_alpha_manual(data, level='ordinal'):
    data = np.asarray(data)
    n_raters, n_items = data.shape

    if np.all(data == data[0, 0]):
        return 1.0

    categories = sorted(set(data.flatten()))
    n_cat = len(categories)
    cat_to_rank = {cat: i for i, cat in enumerate(categories)}
    ranks = np.vectorize(cat_to_rank.get)(data)

    m = n_cat
    distance_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            distance_matrix[i, j] = ((i - j) ** 2) / ((m - 1) ** 2)

    Do = 0
    for j in range(n_items):
        item_ratings = ranks[:, j]
        for a, b in combinations(item_ratings, 2):
            Do += distance_matrix[a, b]
    Do *= 2 / (n_items * n_raters * (n_raters - 1))

    all_ratings = ranks.flatten()
    counts = Counter(all_ratings)
    total = len(all_ratings)

    De = 0
    for i in range(m):
        for j in range(m):
            pi = counts[i] / total
            pj = counts[j] / total
            De += pi * pj * distance_matrix[i, j]

    if De == 0:
        return 1.0 if Do == 0 else np.nan

    return 1 - Do / De
