import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV files
df1 = pd.read_csv(r'gpt_label\tweet_anger_gpt.csv')
df2 = pd.read_csv(r'C:\Users\decroux paul\Documents\code_stage_uk\data\feelings\relation_feelings_hate speech\anger_x_fear.csv')

# Get relevant columns
anger_gpt_score = df1['anger_gpt_score']
anger_label = df2['anger_label']

# Calculate mean and standard deviation
mean_gpt = np.mean(anger_gpt_score)
std_gpt = np.std(anger_gpt_score)
mean_label = np.mean(anger_label)
std_label = np.std(anger_label)

# Define intervals for grouping scores
intervals = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

# Calculate the frequency of scores in each interval for each system
freq_gpt = pd.cut(anger_gpt_score, bins=[i[0] for i in intervals] + [1.0], labels=False, right=False).value_counts(normalize=True).sort_index()
freq_label = pd.cut(anger_label, bins=[i[0] for i in intervals] + [1.0], labels=False, right=False).value_counts(normalize=True).sort_index()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot the histogram of scores for the GPT system
ax1.bar(np.arange(len(freq_gpt))-0.5, freq_gpt.values)
ax1.axvline(mean_gpt*10-0.5, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_gpt:.2f}')
ax1.axvspan(mean_gpt*10 - std_gpt*10-0.5, mean_gpt*10 + std_gpt*10-0.5, color='red', alpha=0.1, label=f'Std: {std_gpt:.2f}')
ax1.set_title('Score Frequency - GPT System')
ax1.set_xlabel('Interval')
ax1.set_ylabel('Frequency')
ax1.set_xticks(range(10))
ax1.set_xticklabels([f'{i[1]}' for i in intervals])
ax1.legend()

# Plot the histogram of scores for the reference system
ax2.bar(np.arange(len(freq_label))-0.5, freq_label.values)
ax2.axvline(mean_label*10-0.5, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_label:.2f}')
ax2.axvspan(mean_label*10 - std_label*10-0.5, mean_label*10 + std_label*10-0.5, color='red', alpha=0.1, label=f'Std: {std_label:.2f}')
ax2.set_title('Score Frequency - Reference System')
ax2.set_xlabel('Interval')
ax2.set_ylabel('Frequency')
ax2.set_xticks(range(10))
ax2.set_xticklabels([f'{i[1]}' for i in intervals])
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()

plt.show()
