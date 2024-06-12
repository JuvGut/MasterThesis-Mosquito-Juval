import matplotlib.pyplot as plt
import pandas as pd

# Load the datasets from the Excel files
file_paths = [
    "/Users/Juval/Downloads/Inception-ResNet-results/metrics files/243-metrics.xlsx",
    "/Users/Juval/Downloads/Inception-ResNet-results/metrics files/817-metrics.xlsx",
    "/Users/Juval/Downloads/Inception-ResNet-results/metrics files/combined-metrics.xlsx"
]

# Read the Excel files
dataframes = [pd.read_excel(path) for path in file_paths]

# Display the first few rows of each dataframe to inspect their structure
for df in dataframes:
    print(df.head())
# Calculate the average score for each metric in each dataset
averaged_data = []
for df in dataframes:
    df['Average'] = df[['Train Set', 'Validation Set', 'Test Set']].mean(axis=1)
    averaged_data.append(df[['Metric', 'Average']])

# Merge the averaged data for plotting
merged_data = pd.concat(averaged_data, axis=1, keys=['Dataset1', 'Dataset2', 'Dataset3'])

# Now extract the Metric names and the averages from each dataset for plotting
metric_names = averaged_data[0]['Metric']
averages_dataset1 = averaged_data[0]['Average']
averages_dataset2 = averaged_data[1]['Average']
averages_dataset3 = averaged_data[2]['Average']

# Create a grouped bar chart
x = range(len(metric_names))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x, averages_dataset1, width, label='Dataset 1')
bars2 = ax.bar([p + width for p in x], averages_dataset2, width, label='Dataset 2')
bars3 = ax.bar([p + width*2 for p in x], averages_dataset3, width, label='Dataset 3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metric Scores Across Datasets')
ax.set_xticks([p + width for p in x])
ax.set_xticklabels(metric_names)
ax.legend()

# Rotate the tick labels for better viewing
plt.xticks(rotation=45)
# plt.show()



# Extracting the 'Test Set' scores from each dataset
test_set_scores = [df[['Metric', 'Test Set']] for df in dataframes]

# Now extract the Metric names and the test set scores from each dataset for plotting
test_scores_dataset1 = test_set_scores[0]['Test Set']
test_scores_dataset2 = test_set_scores[1]['Test Set']
test_scores_dataset3 = test_set_scores[2]['Test Set']

# Create a grouped bar chart focusing only on the Test Set
fig, ax = plt.subplots()
bars1 = ax.bar(x, test_scores_dataset1, width, label='Camera 243')
bars2 = ax.bar([p + width for p in x], test_scores_dataset2, width, label='Camera 817')
bars3 = ax.bar([p + width*2 for p in x], test_scores_dataset3, width, label='combined')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metric Scores in Test Set Across Datasets')
ax.set_xticks([p + width for p in x])
ax.set_xticklabels(metric_names)
ax.legend()

# Rotate the tick labels for better viewing
plt.xticks(rotation=45)
plt.savefig("/Users/Juval/Downloads/Inception-ResNet-results/metrics files/compare-bar-plot.png", bbox_inches='tight')
plt.show()