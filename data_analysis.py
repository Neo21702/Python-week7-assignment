# analyze_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Error handling for dataset loading
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First 5 rows of the dataset:\n", df.head())

# Explore structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in the dataset:\n", df.isnull().sum())

# Clean the dataset (none in Iris, but this is how you'd handle them)
df.dropna(inplace=True)  # Example cleanup step

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Describe dataset (summary statistics)
print("\nBasic Statistics:\n", df.describe())

# Group by species and get the mean of each feature
grouped_means = df.groupby('species').mean()
print("\nMean values per species:\n", grouped_means)

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Set the Seaborn style
sns.set(style="whitegrid")

# Line Chart: Plot the sepal length across all samples (simulated time-series)
plt.figure(figsize=(10, 5))
plt.plot(df['sepal length (cm)'], label='Sepal Length', color='b')
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
grouped_means['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='orange', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# Scatter Plot: Sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# -------------------------------
# Findings and Observations
# -------------------------------
print("\nObservations:")
print("- Setosa species tends to have smaller petal and sepal dimensions.")
print("- There is a strong positive correlation between sepal length and petal length.")
print("- Petal length varies significantly between species, making it a good predictor for classification.")
