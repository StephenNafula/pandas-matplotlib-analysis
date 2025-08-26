"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib
Author: [Your Name]
"""

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore the Dataset
# -----------------------------

try:
    # Load iris dataset from sklearn
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame  # Convert to pandas DataFrame

    # Display first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check dataset info
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (fill missing values if any)
    df = df.fillna(df.mean(numeric_only=True))

except FileNotFoundError:
    print("Error: Dataset not found.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean values
grouped = df.groupby("target").mean()
print("\nMean values grouped by species:")
print(grouped)

# Pattern observation
print("\nObservation:")
print("The grouped statistics show clear differences in petal and sepal measurements among species.")

# -----------------------------
# Task 3: Data Visualization
# -----------------------------

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line chart (simulate trend over index as no time column is available)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.plot(df.index, df["petal length (cm)"], label="Petal Length")
plt.title("Line Chart: Sepal & Petal Lengths Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
df.groupby("target")["petal length (cm)"].mean().plot(kind="bar", color=["#FF9999","#66B3FF","#99FF99"])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.xticks([0,1,2], iris_data.target_names)
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species", labels=iris_data.target_names)
plt.show()
