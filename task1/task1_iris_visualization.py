# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_theme()

print("Libraries Imported Successfully!")

# 2. Load Dataset
# Load Iris Dataset from Seaborn
df = sns.load_dataset('iris')
print("Dataset Loaded Successfully!")
# 3. Basic Dataset Inspection
print("Shape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())
# 4. Dataset Information
print("\nDataset Information:")
df.info()
# 5. Statistical Summary
print("\nStatistical Summary:")
print(df.describe())
# 6. Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# 7. Scatter Plot

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='sepal_length',
    y='sepal_width',
    hue='species'
)

plt.title('Sepal Length vs Sepal Width')
plt.show()
# 8. Histograms
df.hist(figsize=(12,8))
plt.suptitle("Feature Distributions")
plt.show()
# 9. Box Plots
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Box Plot for Outlier Detection")
plt.show()
# 10. Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64','int64'])
plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()
# 11. Pair Plot
sns.pairplot(df, hue='species')
plt.show()
