import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the data
# Replace 'your_path' with the actual path to your data file
df = pd.read_csv('scaps_humidity_integrated_data.csv')

# Basic data preprocessing
# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Remove any rows with missing values if present
df = df.dropna()

# Check data types and convert if necessary
print("\nData types of columns:")
print(df.dtypes)

# Ensure numerical columns are properly formatted
numerical_columns = ['Temperature_C', 'Relative_Humidity', 'Exposure_Time', 'PCE', 'Jsc', 'FF', 'Voc']
for col in numerical_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Display basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})
# Temperature vs PCE with humidity color mapping
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Temperature_C'], df['PCE'], 
                     c=df['Relative_Humidity'], cmap='viridis', 
                     s=100, alpha=0.6)
colorbar = plt.colorbar(scatter)
colorbar.set_label('Relative Humidity (%)', size=14)
plt.xlabel('Temperature (°C)')
plt.ylabel('PCE (%)')
plt.title('Temperature vs PCE with Humidity Effect')
plt.tight_layout()
plt.show()

# Humidity vs PCE with temperature color mapping
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Relative_Humidity'], df['PCE'], 
                     c=df['Temperature_C'], cmap='plasma', 
                     s=100, alpha=0.6)
colorbar = plt.colorbar(scatter)
colorbar.set_label('Temperature (°C)', size=14)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('PCE (%)')
plt.title('Humidity vs PCE with Temperature Effect')
plt.tight_layout()
plt.show()

# Exposure Time vs PCE
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Exposure_Time'], df['PCE'], 
                     c=df['Relative_Humidity'], cmap='viridis', 
                     s=100, alpha=0.6)
colorbar = plt.colorbar(scatter)
colorbar.set_label('Relative Humidity (%)', size=14)
plt.xlabel('Exposure Time (hours)')
plt.ylabel('PCE (%)')
plt.title('PCE Degradation Over Time')
plt.tight_layout()
plt.show()

# Parameter Correlations Heatmap
plt.figure(figsize=(10, 10))
params = ['PCE', 'Jsc', 'FF', 'Voc', 'Temperature_C', 'Relative_Humidity', 'Exposure_Time']
correlation_matrix = df[params].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', annot_kws={'size': 12}, square=True)
plt.title('Parameter Correlations')
plt.tight_layout()
plt.show()

# Box plots for PCE distribution at different temperature ranges
plt.figure(figsize=(12, 12))
df['Temp_Range'] = pd.qcut(df['Temperature_C'], q=5)
sns.boxplot(x='Temp_Range', y='PCE', data=df)
plt.xlabel('Temperature Range')
plt.ylabel('PCE (%)')
plt.title('PCE Distribution Across Temperature Ranges')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Joint plot of Humidity vs Temperature with PCE as color
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Temperature_C'], df['Relative_Humidity'],
                     c=df['PCE'], cmap='viridis', s=100, alpha=0.6)
colorbar = plt.colorbar(scatter)
colorbar.set_label('PCE (%)', size=14)
plt.xlabel('Temperature (°C)')
plt.ylabel('Relative Humidity (%)')
plt.title('Temperature-Humidity-PCE Relationship')
plt.tight_layout()
plt.show()

# Predicted vs Actual PCE
# Prepare data for prediction
X = df[['Temperature_C', 'Relative_Humidity', 'Exposure_Time', 'Jsc', 'FF', 'Voc']]
y = df['PCE']

# Split the data and train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X)

# Plot Predicted vs Actual PCE
plt.figure(figsize=(12, 8))
plt.scatter(y, y_pred, alpha=0.5, s=100)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual PCE (%)')
plt.ylabel('Predicted PCE (%)')
plt.title(f'Predicted vs Actual PCE (R² = {r2_score(y, y_pred):.3f})')
plt.tight_layout()
plt.show()

# PCE over time for different temperature ranges
plt.figure(figsize=(12, 8))
for temp_range in pd.qcut(df['Temperature_C'], q=3).unique():
    mask = pd.qcut(df['Temperature_C'], q=3) == temp_range
    plt.plot(df[mask]['Exposure_Time'], df[mask]['PCE'], 'o-', 
             label=f'Temp: {temp_range}', alpha=0.6, linewidth=2, markersize=8)

plt.xlabel('Exposure Time (hours)', weight='bold', fontsize=14)
plt.ylabel('PCE (%)', weight='bold', fontsize=14)
plt.title('PCE Evolution Over Time for Different Temperature Ranges', 
          weight='bold', fontsize=16)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# PCE over time for different humidity ranges
plt.figure(figsize=(12, 8))
for hum_range in pd.qcut(df['Relative_Humidity'], q=3).unique():
    mask = pd.qcut(df['Relative_Humidity'], q=3) == hum_range
    plt.plot(df[mask]['Exposure_Time'], df[mask]['PCE'], 'o-', 
             label=f'RH: {hum_range}', alpha=0.6, linewidth=2, markersize=8)

plt.xlabel('Exposure Time (hours)', weight='bold', fontsize=14)
plt.ylabel('PCE (%)', weight='bold', fontsize=14)
plt.title('PCE Evolution Over Time for Different Humidity Ranges', 
          weight='bold', fontsize=16)
plt.legend(prop={'weight': 'bold', 'size': 12})
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
