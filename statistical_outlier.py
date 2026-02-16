import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Read dataset
# ---------------------------
df = pd.read_csv("sensor_data.csv")

# ---------------------------
# Statistical Approach (Z-Score)
# ---------------------------
def detect_outliers_zscore(series):
    z = np.abs((series - series.mean()) / series.std())
    return z > 3  # threshold

df['Outlier'] = detect_outliers_zscore(df['Temperature']) | detect_outliers_zscore(df['Vibration'])
df['Status'] = np.where(df['Outlier'], 'Outlier', 'Normal')

# ---------------------------
# Display output
# ---------------------------
print("=== Statistical Approach (Z-Score) ===\n")
print(df[['Temperature', 'Vibration', 'Status']])

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['Temperature'], df['Vibration'], c=df['Outlier'], cmap='coolwarm', s=80)
plt.title('Statistical Approach (Z-Score)')
plt.xlabel('Temperature')
plt.ylabel('Vibration')
plt.show()
