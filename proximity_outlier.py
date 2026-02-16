import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# ---------------------------
# Read dataset
# ---------------------------
df = pd.read_csv("sensor_data.csv")
X = df[['Temperature', 'Vibration']]

# ---------------------------
# Proximity-based (Local Outlier Factor)
# ---------------------------
lof = LocalOutlierFactor(n_neighbors=5)
labels = lof.fit_predict(X)
df['Outlier'] = labels == -1
df['Status'] = np.where(df['Outlier'], 'Outlier', 'Normal')

# ---------------------------
# Display output
# ---------------------------
print("=== Proximity-based Approach (LOF) ===\n")
print(df[['Temperature', 'Vibration', 'Status']])

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['Temperature'], df['Vibration'], c=df['Outlier'], cmap='coolwarm', s=80)
plt.title('Proximity-based Approach (LOF)')
plt.xlabel('Temperature')
plt.ylabel('Vibration')
plt.show()
