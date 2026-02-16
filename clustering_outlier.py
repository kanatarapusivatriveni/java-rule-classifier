import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ---------------------------
# Read dataset
# ---------------------------
df = pd.read_csv("sensor_data.csv")
X = df[['Temperature', 'Vibration']]

# ---------------------------
# Clustering-based (DBSCAN)
# ---------------------------
db = DBSCAN(eps=3, min_samples=3)
labels = db.fit_predict(X)
df['Cluster'] = labels
df['Outlier'] = df['Cluster'] == -1
df['Status'] = np.where(df['Outlier'], 'Outlier', 'Normal')

# ---------------------------
# Display output
# ---------------------------
print("=== Clustering-based Approach (DBSCAN) ===\n")
print(df[['Temperature', 'Vibration', 'Cluster', 'Status']])

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['Temperature'], df['Vibration'], c=df['Cluster'], cmap='tab10', s=80)
plt.scatter(df[df['Outlier']]['Temperature'], df[df['Outlier']]['Vibration'], 
            color='red', edgecolors='black', s=100, label='Outliers')
plt.title('Clustering-based Approach (DBSCAN)')
plt.xlabel('Temperature')
plt.ylabel('Vibration')
plt.legend()
plt.show()
