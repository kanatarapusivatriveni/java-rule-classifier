import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---------------------------
# Read dataset
# ---------------------------
df = pd.read_csv("sensor_data.csv")
X = df[['Temperature', 'Vibration']]

# ---------------------------
# Classification-based (Isolation Forest)
# ---------------------------
iso = IsolationForest(contamination=0.1, random_state=42)
labels = iso.fit_predict(X)
df['Outlier'] = labels == -1
df['Status'] = np.where(df['Outlier'], 'Outlier', 'Normal')

# ---------------------------
# Display output
# ---------------------------
print("=== Classification-based Approach (Isolation Forest) ===\n")
print(df[['Temperature', 'Vibration', 'Status']])

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['Temperature'], df['Vibration'], c=df['Outlier'], cmap='coolwarm', s=80)
plt.title('Classification-based Approach (Isolation Forest)')
plt.xlabel('Temperature')
plt.ylabel('Vibration')
plt.show()
