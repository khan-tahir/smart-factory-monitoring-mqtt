from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Generate dummy data
X_train = np.random.rand(1000, 3) * [100, 1, 150]  # Simulated sensor data (Temp, Vib, Pressure)
y_train = np.random.choice([0, 1], size=1000)  # 0 = Normal, 1 = Maintenance Needed

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "predictive_maintenance_model.pkl")
