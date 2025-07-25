import pandas as pd
import numpy as np
import pickle
import os
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data_path = "data/train.csv"
df = pd.read_csv(data_path)

# Drop rows with missing target
df.dropna(subset=["SalePrice"], inplace=True)

# Select numeric features only
numeric_df = df.select_dtypes(include=[np.number]).copy()
numeric_df.fillna(numeric_df.mean(), inplace=True)

X = numeric_df.drop("SalePrice", axis=1)
y = numeric_df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save artifacts
os.makedirs("outputs", exist_ok=True)
with open("outputs/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("outputs/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("outputs/feature_names.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")