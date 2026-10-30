import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# 1. Load sample dataset from CICIoT2023
data_path = 'CICIoT2023_sample.csv'

try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset file not found. Generating sample data structure...")
    np.random.seed(42)
    feature_names = [f'feature_{i}' for i in range(80)]
    X_dummy = np.random.rand(1000, 80)
    y_dummy = np.random.choice(['Benign', 'DDoS-TCP_Flood', 'DDoS-UDP_Flood'], size=1000)
    df = pd.DataFrame(X_dummy, columns=feature_names)
    df['label'] = y_dummy

# 2. Preprocessing & Null Imputation
df = df.dropna()

# 3. Separate Features and Label
X = df.drop(columns=['label'])
y = df['label']

# 4. Feature Selection using Random Forest Importance
selector = RandomForestClassifier(n_estimators=50, random_state=42)
selector.fit(X, y)

# Extract Top 15 Features
importances = selector.feature_importances_
top_indices = np.argsort(importances)[-15:]
selected_features = X.columns[top_indices]

print("Selected Top 15 Features:")
print(selected_features.tolist())

# 5. Train Lightweight LightGBM Classifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
model.fit(X_train, y_train)

# 6. Evaluate Model Performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\nLightGBM Model Accuracy: {acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
