import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# 1. تحميل عينة من مجموعة بيانات CICIoT2023
# (يمكنك استخدام رابط البيانات المباشر أو الملف المحلي)
df = pd.read_csv("CICIoT2023_sample.csv")

# 2. تنظيف البيانات والتعامل مع القيم المفقودة
df = df.dropna()

# 3. فصل الخصائص (Features) عن الهدف (Label)
X = df.drop(columns=['label'])
y = df['label']

# 4. اختيار الخصائص الخفيفة (Feature Selection using Random Forest Importance)
selector = RandomForestClassifier(n_estimators=50, random_state=42)
selector.fit(X, y)

# تحديد أفضل 15 خاصية مؤثرة تقليلاً للجهد الحسابي
importances = selector.feature_importances_
top_indices = np.argsort(importances)[-15:]
selected_features = X.columns[top_indices]

print("الخصائص الـ 15 المختارة للنموذج الخفيف:")
print(selected_features)

# 5. تدريب النموذج الخفيف (LightGBM)
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# 6. تقييم الأداء
y_pred = model.predict(X_test)
print(f"\nدقة النموذج الخفيف (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
