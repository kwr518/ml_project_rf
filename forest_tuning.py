# ======================================================
# 04_random_forest_tuning.py
# RandomForest 하이퍼파라미터 튜닝
# ======================================================

import sqlite3
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 설정
# =========================
DB_NAME = "water_quality_full.db"
TABLE_NAME = "water_quality"

FEATURES = [
    'HR', 'RE', 'NON', 'BRO', 'AL', 'CF', 'SO',
    'TU',   # 탁도
    'RC'    # 잔류염소
]
TARGET = "PH"

RANDOM_STATE = 42
MODEL_PATH = "rf_ph_model_tuned1.pkl"

# =========================
# 1. DB 로드
# =========================
conn = sqlite3.connect(DB_NAME)
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
conn.close()

print("총 데이터 수:", len(df))

# =========================
# 2. 전처리
# =========================
df = df.replace({
    "불검출": 0,
    "검출": 1,
    "적합": 1,
    "부적합": 0,
    "일반세균": 1
})

for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES + [TARGET])

print("모델링 데이터 수:", len(df))

# =========================
# 3. Train / Test 분리
# =========================
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# =========================
# 4. 튜닝 파라미터 공간
# =========================
param_dist = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [None, 8, 12, 16, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.5, 0.8]
}

base_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# =========================
# 5. RandomizedSearchCV
# =========================
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,   # 시간 여유 없으면 15
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("\n 하이퍼파라미터 탐색 시작...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\n 최적 파라미터")
for k, v in search.best_params_.items():
    print(f"{k}: {v}")

# =========================
# 6. 성능 평가 (튜닝 모델)
# =========================
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n 튜닝된 RandomForest 성능")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =========================
# 7. Feature Importance
# =========================
importances = pd.Series(
    best_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n🔍 Feature Importance (Tuned)")
print(importances)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Tuned RandomForest Feature Importance (pH 예측)")
plt.ylabel("Importance")
plt.xlabel("수질 항목")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# =========================
# 8. 모델 저장
# =========================
joblib.dump(
    {
        "model": best_model,
        "features": FEATURES,
        "target": TARGET
    },
    MODEL_PATH
)

print(f"\n 튜닝 모델 저장 완료 → {MODEL_PATH}")
