# ======================================================
# 03_random_forest_regression.py
# RandomForest 회귀로 pH 예측 + 모델 저장
# ======================================================

import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 설정
# =========================
DB_NAME = "water_quality_full.db"
TABLE_NAME = "water_quality"

#  공정 관점 반영한 최종 변수
FEATURES = [
    'HR',   # 경도
    'RE',   # 증발잔류물
    'NON',  # 질산성질소
    'BRO',  # 브롬산염
    'AL',   # 알루미늄
    'CF',   # 클로로포름
    'SO',   # 황산이온
    'TU',   # 탁도 (응집 판단)
    'RC'    # 잔류염소 (소독 운전 핵심)
]

TARGET = 'PH'
MODEL_PATH = "rf_ph_model.pkl"
RANDOM_STATE = 42

# =========================
# 1. DB → DataFrame
# =========================
conn = sqlite3.connect(DB_NAME)
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
conn.close()

print("총 데이터 수:", len(df))

# =========================
# 2. 전처리
# =========================
# 문자열 → 수치 치환
df = df.replace({
    "불검출": 0,
    "검출": 1,
    "적합": 1,
    "부적합": 0,
    "일반세균": 1
})

# 숫자 변환
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 결측치 제거
df = df.dropna(subset=FEATURES + [TARGET])

print("모델 학습 데이터 수:", len(df))

# =========================
# 3. 학습 / 테스트 분리
# =========================
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# =========================
# 4. RandomForest 학습
# =========================
rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =========================
# 5. 예측
# =========================
y_pred = rf.predict(X_test)

# =========================
# 6. 성능 평가
# =========================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n RandomForest 회귀 성능")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# =========================
# 7. Feature Importance
# =========================
importances = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\n🔍 Feature Importance")
print(importances)

# =========================
# 8. 모델 저장
# =========================
joblib.dump({
    "model": rf,
    "features": FEATURES,
    "target": TARGET
}, MODEL_PATH)

print(f"\n✅ 모델 저장 완료 → {MODEL_PATH}")

# =========================
# 9. 중요도 시각화
# =========================
plt.figure(figsize=(9, 5))
importances.plot(kind="bar")
plt.title("RandomForest Feature Importance (pH 예측)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
