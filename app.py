# ======================================================
# app.py
# AI 기반 정수장 pH 예측 및 약품 운전 판단 시스템
# ======================================================
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def judge_drinking(ph, tu, rc):
    """
    국내 먹는물 수질 기준 + 현실 운전 기준 반영
    """

    # ❌ 명확한 부적합
    if ph < 5.8 or ph > 8.5:
        return "❌ 음용 권고 안함 (pH 기준 초과)"

    if tu > 2.0:
        return "❌ 음용 권고 안함 (탁도 기준 초과)"

    # ⚠️ 주의 구간
    if rc < 0.2:
        return "⚠️ 끓여서 음용 권장 (소독력 부족)"

    if tu > 1.0:
        return "⚠️ 끓여서 음용 권장 (탁도 주의)"

    # ✅ 정상
    return "✅ 음용 가능"


# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="정수장 수질 예측 시스템", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "water_quality_full.db")
TABLE_NAME = "water_quality"
MODEL_PATH = os.path.join(BASE_DIR, "rf_ph_model_tuned.pkl")

# =========================
# 모델 로드
# =========================
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]


# =========================
# 약품 판단 로직 (개선 버전)
# =========================
def judge_chemical(ph, rc, tu):
    if ph > 8.0 or tu > 1.0:
        return "🔴 응집제 투입 증가 필요"
    elif ph < 6.5:
        return "🔴 알칼리제 투입 증가 필요"
    elif rc < 0.4:
        return "🟡 소독제 추가 필요"
    else:
        return "🟢 정상 또는 미세 조정 수준"


# =========================
# 탭 구성
# =========================
tab2, tab3, tab4= st.tabs(
    [
        "pH 예측 시연",
        "2026년 예측 분석",
        "음용 안전 판단",
    ]
)

# ======================================================
# TAB 2. pH 예측 시연
# ======================================================
with tab2:
    st.header("🧪 수질 입력 → pH 예측")

    col1, col2, col3 = st.columns(3)

    with col1:
        HR = st.slider("경도 (HR)", 10.0, 100.0, 40.0)
        BRO = st.slider("브롬산염 (BRO)", 0.0, 0.02, 0.005)
        SO = st.slider("황산이온 (SO)", 5.0, 50.0, 15.0)

    with col2:
        RE = st.slider("증발잔류물 (RE)", 50.0, 300.0, 150.0)
        AL = st.slider("알루미늄 (AL)", 0.0, 0.3, 0.05)
        TU = st.slider("탁도 (TU)", 0.0, 5.0, 0.3)

    with col3:
        NON = st.slider("질산성질소 (NON)", 0.0, 5.0, 1.0)
        CF = st.slider("클로로포름 (CF)", 0.0, 0.05, 0.01)
        RC = st.slider("잔류염소 (RC)", 0.0, 2.0, 0.6)

    if st.button("🔍 pH 예측"):
        input_data = pd.DataFrame(
            [
                {
                    "HR": HR,
                    "RE": RE,
                    "NON": NON,
                    "BRO": BRO,
                    "AL": AL,
                    "CF": CF,
                    "SO": SO,
                    "TU": TU,
                    "RC": RC,
                }
            ]
        )

        pred_ph = model.predict(input_data)[0]

        st.subheader(f"📌 예측 pH : {pred_ph:.2f}")
        st.info(judge_chemical(pred_ph, RC, TU))

        st.caption("※ 본 예측은 과거 학습된 수질 범위 내 상대적 변화에 기반합니다.")


# ======================================================
# TAB 3. 연도·월 선택 지역별 pH 예측
# ======================================================
with tab3:
    st.header("📈 연도·월 선택 지역별 pH 예측 (미래 예측)")

    # -------------------------
    # 데이터 로드
    # -------------------------
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # 전처리
    df = df.replace({"불검출": 0, "검출": 1, "적합": 1, "부적합": 0})

    for col in FEATURES + ["PH"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + ["PH"])

    # -------------------------
    # 🔮 예측 대상 연도 / 월 선택 (미래)
    # -------------------------
    col_y, col_m = st.columns(2)

    with col_y:
        pred_year = st.selectbox("📅 예측 대상 연도", [2026, 2027, 2028])

    with col_m:
        pred_month = st.selectbox("📅 예측 대상 월", list(range(1, 13)))


    base_df = df[
        (df["year"].astype(int) == 2025) & (df["month"].astype(int) == pred_month)
    ]

    if base_df.empty:
        st.warning("선택한 월의 2025년 데이터가 없어 연평균으로 대체합니다.")
        base_df = df[df["year"].astype(int) == 2025]

    # -------------------------
    # 지역별 예측
    # -------------------------
    results = []

    for region in base_df["region"].unique():
        region_df = base_df[base_df["region"] == region]
        X_mean = region_df[FEATURES].mean().to_frame().T

        pred_ph = model.predict(X_mean)[0]

        tu_val = X_mean["TU"].values[0]
        rc_val = X_mean["RC"].values[0]

        results.append(
            {
                "지역": region,
                "예측 pH": round(pred_ph, 2),
                "탁도(TU)": round(X_mean["TU"].values[0], 2),
                "잔류염소(RC)": round(X_mean["RC"].values[0], 2),
                "약품 판단": judge_chemical(
                    pred_ph, X_mean["RC"].values[0], X_mean["TU"].values[0]
                ),
                "🚰 음용 안전": judge_drinking(
                    pred_ph,
                    X_mean["TU"].values[0],  #
                    X_mean["RC"].values[0],  #
                ),
            }
        )

    result_df = pd.DataFrame(results).sort_values(by="예측 pH", ascending=False)

    # -------------------------
    # 결과 출력
    # -------------------------
    st.subheader(f"📊 {pred_year}년 {pred_month}월 지역별 pH 예측")
    st.dataframe(result_df, width="stretch")

    st.subheader("🚨 약품 사용 부담 증가 예상 TOP5")
    st.table(result_df.head(5))

    st.caption(
        "※ 본 예측은 2025년 동일 월 수질 조건을 기반으로 "
        f"{pred_year}년 {pred_month}월을 가정한 시나리오 예측입니다."
    )

# ======================================================
# TAB 4. 음용 안전 판단
# ======================================================
with tab4:
    st.header("수돗물 음용 안전 판단")

    st.markdown(
        """
    **판단 기준 (요약)**  
    - pH: 6.5 ~ 8.5  
    - 탁도(TU): ≤ 1.0 NTU  
    - 잔류염소(RC): ≥ 0.4 mg/L
    """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        ph_val = st.slider("예측 pH", 5.0, 9.5, 7.2, key="drink_ph")

    with col2:
        tu_val = st.slider("탁도 (TU)", 0.0, 5.0, 0.3, key="drink_tu")

    with col3:
        rc_val = st.slider("잔류염소 (RC)", 0.0, 2.0, 0.6, key="drink_rc")

    if st.button("음용 가능 여부 판단"):
        if 6.5 <= ph_val <= 8.5 and tu_val <= 1.0 and rc_val >= 0.4:
            st.success("🟢 음용 가능 (기준 충족)")
        else:
            st.error("🔴 음용 권고하지 않음")
            st.warning(judge_chemical(ph_val, rc_val, tu_val))

        st.caption("※ 본 판단은 법적 판정이 아닌 시뮬레이션 기반 참고용입니다.")

