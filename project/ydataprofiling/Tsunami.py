import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # RandomForest 회귀 임포트
from sklearn.metrics import mean_squared_error, r2_score

# --- [한글 폰트 설정] ---
plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False 

# --- [경로 및 폴더 설정] ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_path, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 1. 데이터 로드 및 전처리
df = pd.read_csv('dataset/Tsunami.csv')
cols = ['causeCode', 'eqMagnitude', 'numRunups', 'warningStatusId', 'oceanicTsunami', 'tsIntensity']
data = df[cols].dropna(subset=['tsIntensity']).copy()

data['eqMagnitude'] = data['eqMagnitude'].fillna(data['eqMagnitude'].median())
data['numRunups'] = data['numRunups'].fillna(0)
data['warningStatusId'] = data['warningStatusId'].fillna(0)
data['oceanicTsunami'] = data['oceanicTsunami'].astype(float)

# 2. 데이터 분할 및 스케일링
X = data.drop(['tsIntensity'], axis=1)
y = data['tsIntensity'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. 메인 모델을 RandomForestRegressor로 설정 ---
main_model = RandomForestRegressor(n_estimators=100, random_state=42)
main_model.fit(X_train_scaled, y_train)

# --- [Streamlit UI] ---
st.title("쓰나미 강도 수치 예측 시스템 (Random Forest)")
st.write("본 시스템은 **Random Forest Regressor**를 사용하여 예상 강도 수치를 실시간으로 예측합니다.")

# 사이드바 입력
st.sidebar.header("실시간 데이터 입력")
cause_options = {
    "지진 (Earthquake)": 1.0,
    "원인 미상 (Unknown)": 0.0,
    "해저 산사태 (Landslide)": 3.0,
    "화산 폭발 (Volcanic Eruption)": 6.0,
    "기타 요인 (Meteorological 등)": 9.0
}
selected_cause_name = st.sidebar.selectbox("발생 원인 선택", list(cause_options.keys()))
in_cause = cause_options[selected_cause_name]
in_mag = st.sidebar.slider("지진 규모", 4.0, 9.0, 8.5)
in_run = st.sidebar.number_input("소상 횟수 (Runups)", value=20.0)
in_warn = st.sidebar.selectbox("경보 상태 레벨", [0, 1, 2, 3, 4])
in_ocean = st.sidebar.checkbox("해양 쓰나미 발생 여부", value=True)

# 예측 수행
new_event = pd.DataFrame([[in_cause, in_mag, in_run, in_warn, float(in_ocean)]], columns=X.columns)
new_event_scaled = scaler.transform(new_event)
predicted_intensity = main_model.predict(new_event_scaled)[0]

# --- 결과 시각화 ---
st.divider()
st.subheader(f"{selected_cause_name} 기반 강도 예측 결과")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.write("### 분석 결과")
    if predicted_intensity > 3.0:
        st.error(f"예상 강도: **{predicted_intensity:.2f}** (위험)")
    elif predicted_intensity > 1.5:
        st.warning(f"예상 강도: **{predicted_intensity:.2f}** (주의)")
    else:
        st.success(f"예상 강도: **{predicted_intensity:.2f}** (안전)")
    
    st.metric("예측된 쓰나미 강도", f"{predicted_intensity:.2f}")

with col2:
    st.write("### 모델 예측 분포 (Actual vs Predicted)")
    y_test_pred = main_model.predict(X_test_scaled)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5, color='forestgreen', ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)


st.info(f"**AI 분석:** 현재 입력된 조건에서 RandomForest Model은 쓰나미 강도를 약 **{predicted_intensity:.2f}**로 추정")