import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- 1. 환경 설정 ---
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['Defective (불량)', 'Good (정상)'] 

# [수정] 절대 경로를 사용하여 배포 환경에서도 모델을 정확히 찾도록 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'tire_classification_model.h5')

# --- 2. 모델 로드 함수 ---
@st.cache_resource
def load_tire_model():
    # 먼저 파일이 실제로 존재하는지 확인
    if not os.path.exists(MODEL_PATH):
        return f"파일을 찾을 수 없습니다: {MODEL_PATH}"
    
    try:
        # MobileNetV2 전처리 함수 매핑
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        
        # 모델 로드
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={
                'preprocess_input': preprocess_input,
                'function': preprocess_input
            },
            compile=False
        )
        return model
    except Exception as e:
        return str(e)

# --- 3. UI 디자인 ---
st.set_page_config(page_title="Tire Guard AI", page_icon="🚗", layout="centered")

st.title("타이어 결함 탐지 시스템")
st.markdown("""
이 시스템은 딥러닝(CNN)을 사용하여 타이어의 상태를 분석합니다.  
사진을 업로드하면 실시간으로 결함 여부를 판단합니다.
""")

# 모델 로드 실행
model = load_tire_model()

# 모델 로드 실패 시 안내
if isinstance(model, str):
    st.error(f"⚠️ 모델 파일을 로드할 수 없습니다. 경로를 확인하세요.")
    st.warning(f"에러 내용: {model}")
    st.info(f"현재 작업 디렉토리: {os.getcwd()}")
    # 디버깅을 위해 model 폴더 내용 출력
    if os.path.exists(os.path.join(BASE_DIR, 'model')):
        st.write("model 폴더 내 파일들:", os.listdir(os.path.join(BASE_DIR, 'model')))
else:
    # --- 4. 이미지 업로드 섹션 ---
    uploaded_file = st.file_uploader("타이어 측면 사진을 선택하세요...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # 원본 이미지 표시
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        # --- 5. 예측 수행 ---
        with col2:
            with st.spinner('AI 분석 중...'):
                img = image.convert('RGB')
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 예측
                predictions = model.predict(img_array)
                
                # 결과 해석
                result_index = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                label = CLASS_NAMES[result_index]

                # 결과 출력 가시화
                st.subheader("진단 결과")
                if result_index == 0: # Defective
                    st.error(f"### {label}")
                else: # Good
                    st.success(f"### {label}")
                
                st.metric(label="분석 신뢰도", value=f"{confidence:.2f}%")
                
                # 확률 분포 그래프
                chart_data = {name: float(prob) for name, prob in zip(CLASS_NAMES, predictions[0])}
                st.bar_chart(chart_data)