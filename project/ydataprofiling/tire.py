import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. 환경 설정
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['Defective (불량)', 'Good (정상)'] 
MODEL_PATH = './model/tire_classification_model.h5'

# 2. 모델 로드 함수 (Lambda 레이어 에러 해결 포함)
@st.cache_resource
def load_tire_model():
    try:
        # MobileNetV2의 전처리 함수를 명시적으로 매핑
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        
        # 모델 로드 시 custom_objects를 전달하여 'preprocess_input' 또는 'function' 에러 방지
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

# 3. UI 디자인
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
    st.error(f"⚠️ 모델 파일을 로드할 수 없습니다. 파일명과 경로를 확인하세요.\n(에러 내용: {model})")
    st.info(f"현재 작업 디렉토리: {os.getcwd()}")
else:
    # 4. 이미지 업로드 섹션
    uploaded_file = st.file_uploader("타이어 측면 사진을 선택하세요...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # 원본 이미지 표시
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        # 5. 예측 수행
        with col2:
            with st.spinner('AI 분석 중...'):
                img = image.convert('RGB')
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가
                
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