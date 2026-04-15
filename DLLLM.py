import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import ollama
from PIL import Image
import os

# --- 초기 설정 ---
# 1. 이전 단계에서 학습시켜 저장한 모델 파일 (.h5) 경로
MODEL_PATH = 'aircraft_defect_model_v1.h5' 
IMG_HEIGHT, IMG_WIDTH = 180, 180

# 2. 클래스 이름 (Confusion Matrix 폴더 순서와 동일하게 유지)
class_names = ['crack', 'dent', 'missing-head', 'paint-off', 'scratch']

# --- 핵심 기능 함수 정의 ---

# [기능 1] 딥러닝 모델 로드 (Streamlit 캐시 활용으로 속도 향상)
@st.cache_resource
def load_defect_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n학습된 모델을 먼저 저장해주세요.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# [기능 2] 업로드된 이미지 분석 및 분류
def classify_defect(img_file, model):
    # 이미지 전처리
    img = Image.open(img_file)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # 배치 차원 추가

    # 예측
    predictions = model.predict(img_array)
    
    # Keras.preprocessing.image_dataset_from_directory를 썼다면
    # 전처리로 preprocess_input이나 rescaling을 모델 내부에 넣었는지 확인해야 합니다.
    # 만약 모델 외부에 있었다면 여기서 img_array = img_array / 255.0 등을 해줘야 합니다.

    # Softmax로 확률 반환 (만약 모델 마지막 층에 Softmax가 없다면 적용)
    score = tf.nn.softmax(predictions[0])
    
    class_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    return class_names[class_idx], confidence

# [기능 3] LLM (Gemma2)에게 정비 리포트 생성 요청
def generate_maintenance_report(defect_type, confidence):
    # Gemma2에게 줄 프롬프트 설계
    # 모델의 신뢰도(Confidence) 정보도 함께 주어, LLM이 더 정확한 답변을 유도합니다.
    prompt = f"""
    당신은 항공기 정비 전문가입니다. 
    최신 AI 모델이 항공기 표면 사진을 분석하여 다음 결함을 찾아냈습니다.
    
    분석 결과:
    - 결함 종류: {defect_type}
    - AI 신뢰도: {confidence:.2f}%
    
    항공기 안전을 위해 이 결함에 대한 '정비 분석 리포트'를 한국어로 작성해주세요.
    리포트에는 다음 내용이 포함되어야 합니다.
    1. 해당 결함의 정의와 항공기 구조에 미칠 수 있는 영향 (안전 측면)
    2. 이 결함이 발견되었을 때 정비사가 즉시 수행해야 할 표준 조치 사항 (예: 비파괴 검사, 부품 교체 등)
    3. AI 신뢰도가 낮은 경우(예: 80% 미만)라면, 육안 재확인 필요성을 언급해주세요.
    """
    
    # Ollama를 통해 Gemma2 모델 호출
    # 로컬 CPU 환경이므로 응답까지 수 초 ~ 수 분 소요될 수 있습니다.
    with st.spinner('정비 전문가 AI가 리포트를 작성 중입니다... (CPU 연산 중)'):
        response = ollama.chat(model='gemma2', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
    return response['message']['content']

# --- Streamlit 웹 UI 구성 ---

# 1. 페이지 제목 및 로고
st.set_page_config(page_title="항공기 결함 진단 AI", page_icon="✈️", layout="wide")
st.title("✈️ 항공기 결함 진단 어시스턴트 System")
st.markdown("---")

# 2. 사이드바 (정보 및 모델 로드)
st.sidebar.header("시스템 정보")
st.sidebar.info("i7-4770(CPU) 환경 최적화")
st.sidebar.markdown(f"**딥러닝 모델:** MobileNetV2 (Acc: 93.3%)")
st.sidebar.markdown(f"**LLM 모델:** Gemma2 (via Ollama)")
defect_model = load_defect_model()

# 3. 메인 화면 구성 (레이아웃 분할)
col1, col2 = st.columns([1, 1.5]) # 왼쪽(사진), 오른쪽(분석 결과)

with col1:
    st.header("이미지 업로드")
    uploaded_file = st.file_uploader("항공기 표면 사진을 올려주세요.", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption='업로드된 이미지', use_column_width=True)

with col2:
    st.header("AI 진단 결과")
    
    if uploaded_file is not None and defect_model is not None:
        # Step 1: 딥러닝 분석
        with st.spinner('딥러닝 모델이 사진을 분석 중입니다...'):
            defect_type, confidence = classify_defect(uploaded_file, defect_model)
        
        # 분석 결과 출력
        st.success("✅ 딥러닝 분석 완료!")
        
        # 커스텀 스타일로 결과 표시
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0;">진단 결과</h3>
            <p style="font-size: 1.2rem; margin: 5px 0;">분석된 결함: <strong>{defect_type.upper()}</strong></p>
            <p style="font-size: 1rem; margin: 5px 0;">AI 신뢰도: <strong>{confidence:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 신뢰도에 따른 경고 메시지 (선택 사항)
        if confidence < 80:
             st.warning(f"⚠️ AI 신뢰도가 {confidence:.2f}%로 낮습니다. 전문가의 육안 재확인이 강력히 권고됩니다.")

        # Step 2: LLM 리포트 생성 버튼
        if st.button('정비 리포트 생성 요청'):
            report = generate_maintenance_report(defect_type, confidence)
            
            st.markdown("### 📄 정비 전문가 AI 리포트")
            st.markdown(report) # LLM의 답변 출력

    elif defect_model is None:
        st.error("딥러닝 모델 로드 실패. 사이드바의 정보를 확인해주세요.")
    else:
        st.info("왼쪽 화면에서 항공기 사진을 업로드하면 분석이 시작됩니다.")

# --- 페이지 하단 ---
st.markdown("---")
st.caption("본 시스템은 보조 도구이며, 최종 정비 판단은 자격을 갖춘 정비사에 의해 이루어져야 합니다.")