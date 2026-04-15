import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import ollama
from PIL import Image
import os
# 전처리 함수를 상단에서 미리 임포트합니다.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 초기 설정 ---
MODEL_PATH = 'aircraft_defect_model_v1.h5' 
IMG_HEIGHT, IMG_WIDTH = 180, 180 

# 클래스 이름
class_names = ['crack', 'dent', 'missing-head', 'paint-off', 'scratch']

# --- 핵심 기능 함수 정의 ---

@st.cache_resource
def load_defect_model():
    """모델 로드 및 Lambda 레이어(preprocess_input) 등록"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None
    
    try:
        # custom_objects를 통해 Lambda 레이어에 쓰인 함수를 연결합니다.
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'preprocess_input': preprocess_input}
        )
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

def classify_defect(img_file, model):
    img = Image.open(img_file)
    
    img = img.resize((IMG_HEIGHT, IMG_WIDTH)) 
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    
    score = predictions[0]
    class_idx = np.argmax(score)
    confidence = 100 * score[class_idx]
    
    return class_names[class_idx], confidence

def generate_maintenance_report(defect_type, confidence):
    """LLM(Gemma2)을 통한 정비 리포트 생성"""
    prompt = f"""
    당신은 항공기 정비 전문가입니다. 
    최신 AI 모델이 항공기 표면 사진을 분석하여 다음 결함을 찾아냈습니다.
    
    분석 결과:
    - 결함 종류: {defect_type}
    - AI 신뢰도: {confidence:.2f}%
    
    항공기 안전을 위해 이 결함에 대한 '정비 분석 리포트'를 한국어로 작성해주세요.
    리포트에는 다음 내용이 포함되어야 합니다.
    1. 해당 결함의 정의와 항공기 구조에 미칠 수 있는 영향 (안전 측면)
    2. 이 결함이 발견되었을 때 정비사가 즉시 수행해야 할 표준 조치 사항
    3. AI 신뢰도가 낮은 경우(80% 미만) 육안 재확인 필요성 언급
    """
    
    with st.spinner('정비 전문가 AI가 리포트를 작성 중입니다... (Ollama 실행 중)'):
        try:
            response = ollama.chat(model='gemma2', messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            return f"LLM 호출 중 오류가 발생했습니다: {e}. Ollama가 실행 중인지 확인하세요."

# --- Streamlit 웹 UI 구성 ---

st.set_page_config(page_title="항공기 결함 진단 AI", page_icon="✈️", layout="wide")
st.title("✈️ 항공기 결함 진단 어시스턴트")
st.markdown("---")

# 사이드바
st.sidebar.header("시스템 정보")
st.sidebar.info("i7-4770(CPU) / 16GB RAM 최적화")
defect_model = load_defect_model()

# 메인 화면
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("이미지 업로드")
    uploaded_file = st.file_uploader("항공기 표면 사진을 올려주세요.", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption='업로드된 이미지', use_container_width=True)

with col2:
    st.header("AI 진단 결과")
    
    if uploaded_file is not None and defect_model is not None:
        with st.spinner('분석 중...'):
            defect_type, confidence = classify_defect(uploaded_file, defect_model)
        
        st.success("✅ 분석 완료")
        
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3>진단 결과: <span style="color:#e74c3c;">{defect_type.upper()}</span></h3>
            <p>AI 신뢰도: <strong>{confidence:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence < 80:
             st.warning("⚠️ 신뢰도가 낮습니다. 반드시 숙련된 정비사의 육안 확인이 필요합니다.")

        if st.button('정비 리포트 생성 요청'):
            report = generate_maintenance_report(defect_type, confidence)
            st.markdown("### 📄 정비 전문가 AI 리포트")
            st.write(report)

    elif defect_model is None:
        st.error("모델 로드 실패. MODEL_PATH를 확인하세요.")
    else:
        st.info("사진을 업로드하면 AI 분석이 시작됩니다.")

st.markdown("---")
st.caption("© 2026 Aircraft Defect Detection System (Gemma2 based)")