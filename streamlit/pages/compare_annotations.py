import streamlit as st
import os 
import json
import glob
import cv2
from PIL import Image
from utils.streamlit_data_loader import draw_annotations

# Ensemble를 위해 inference 이후 생성된 복수의 어노테이션(csv) 파일을 비교하는 뷰어 (test 이미지 기준)

def main():
    st.title("영수증 어노테이션 비교 뷰어")
    
    nation_dict = {
        '베트남어': 'vietnamese_receipt',
        '태국어': 'thai_receipt',
        '중국어': 'chinese_receipt',
        '일본어': 'japanese_receipt',
    }

    # 기본 경로 설정
    base_dir = "../data" # 데이터셋 경로
    csv_dir = "../prediction" # 어노테이션(csv) 폴더 경로 
    
    # 사이드바에 언어 선택 옵션 추가
    with st.sidebar:
        st.header("설정")
        selected_lang = st.selectbox(
            "언어 선택",
            options=list(nation_dict.keys())
        )

    # 이미지 디렉토리 설정 (test 고정)
    img_dir = os.path.join(base_dir, nation_dict[selected_lang], 'img', 'test')
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.error("이미지 파일을 찾을 수 없습니다.")
        return

    # prediction 폴더의 모든 CSV 파일 로드
    csv_files = glob.glob(csv_dir + '/*.csv')
    
    # 모든 CSV 파일의 어노테이션을 미리 로드
    all_annotations = {}
    for csv_path in csv_files:
        try:
            with open(csv_path) as f:
                data = json.load(f)
                all_annotations[os.path.basename(csv_path)] = data['images']
        except Exception as e:
            st.error(f"어노테이션 로드 실패 ({os.path.basename(csv_path)}): {e}")

    # 중앙에 이미지 선택 컨트롤
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        selected_image = st.selectbox("이미지 선택", image_files)

    if selected_image:
        # 이미지 로드
        img_path = os.path.join(img_dir, selected_image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지가 가로로 되어있다면 세로로 회전
        if image.shape[1] > image.shape[0]:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # PIL Image로 변환
        image = Image.fromarray(image)

        # 4열로 표시
        cols = st.columns(4)

        # 원본 이미지
        with cols[0]:
    
            st.image(image, caption="원본 이미지", use_column_width=True)
            with st.expander("이미지 정보", expanded=False):
                st.write("이미지 크기:", image.size)
                st.write("이미지 모드:", image.mode)

        # 각 CSV 파일의 어노테이션 표시
        for idx, csv_path in enumerate(csv_files, 1):
            col_idx = idx % 4  # 4열로 나누기
            
            # 새로운 행이 필요한 경우
            if col_idx == 0:
                cols = st.columns(4)
            
            csv_name = os.path.basename(csv_path)
            annotation = all_annotations[csv_name].get(selected_image, {})
            
            with cols[col_idx]:
        
                if annotation:
                    annotated_image = draw_annotations(image, annotation)
                    st.image(annotated_image, caption=csv_name, use_column_width=True)
                    
                    with st.expander("어노테이션 정보", expanded=False):
                        st.write("CSV 파일:", csv_name)
                        st.write("이미지 크기:", image.size)
                        st.write("이미지 모드:", image.mode)
                        st.write("단어 수:", len(annotation.get('words', {})))
                else:
                    st.image(image, caption=f"어노테이션 없음 ({csv_name})", use_column_width=True)
                    with st.expander("어노테이션 정보", expanded=False):
                        st.write("CSV 파일:", csv_name)
                        st.write("어노테이션 없음")
if __name__ == "__main__":
    main()