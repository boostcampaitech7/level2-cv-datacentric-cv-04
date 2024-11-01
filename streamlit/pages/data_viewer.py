import streamlit as st
from utils.streamlit_data_loader import load_image_and_annotation, draw_annotations
def main():
    st.title("영수증 이미지 뷰어")
    

    nation_dict = {
        '베트남어': 'vietnamese_receipt',
        '태국어': 'thai_receipt',
        '중국어': 'chinese_receipt',
        '일본어': 'japanese_receipt',
    }

    # 기본 경로 설정 (실제 경로로 수정 필요)
    base_dir = "../data"  # 예시 경로
    
    # 사이드바에 언어 선택 옵션 추가
    with st.sidebar:
        st.header("설정")
        selected_lang = st.selectbox(
            "언어 선택",
            options=list(nation_dict.keys())
        )
        
        # train/test 선택
        split = st.radio(
            "데이터셋 선택",
            options=['train', 'test'],
            horizontal=True
        )
    
    # 이미지와 어노테이션 로드
    image, annotations = load_image_and_annotation(base_dir, selected_lang, split, nation_dict)
    
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # 원본 이미지 표시
            st.subheader("원본 이미지")
            st.image(image, caption="원본 이미지", use_column_width=True)
        
        with col2:
            # 어노테이션이 표시된 이미지
            st.subheader("어노테이션 결과")
            annotated_image = draw_annotations(image, annotations)
            st.image(annotated_image, caption="어노테이션 결과", use_column_width=True)
        
        # 이미지 정보 표시
        with st.expander("이미지 정보", expanded=False):
            st.write("이미지 크기:", image.size)
            st.write("이미지 모드:", image.mode)
            if annotations:
                st.write("단어 수:", len(annotations.get('words', {})))

if __name__ == "__main__":
    main()