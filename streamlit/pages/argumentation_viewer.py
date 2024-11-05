import streamlit as st
import numpy as np
from PIL import Image
from utils.streamlit_data_loader import load_image_and_annotation
from utils.preprocessing import resize_img, adjust_height, improved_background_removal_rgb,rotate_img, crop_img, detect_receipt, lab_processing, hsv_processing, color_processing, preprocess_receipt
from utils.streamlit_data_loader import draw_boxes, get_preprocessing_stats
import cv2

def main():
    # 페이지 설정을 가장 먼저 해야 합니다
    st.set_page_config(layout="wide")
    
    st.title("전처리 과정 시각화")
    
    nation_dict = {
        '베트남어': 'vietnamese_receipt',
        '태국어': 'thai_receipt',
        '중국어': 'chinese_receipt',
        '일본어': 'japanese_receipt',
    }

    base_dir = "../data"
    
    with st.sidebar:
        st.header("설정")
        selected_lang = st.selectbox(
            "언어 선택",
            options=list(nation_dict.keys())
        )
        
        split = st.radio(
            "데이터셋 선택",
            options=['train', 'test'],
            horizontal=True
        )

        # 전처리 파라미터
        st.header("전처리 파라미터")
        image_size = st.slider("목표 이미지 크기", 512, 2048, 1024, 256)
        
        # 크롭 사이즈는 이미지 크기보다 작아야 함
        max_crop_size = min(image_size, 1024)
        crop_size = st.slider("크롭 크기", 512, max_crop_size, min(512, max_crop_size), 128)
        
        # 나머지 파라미터들
        height_ratio = st.slider("높이 조정 범위", 0.0, 0.4, 0.2, 0.1)
        angle_range = st.slider("회전 각도 범위", 0, 30, 10)

        # 파라미터 경고
        if crop_size > image_size:
            st.warning("크롭 크기가 리사이즈된 이미지 크기보다 큽니다. 자동으로 조정됩니다.")
            crop_size = image_size

    # 이미지와 어노테이션 로드
    image, annotations = load_image_and_annotation(base_dir, selected_lang, split, nation_dict)
    
    # 이미지와 어노테이션 로드 부분 이후
    if image is not None:
        vertices = []
        labels = []
        
        # # annotations 구조 처리 방식 수정
        # if isinstance(annotations, dict):
        #     if 'words' in annotations:  # train 데이터 형식
        #         for word_info in annotations['words'].values():
        #             if isinstance(word_info, dict) and 'points' in word_info:
        #                 vertices.append(np.array(word_info['points']).reshape(-1))
        #                 labels.append(1)  # 텍스트 영역은 1로 표시
        #     elif 'annotations' in annotations:  # test 데이터 형식
        #         for anno in annotations['annotations']:
        #             if 'points' in anno:
        #                 vertices.append(np.array(anno['points']).reshape(-1))
        #                 labels.append(1)

        vertices = np.array(vertices) if vertices else np.array([])
        labels = np.array(labels)

        # if len(vertices) == 0:
        #     st.error("이미지에서 텍스트 영역을 찾을 수 없습니다.")
        #     return
        
        # 전처리 단계별 처리
        steps = []
        
        # 원본
        steps.append(("원본 이미지", image, vertices))
        
        converted_lab_image = lab_processing(image)
        steps.append(("LAB 변환 후", converted_lab_image, vertices))
        
        converted_hsv_image = hsv_processing(image)
        steps.append(("HSV 변환 후", converted_hsv_image, vertices))

        converted_image = color_processing(image)
        steps.append(("색공간 변환 후", converted_image, vertices))
        
        # edges = edge_processing(image)
        # steps.append(("엣지 검출 후", edges, vertices))
        
        # # # 영수증 전처리 추가
        # preprocessed = preprocess_receipt(image)
        # steps.append(("영수증 전처리 후", preprocessed, vertices))
        result = preprocess_receipt(image)
        steps.append(("영수증 전처리 후", result, vertices))
        
        improved_background_removal_image = improved_background_removal_rgb(image)
        steps.append(("개선된 배경 제거 후", improved_background_removal_image, vertices))
        # masked_image = detect_receipt(converted_image)   
        # steps.append(("경계 검출 후", masked_image, vertices)) 

        # selected_image = select_receipt(converted_image)
        # steps.append(("윤곽선 선택 후", selected_image, vertices))
        
        # draw_contours_image = draw_contours(converted_image)
        # steps.append(("윤곽선 검출 후", draw_contours_image, vertices))
        
        # detect_rectangle_image = detect_rectangle(converted_image)
        # steps.append(("영수증 검출 후", detect_rectangle_image, vertices))
        
        # 리사이즈
        resized_img, resized_vertices = resize_img(Image.fromarray(converted_image), vertices, image_size)
        steps.append(("리사이즈 후", resized_img, resized_vertices))
        
        # 높이 조정
        adjusted_img, adjusted_vertices = adjust_height(resized_img, resized_vertices, height_ratio)
        steps.append(("높이 조정 후", adjusted_img, adjusted_vertices))
        
        # 회전
        rotated_img, rotated_vertices = rotate_img(adjusted_img, adjusted_vertices, angle_range)
        steps.append(("회전 후", rotated_img, rotated_vertices))
        
        # 크롭
        crop_size = min(crop_size, rotated_img.size[0], rotated_img.size[1])  # 이미지 크기에 맞게 조정
        cropped_img, cropped_vertices = crop_img(rotated_img, rotated_vertices, labels, crop_size)
        # PIL Image를 numpy array로 변환
        cropped_img = np.array(cropped_img)
        
        # annotation 그리기 - vertices가 비어있지 않은 경우에만 처리
        if cropped_vertices is not None and len(cropped_vertices) > 0:
            for vertex in cropped_vertices:
                if vertex.size > 0:  # vertex가 비어있지 않은 경우에만 처리
                    pts = vertex.reshape((4, 2)).astype(np.int32)
                    cv2.polylines(cropped_img, [pts], True, (0, 255, 0), 2)
        
        steps.append(("크롭 후", cropped_img, cropped_vertices))


        # 결과 표시 부분 수정
        st.write("### 전처리 과정 시각화")
        
        # 전체 과정 미리보기 - 2행으로 나누어 표시
        st.write("#### 전체 과정 미리보기")
        
        # steps 길이에 맞춰 컬럼 생성
        num_steps = len(steps)
        row1_cols = st.columns(num_steps)
        
        # 모든 단계를 한 줄에 표시
        for idx in range(num_steps):
            with row1_cols[idx]:
                title, img, verts = steps[idx]
                st.caption(title)
                img_array = np.array(img) if isinstance(img, Image.Image) else img
                img_with_boxes = draw_boxes(img_array, verts)
                st.image(img_with_boxes, channels="BGR", use_column_width=True)
        
        # 구분선 추가
        st.markdown("---")

        # 상세 보기 탭
        st.write("#### 상세 보기")
        tabs = st.tabs([step[0] for step in steps])
        
        for idx, tab in enumerate(tabs):
            with tab:
                title, img, verts = steps[idx]
                col1, col2 = st.columns([1, 1])  # 이미지와 정보를 3:1 비율로 변경
                
                with col1:
                    img_array = np.array(img) if isinstance(img, Image.Image) else img
                    img_with_boxes = draw_boxes(img_array, verts)
                    st.image(img_with_boxes, channels="BGR", use_column_width=True)
                
                with col2:
                    st.markdown("##### 이미지 정보")
                    stats = get_preprocessing_stats(img, verts)
                    st.markdown(f"""
                    - **크기**: {stats['shape']}
                    - **박스 수**: {stats['num_boxes']}
                    """)
                    
                    # 추가 정보 표시를 마크다운으로 변경
                    if idx == 1:
                        st.markdown(f"- **목표 크기**: {image_size}")
                    elif idx == 2:
                        st.markdown(f"- **높이 조정 범위**: ±{height_ratio*100}%")
                    elif idx == 3:
                        st.markdown(f"- **회전 각도 범위**: ±{angle_range}°")
                    elif idx == 4:
                        st.markdown(f"- **크롭 크기**: {crop_size}x{crop_size}")

if __name__ == "__main__":
    main()