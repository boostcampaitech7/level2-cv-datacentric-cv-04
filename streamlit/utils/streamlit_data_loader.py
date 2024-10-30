import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import cv2
import os
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw

## EDA에 사용했던 내용 모듈화 ###
def rectify_poly(poly: np.ndarray, direction: str = 'Horizontal') -> Tuple[float, float]:
    """다각형의 높이와 너비를 계산하는 함수
        input: poly(다각형 좌표), direction(방향)
        output: height(높이), width(너비) 
    """
    try:
        rect = cv2.minAreaRect(poly)
        (_, _), (width, height), angle = rect
        
        if direction == 'Horizontal':
            if width < height:
                width, height = height, width
        return height, width
    except:
        return None

def load_and_process_data(json_path: str) -> Tuple[Dict, pd.DataFrame]:
    """JSON 파일을 로드하고 전처리하는 함수
        input: UFO 형식의 json 파일 경로
        output: 데이터프레임, bbox 크기 리스트, 단어 높이 리스트, 단어 너비 리스트, 언어 리스트
    """
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 데이터 저장을 위한 리스트 초기화
    image_data = []
    word_data = []
    
    # 데이터 처리
    for image_key, image_value in data["images"].items():
        # 이미지 정보 저장
        image_info = {
            'image': image_key,
            'image_width': image_value['img_w'],
            'image_height': image_value['img_h']
        }
        
        word_count = 0
        word_ann = image_value['words']
        
        for word in word_ann.values():
            if 'transcription' in word and 'points' in word:
                word_count += 1
                poly = np.int32(word['points'])
                
                size = rectify_poly(poly, 'Horizontal')
                if size is not None:
                    word_info = {
                        'image': image_key,
                        'bbox_size': np.sum(size[0] * size[1]),
                        'word_height': size[0],
                        'word_width': size[1],
                        'language': word.get('language', ['unknown'])[0] if isinstance(word.get('language', ['unknown']), list) else word.get('language', 'unknown')
                    }
                    word_data.append(word_info)
        
        image_data.append(image_info)
        image_info['word_counts'] = word_count
        image_data.append(image_info)
    
    # DataFrame 생성
    df_images = pd.DataFrame(image_data)
    df_words = pd.DataFrame(word_data)
    
    return data, df_images, df_words


def read_json(filename: str):
    """JSON 파일을 로드하는 함수"""
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def load_image_and_annotation(base_dir: str, lang: str, split: str, nation_dict: dict):
    """폴더에서 이미지와 JSON 어노테이션을 로드하는 함수
    input: 
        base_dir: 데이터 기본 경로
        lang: 선택된 언어
        split: train/test 선택
    """
    # 경로 설정
    img_dir = os.path.join(base_dir, nation_dict[lang], 'img', split)
    json_path = os.path.join(base_dir, nation_dict[lang], 'ufo', f'{split}.json')
    
    # 이미지 파일 리스트 가져오기
    if not os.path.exists(img_dir):
        st.error(f"경로를 찾을 수 없습니다: {img_dir}")
        return None, None
        
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.error("이미지 파일을 찾을 수 없습니다.")
        return None, None
    
    # 선택할 이미지 파일명
    selected_image = st.selectbox("이미지 선택", image_files)
    
    if selected_image:
        # 이미지 로드
        img_path = os.path.join(img_dir, selected_image)
        image = Image.open(img_path).convert("RGB")
        
        # JSON 데이터 로드
        try:
            annotations = read_json(json_path)
            return image, annotations.get('images', {}).get(selected_image, None)
        except FileNotFoundError:
            st.error(f"어노테이션 파일을 찾을 수 없습니다: {json_path}")
            return image, None
    
    return None, None
def draw_annotations(image, annotations):
    """이미지에 어노테이션을 그리는 함수
        input: 이미지, annotations
        output: image with annotations
    """
    if annotations is None:
        return image
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    if 'words' in annotations:
        for obj_k, obj_v in annotations['words'].items():
            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v['points']]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]

            draw.polygon(pts, outline=(255, 0, 0))                
            draw.text(
                (pt1[0]-3, pt1[1]-12),
                obj_k,
                fill=(0, 0, 0)
            )
    
    return img_copy