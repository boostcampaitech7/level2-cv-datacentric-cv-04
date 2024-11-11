import json
import cv2
import numpy as np
from pathlib import Path

def detect_horizontal_lines(json_path, image_folder):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    generated_images_count = 0  # 생성된 이미지 개수를 세기 위한 변수

    for image_name, image_data in data['images'].items():
        # 이미지 로드
        image_path = Path(image_folder) / image_name
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"이미지를 찾을 수 없습니다: {image_path}")
            continue
            
        # 모든 bbox 영역을 위한 마스크 초기화
        text_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        line_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        has_filtered_word = False  # 필터링된 단어가 있는지 확인하는 변수

        for word_id, word_info in image_data['words'].items():
            points = np.array(word_info['points'])
            
            # 모든 텍스트 영역을 text_mask에 추가
            cv2.fillPoly(text_mask, [points.astype(np.int32)], 255)
            
            box_width = int(abs(points[1][0] - points[0][0]))
            box_height = int(abs(points[2][1] - points[1][1]))
            
            aspect_ratio = box_width / box_height if box_height != 0 else 0
            
            # 필터링 조건
            if aspect_ratio > 20.0 or word_info['transcription'] == "":
                # 가로선 영역을 line_mask에 추가
                cv2.fillPoly(line_mask, [points.astype(np.int32)], 255)
                print(f"발견된 가로선 - 이미지: {image_name}, Word ID: {word_id}, 비율: {aspect_ratio:.2f}")
                has_filtered_word = True  # 필터링된 단어가 발견됨

        # 필터링된 단어가 하나라도 있는 경우에만 이미지 저장
        if has_filtered_word:
            # 가로선이 아닌 텍스트 영역만 선택 (text_mask - line_mask)
            blur_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(line_mask))
            blur_mask = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR)
            blur_mask = blur_mask > 0
            
            # 블러 처리
            blurred_image = cv2.GaussianBlur(image, (101, 101), 15)
            result_image = image.copy()
            result_image[blur_mask] = blurred_image[blur_mask]
        
            # 결과 이미지 저장
            output_path = Path("output1") / f"Only_line_{image_name}"
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), result_image)
            
            generated_images_count += 1  # 생성된 이미지 개수 증가
        else:
            print(f"필터링된 단어가 있는 이미지: {image_name} 저장 생략")

    print(f"총 생성된 이미지 개수: {generated_images_count}")

if __name__ == "__main__":
    json_path = "../../data/merged_receipts/val.json"
    image_folder = "../../data/merged_receipts/images/val/"
    detect_horizontal_lines(json_path, image_folder)