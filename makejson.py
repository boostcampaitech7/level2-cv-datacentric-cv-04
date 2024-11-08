import os
from pytesseract import pytesseract, Output
import cv2
import json

# 경로 설정
image_dir = "./test"
output_file = "thai_receipt_annotations.json"

# OCR 주석 파일 생성 함수
def generate_ocr_annotations(image_path):
    image = cv2.imread(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)

    # JSON 형식에 맞는 데이터 구조 초기화
    image_annotations = {
        "paragraphs": {},
        "words": {}
    }

    # OCR 결과에서 단어 단위 정보 추출
    for i in range(len(ocr_data["text"])):
        if int(ocr_data["conf"][i]) > 60:  # 신뢰도 60 이상인 항목만 추가
            word_id = f"{i+1:04}"  # 예: "0001"처럼 4자리로 형식화
            transcription = ocr_data["text"][i]
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            
            # 각 단어의 사각형 좌표를 네 점의 포인트로 변환
            points = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)]
            ]

            # JSON 구조에 맞춰 정보 추가
            image_annotations["words"][word_id] = {
                "transcription": transcription,
                "points": points
            }

    return image_annotations

# 모든 이미지에 대해 OCR 주석 생성
all_annotations = {"images": {}}
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일만 처리
        image_path = os.path.join(image_dir, filename)
        all_annotations["images"][filename] = generate_ocr_annotations(image_path)

# JSON 파일로 저장
with open(output_file, "w") as f:
    json.dump(all_annotations, f, indent=4)

print(f"OCR 주석 파일이 {output_file}에 생성되었습니다.")
