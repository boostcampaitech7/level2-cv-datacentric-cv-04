import json
import os
from PIL import Image, ImageDraw

# JSON 파일 경로
json_file_path = './test.json'  # train.json 파일 경로
image_dir = './test/'  # 이미지가 저장된 디렉토리 경로

# JSON 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 이미지에 대해 bbox 표시
for image_name, image_data in data['images'].items():
    # 이미지 파일 경로
    image_path = os.path.join(image_dir, image_name)  # image_name을 직접 사용

    # 이미지 파일 존재 여부 확인
    if not os.path.isfile(image_path):
        print(f"Image {image_path} not found.")
        continue

    # 이미지 열기
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # words에서 bbox 그리기
    for word_key, word_info in image_data['words'].items():
        points = word_info['points']  # points는 4개의 좌표를 포함하는 리스트

        # 사각형 그리기
        if len(points) == 4:  # points가 4개의 좌표를 가지고 있는지 확인
            draw.line([points[0][0], points[0][1], points[1][0], points[1][1]], fill="red", width=2)
            draw.line([points[1][0], points[1][1], points[2][0], points[2][1]], fill="red", width=2)
            draw.line([points[2][0], points[2][1], points[3][0], points[3][1]], fill="red", width=2)
            draw.line([points[3][0], points[3][1], points[0][0], points[0][1]], fill="red", width=2)
        else:
            print(f"Warning: Expected 4 points, but got {len(points)} for word '{word_key}'.")

    # 결과 이미지 저장 또는 표시
    #image.show()  # 이미지를 화면에 표시
    image.save(f'./test2/{image_name}_output.jpg')  # 이미지를 파일로 저장할 경우

print("Bounding boxes drawn on images.")