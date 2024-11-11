import os
import shutil
import json
import random


def blur_train_test_splite() :
    # 설정
    data_dir = '../blur'
    image_dir = os.path.join(data_dir, 'long_distance_image')
    json_file_path = os.path.join(data_dir, 'train_long_distance.json')
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    # 디렉토리 생성
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 이미지 파일 목록
    image_filenames = list(data['images'].keys())
    random.shuffle(image_filenames)  # 무작위로 섞기

    # 훈련 및 검증 데이터 분할
    split_index = int(len(image_filenames) * 0.8)
    train_images = image_filenames[:split_index]
    val_images = image_filenames[split_index:]

    # 훈련 및 검증 이미지 복사 및 JSON 데이터 생성
    train_data = {'images': {}}
    val_data = {'images': {}}

    for img_name in train_images:
        # 실제 파일 이름 생성
        actual_img_name = "line_only_blur_" + img_name  # 'line_only_blur_'를 붙임
        img_path = os.path.join(image_dir, actual_img_name)  # 원래 파일 경로
        if os.path.exists(img_path):
            # 새로운 이름으로 저장 (img_name으로 저장)
            shutil.copy(img_path, os.path.join(train_dir, img_name))
            train_data['images'][img_name] = data['images'][img_name]  # JSON 데이터 추가
        else:
            print(f"Warning: {img_path} does not exist.")

    for img_name in val_images:
        # 실제 파일 이름 생성
        actual_img_name = "line_only_blur_" + img_name  # 'line_only_blur_'를 붙임
        img_path = os.path.join(image_dir, actual_img_name)  # 원래 파일 경로
        if os.path.exists(img_path):
            # 새로운 이름으로 저장 (img_name으로 저장)
            shutil.copy(img_path, os.path.join(val_dir, img_name))
            val_data['images'][img_name] = data['images'][img_name]  # JSON 데이터 추가
        else:
            print(f"Warning: {img_path} does not exist.")

    # 훈련 JSON 데이터 저장
    train_json_path = os.path.join(data_dir, 'train.json')
    with open(train_json_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)

    # 검증 JSON 데이터 저장
    val_json_path = os.path.join(data_dir, 'val.json')
    with open(val_json_path, 'w', encoding='utf-8') as file:
        json.dump(val_data, file, ensure_ascii=False, indent=4)

    print("훈련 및 검증 데이터와 JSON 파일이 성공적으로 생성되었습니다.")

import os


def rename_images_in_directory(source_directory, target_directory):
    """
    Rename images in the specified directory by removing 'line_only_blur_' from the filenames
    and save them to a new directory.
    
    Args:
        source_directory (str): The path to the directory containing the original images.
        target_directory (str): The path to the directory where renamed images will be saved.
    """
    # 타겟 디렉토리 생성
    os.makedirs(target_directory, exist_ok=True)

    # 디렉토리 내의 모든 파일을 확인
    for img_name in os.listdir(source_directory):
        if img_name.startswith("line_only_blur_"):
            # 새로운 파일 이름 생성
            new_img_name = img_name.replace("line_only_blur_", "")  # 'line_only_blur_' 제거
            old_img_path = os.path.join(source_directory, img_name)
            new_img_path = os.path.join(target_directory, new_img_name)
            shutil.copy(old_img_path, new_img_path)  # 파일 이름 변경 및 복사
            print(f"Renamed and copied: {old_img_path} to {new_img_path}")

# 사용 예시
image_directory = '../blur/long_distance_image'  # long_distance_image 폴더 경로
new_image_directory = '../blur/new_images'  # 새로운 이미지 저장 경로
rename_images_in_directory(image_directory, new_image_directory)