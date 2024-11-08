import os
import shutil
from pathlib import Path

def count_images_in_folder(folder):
    """주어진 폴더 내의 이미지 파일 개수를 세는 함수"""
    return len([f for f in os.listdir(folder) if Path(folder) / f].is_file())

def merge_images(source_folders, target_folder):
    # 타겟 폴더 생성 (존재하지 않으면)
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    total_images_copied = 0  # 복사된 이미지 개수를 세기 위한 변수

    for source_folder in source_folders:
        # 소스 폴더 내의 모든 파일을 확인
        for image_file in os.listdir(source_folder):
            source_path = Path(source_folder) / image_file
            target_path = Path(target_folder) / image_file
            
            # 이미지 파일인지 확인
            if source_path.is_file():
                # 파일 복사
                shutil.copy(source_path, target_path)
                print(f"복사됨: {source_path} -> {target_path}")
                total_images_copied += 1  # 복사된 이미지 개수 증가

    return total_images_copied

if __name__ == "__main__":
    # 소스 폴더 경로
    train_folder = "../../data/merged_receipts/images/train"
    val_folder = "../merged_img"
    
    # 타겟 폴더 경로
    merged_folder = "../../blur/only_blur/merged_data"
    
    # 각 폴더의 이미지 개수 세기
    train_image_count = len([f for f in os.listdir(train_folder) if (Path(train_folder) / f).is_file()])
    val_image_count = len([f for f in os.listdir(val_folder) if (Path(val_folder) / f).is_file()])
    
   
    # 이미지 병합
    total_copied = merge_images([train_folder, val_folder], merged_folder)
    print(f"Train 폴더의 이미지 개수: {train_image_count}")
    print(f"Val 폴더의 이미지 개수: {val_image_count}")
    print(f"총 병합된 이미지 개수: {total_copied}")
