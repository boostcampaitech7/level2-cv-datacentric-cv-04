import json
import glob
import os
from pathlib import Path
import shutil
from PIL import Image

def convert_txt_to_ufo(txt_dir, img_dir, target_img_dir):
    ufo_json = {
        "images": {}
    }
    
    # .txt 파일 목록 가져오기
    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    
    for txt_path in txt_files:
        # 원본 이미지 파일명 가져오기 (X00016469612.txt -> X00016469612.jpg)
        base_name = Path(txt_path).stem
        orig_image = f"{base_name}.jpg"
        
        # 새로운 이미지 파일명 설정
        new_image = f"english_receipt_extractor.en.in_external_{base_name}.jpg"
        
        # 이미지 파일 복사 및 크기 가져오기
        src_path = os.path.join(img_dir, orig_image)
        dst_path = os.path.join(target_img_dir, 'train', new_image)
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"이미지 복사 완료: {src_path} -> {dst_path}")
            
            # 이미지 크기 가져오기
            with Image.open(src_path) as img:
                img_w, img_h = img.size
                print(img_w, img_h)
        else:
            print(f"이미지를 찾을 수 없음: {src_path}")
            continue
        
        # UFO 형식의 이미지 데이터 초기화
        ufo_json["images"][new_image] = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": img_w,
            "img_h": img_h,
            "tags": [],
            "relations": []
        }
        
        # .txt 파일에서 bbox와 word 정보 읽기
        word_count = 1
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove commas and split by whitespace
                parts = line.strip().replace(",", " ").split()
        
            # Ensure there are 9 parts: 8 for coordinates and 1 for the word
            if len(parts) < 9:
                print(f"잘못된 형식: {line}")
                continue
        
            coords = list(map(int, parts[:8]))  # 좌표를 정수로 변환
            word = " ".join(parts[8:])  # Join remaining parts as the transcription
                
            # 4자리 숫자 형식의 word ID 생성
            word_id = f"{word_count:04d}"
            word_count += 1
            
            # UFO 형식의 word 정보 저장
            points = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]]
            ]
            
            ufo_json["images"][new_image]["words"][word_id] = {
                "transcription": word,
                "points": points,
            }

    return ufo_json

def main():
    # 데이터 디렉토리 설정
    base_dir = "/data/ephemeral/home/kenlee/level2-cv-datacentric-cv-04/SORIE2019_v2_data"
    
    # 소스 디렉토리 설정
    txt_dir = os.path.join(base_dir, "box")
    img_dir = os.path.join(base_dir, "img")
    
    # 대상 디렉토리 설정
    target_img_dir = os.path.join(base_dir, "processed/img")
    target_ufo_dir = os.path.join(base_dir, "processed/ufo")
    
    # 필요한 디렉토리 생성
    os.makedirs(os.path.join(target_img_dir, 'train'), exist_ok=True)
    os.makedirs(target_ufo_dir, exist_ok=True)
    
    # .txt to UFO 변환
    print("TXT 형식을 UFO 형식으로 변환 중...")
    ufo_json = convert_txt_to_ufo(txt_dir, img_dir, target_img_dir)
    total_count = len(ufo_json["images"])
    print(f"변환 완료: {total_count}개 이미지")
    
    # 결과 저장
    output_path = os.path.join(target_ufo_dir, "train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_json, f, indent=2, ensure_ascii=False)
    
    print(f"UFO 형식 변환 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
