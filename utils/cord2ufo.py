import json
import glob
import os
from pathlib import Path
import shutil

"""
CORD 데이터셋을 UFO 형식으로 변환하고 이미지 파일을 복사
"""
def get_excluded_images():
    """제외할 이미지 번호 목록을 반환합니다."""
    return {
        '002', '004', '098', '127', '264', '286', '291', '293', '296', '302', 
        '308', '316', '362', '386', '403', '404', '427', '449', '459', '463', 
        '484', '501', '515', '554', '556', '559', '561', '576', '588', '627', 
        '647', '676', '680', '736', '760', '778', '789'
    }

def convert_cord_to_ufo(json_dir, img_dir, target_img_dir):
    ufo_json = {
        "images": {}
    }
    
    # 제외할 이미지 번호 가져오기
    excluded_images = get_excluded_images()
    
    # JSON 파일 목록 가져오기
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    for json_path in json_files:
        # 파일 번호 추출 (receipt_00001.json -> 00001)
        file_num = Path(json_path).stem.split('_')[1]
        
        # 제외할 이미지인 경우 건너뛰기
        if file_num in excluded_images:
            print(f"제외된 이미지: receipt_{file_num}")
            continue
            
        # CORD JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            cord_json = json.load(f)
            
        # 원본 이미지 파일명과 새로운 이미지 파일명
        orig_image = f"receipt_{file_num}.png"
        new_image = f"english_receipt_extractor.en.in_external_{file_num}.jpg"
        
        # 이미지 파일 복사
        src_path = os.path.join(img_dir, orig_image)
        dst_path = os.path.join(target_img_dir, 'train', new_image)  # img/train 디렉토리에 저장
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"이미지 복사 완료: {src_path} -> {dst_path}")
        else:
            print(f"이미지를 찾을 수 없음: {src_path}")
            continue
        
        # UFO 형식의 이미지 데이터 초기화
        ufo_json["images"][new_image] = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": cord_json["meta"]["image_size"]["width"],
            "img_h": cord_json["meta"]["image_size"]["height"],
            "tags": [],
            "relations": []
        }
        
        # valid_line의 각 항목을 처리
        word_count = 1
        for line in cord_json["valid_line"]:
            # 각 줄의 단어들을 처리
            for word in line["words"]:
                # 4자리 숫자 형식의 word ID 생성
                word_id = f"{word_count:04d}"
                word_count += 1
                
                # quad 좌표를 points 형식으로 변환
                quad = word["quad"]
                points = [
                    [quad["x1"], quad["y1"]],
                    [quad["x2"], quad["y2"]],
                    [quad["x3"], quad["y3"]],
                    [quad["x4"], quad["y4"]]
                ]
                
                # UFO 형식의 word 정보 저장
                ufo_json["images"][new_image]["words"][word_id] = {
                    "transcription": word["text"],
                    "points": points,
                }
    
    return ufo_json

def main():
    # 데이터 디렉토리 설정
    cord_base_dir = "../CORD"
    target_base_dir = "../data/english_receipt"
    
    if not os.path.exists(target_base_dir):
        os.makedirs(target_base_dir)
    
    # 소스 디렉토리 설정
    train_json_dir = os.path.join(cord_base_dir, "train/json")
    train_img_dir = os.path.join(cord_base_dir, "train/image")
    test_json_dir = os.path.join(cord_base_dir, "test/json")
    test_img_dir = os.path.join(cord_base_dir, "test/image")
    
    # 대상 디렉토리 설정
    target_img_dir = os.path.join(target_base_dir, "img")  # 이미지 저장 디렉토리
    target_ufo_dir = os.path.join(target_base_dir, "ufo")  # UFO JSON 저장 디렉토리
    
    # 필요한 디렉토리 생성
    os.makedirs(os.path.join(target_img_dir, 'train'), exist_ok=True)
    os.makedirs(target_ufo_dir, exist_ok=True)
    
    # CORD to UFO 변환 (train + test)
    print("CORD 형식을 UFO 형식으로 변환 중...")
    
    # train 데이터 변환
    print("train 데이터 처리 중...")
    ufo_json = convert_cord_to_ufo(train_json_dir, train_img_dir, target_img_dir)
    train_count = len(ufo_json["images"])
    print(f"train 데이터 처리 완료: {train_count}개 이미지")
    
    # test 데이터 변환 및 병합
    print("test 데이터 처리 중...")
    test_json = convert_cord_to_ufo(test_json_dir, test_img_dir, target_img_dir)
    if test_json and test_json["images"]:
        ufo_json["images"].update(test_json["images"])
        test_count = len(test_json["images"])
        print(f"test 데이터 처리 완료: {test_count}개 이미지")
    
    # 결과 저장
    output_path = os.path.join(target_ufo_dir, "train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_json, f, indent=2, ensure_ascii=False)
    
    total_count = len(ufo_json["images"])
    print(f"변환 완료! 결과가 {output_path}에 저장되었습니다.")
    print(f"처리된 총 이미지 수: {total_count}")

if __name__ == "__main__":
    main()
