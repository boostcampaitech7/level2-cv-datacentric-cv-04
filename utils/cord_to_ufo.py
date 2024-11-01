import json
import glob
import os
from pathlib import Path

"""
CORD 데이터셋을 UFO 형식으로 변환
"""
def convert_cord_to_ufo(json_dir):
    ufo_json = {
        "images": {}
    }
    
    # JSON 파일 목록 가져오기
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    for json_path in json_files:
        # CORD JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            cord_json = json.load(f)
            
        # 파일 번호 추출 (receipt_00001.json -> 00001)
        file_num = Path(json_path).stem.split('_')[1]
        
        # 새로운 이미지 파일명 생성
        image_filename = f"english_receipt_extractor.en.in_external_{file_num}.jpg"
        
        # UFO 형식의 이미지 데이터 초기화
        ufo_json["images"][image_filename] = {
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
                ufo_json["images"][image_filename]["words"][word_id] = {
                    "transcription": word["text"],
                    "points": points,
                }
    
    return ufo_json

def main():
    # 데이터 디렉토리 설정
    base_dir = "../CORD/train"
    json_dir = os.path.join(base_dir, "json")
    
    # CORD to UFO 변환
    print("CORD 형식을 UFO 형식으로 변환 중...")
    ufo_json = convert_cord_to_ufo(json_dir)
    
    # 결과 저장
    output_path = os.path.join(base_dir, "train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_json, f, indent=2, ensure_ascii=False)
    
    print(f"변환 완료! 결과가 {output_path}에 저장되었습니다.")
    print(f"처리된 총 이미지 수: {len(ufo_json['images'])}")

if __name__ == "__main__":
    main()
