import json
import os
from pathlib import Path

def merge_json_files(input_folder, output_file):
    merged_data = {'images': {}}

    # 입력 폴더 내의 모든 JSON 파일을 읽어들임
    for json_file in Path(input_folder).glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 각 JSON 파일의 'images' 데이터를 merged_data에 추가
            for image_name, image_info in data['images'].items():
                if image_name not in merged_data['images']:
                    merged_data['images'][image_name] = image_info
                else:
                    print(f"중복된 이미지 이름 발견: {image_name}. 기존 데이터 유지.")

    # 결과를 merged_data.json으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"모든 JSON 파일이 {output_file}로 병합되었습니다.")

if __name__ == "__main__":
    input_folder = "../../blur/only_blur"  # JSON 파일이 있는 폴더 경로
    output_file = "../../blur/only_blur/merged_data.json"  # 결과 파일 경로
    merge_json_files(input_folder, output_file)
