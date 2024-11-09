import json
from typing import Dict
from pathlib import Path

def add_language_tags(language_files: Dict[str, list]) -> None:
    """
    JSON 파일들에 언어 태그를 추가하는 함수
    
    Args:
        language_files (Dict[str, list]): 언어와 파일 경로 리스트를 담은 딕셔너리
    """
    for language, file_paths in language_files.items():
        for file_path in file_paths:
            try:
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                total_images = len(data['images'])
                processed_images = 0

                # 각 이미지에 대해
                for image_name in data['images']:
                    # 해당 이미지의 모든 word에 대해
                    for word_id in data['images'][image_name]['words']:
                        # language 필드 추가
                        data['images'][image_name]['words'][word_id]['language'] = language
                    
                    processed_images += 1
                    if processed_images % 100 == 0:  # 100개 이미지마다 진행상황 출력
                        print(f'{language} - {Path(file_path).name}: {processed_images}/{total_images} 이미지 처리 완료')

                # 수정된 JSON 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f'{language} - {Path(file_path).name} 파일 처리 완료!')
                
            except FileNotFoundError:
                print(f'경고: {file_path} 파일을 찾을 수 없습니다.')
            except Exception as e:
                print(f'에러: {language} 파일 처리 중 문제가 발생했습니다. {str(e)}')

def get_default_language_files() -> Dict[str, list]:
    """
    기본 언어 파일 경로를 반환하는 함수
    
    Returns:
        Dict[str, list]: 언어와 파일 경로 리스트를 담은 딕셔너리
    """
    base_path = Path('../data')
    return {
        'chinese': [
            str(base_path / 'chinese_receipt/ufo/train.json'),
            str(base_path / 'chinese_receipt/ufo/test.json')
        ],
        'japanese': [
            str(base_path / 'japanese_receipt/ufo/train.json'),
            str(base_path / 'japanese_receipt/ufo/test.json')
        ],
        'thai': [
            str(base_path / 'thai_receipt/ufo/train.json'),
            str(base_path / 'thai_receipt/ufo/test.json')
        ],
        'vietnamese': [
            str(base_path / 'vietnamese_receipt/ufo/train.json'),
            str(base_path / 'vietnamese_receipt/ufo/test.json')
        ]
    }

if __name__ == "__main__":
    # 스크립트로 직접 실행될 때만 실행됨
    language_files = get_default_language_files()
    add_language_tags(language_files)