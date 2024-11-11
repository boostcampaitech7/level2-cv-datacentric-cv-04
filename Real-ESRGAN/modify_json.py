import json
import argparse
import os

def sr_modify_json(input_json_path, output_json_path, s):
    # JSON 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # images 내의 모든 이미지에 대해 처리
    for image_name, image_data in data.get('images', {}).items():
        img_w = image_data.get('img_w', 0)  # 기본값 0
        img_h = image_data.get('img_h', 0)  # 기본값 0

        # img_w와 img_h가 존재하는 경우에만 수정
        if img_w > 0 and img_h > 0:
            image_data['img_w'] = img_w * s
            image_data['img_h'] = img_h * s
        else:
            print(f"Warning: 'img_w' or 'img_h' not found in {input_json_path} for image {image_name}. Using default values.")

        # points 수정
        for word in image_data.get('words', {}).values():  # 'words'가 없을 경우 빈 딕셔너리 사용
            for point in word.get('points', []):  # 'points'가 없을 경우 빈 리스트 사용
                point[0] *= s  # x 좌표 수정
                point[1] *= s  # y 좌표 수정

    # 수정된 내용을 새로운 JSON 파일로 저장
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    # ArgumentParser 설정
    parser = argparse.ArgumentParser(description='Modify JSON points and image size based on scale factor.')
    parser.add_argument('--scale', type=float, nargs='?', default=2, help='Scale factor for modifying points and image size')
    parser.add_argument('--input_base_path', type=str, nargs='?', default='../', help='Base path for input data (should be "data" folder)')
    parser.add_argument('--output_base_path', type=str, nargs='?', default='../up2x_results', help='Base path for output data')

    args = parser.parse_args()

    # 입력 폴더 경로 설정
    input_folders = [
        'data/chinese_receipt/ufo/test.json',
        'data/chinese_receipt/ufo/train.json',
        'data/japanese_receipt/ufo/test.json',
        'data/japanese_receipt/ufo/train.json',
        'data/thai_receipt/ufo/test.json',
        'data/thai_receipt/ufo/train.json',
        'data/vietnamese_receipt/ufo/test.json',
        'data/vietnamese_receipt/ufo/train.json',
    ]

    # 전체 입력 및 출력 경로 생성
    input_paths = [os.path.join(args.input_base_path, folder) for folder in input_folders]
    output_paths = [os.path.join(args.output_base_path, folder) for folder in input_folders]

    # 각 JSON 파일 수정
    for input_path, output_path in zip(input_paths, output_paths):
        print(f'Modifying {input_path} and saving to {output_path}')
        sr_modify_json(input_path, output_path, args.scale)

if __name__ == '__main__':
    main()