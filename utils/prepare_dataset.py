import json
import os
import random
from pathlib import Path
import shutil
from argparse import ArgumentParser

def load_ufo_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_and_split_dataset(
    root_dir='../data',
    languages=['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt','english_receipt'],
    val_ratio=0.2,
    seed=42
):
    # 현재 스크립트 위치 기준으로 상대 경로 계산
    script_dir = Path(__file__).parent
    root_dir = (script_dir / root_dir).resolve()
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Root directory: {root_dir}")
    
    random.seed(seed)
    
    # 병합된 데이터를 저장할 새 디렉토리 생성
    merged_dir = root_dir / 'merged_receipts'
    merged_dir.mkdir(exist_ok=True)
    (merged_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (merged_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    
    merged_data = {'images': {}}
    
    # 각 언어별 데이터 로드 및 병합
    for lang in languages:
        lang_dir = root_dir / lang
        train_json = lang_dir / 'ufo' / 'train.json'
        
        if not train_json.exists():
            print(f"Warning: {train_json} not found, skipping...")
            continue
            
        print(f"Processing {lang}...")
        train_data = load_ufo_data(train_json)
        
        # 이미지 경로와 어노테이션 정보 수집
        for img_name, img_info in train_data['images'].items():
            src_img_path = lang_dir / 'img' / 'train' / img_name
            if src_img_path.exists():
                # 원본 이미지 경로 저장
                img_info['file_name'] = str(src_img_path)
                merged_data['images'][f"{lang}_{img_name}"] = img_info
            else:
                print(f"Warning: Image not found: {src_img_path}")
    
    if not merged_data['images']:
        raise ValueError("No valid images found to process!")
    
    print(f"Total images found: {len(merged_data['images'])}")
    
    # 전체 데이터를 train/val로 분할
    all_images = list(merged_data['images'].keys())
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # 분할된 데이터 저장
    train_data = {'images': {}}
    val_data = {'images': {}}
    
    def copy_and_save_data(image_list, target_data, target_split):
        for img_name in image_list:
            img_info = merged_data['images'][img_name]
            target_data['images'][img_name] = img_info
            
            # 원본 이미지 경로에서 복사
            src_path = Path(img_info['file_name'])
            if not src_path.exists():
                print(f"Warning: Source image not found: {src_path}")
                continue
                
            dst_path = merged_dir / 'images' / target_split / img_name
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {img_name} to {target_split}")
            except Exception as e:
                print(f"Error copying {img_name}: {str(e)}")
    
    print("\nProcessing train split...")
    copy_and_save_data(train_images, train_data, 'train')
    
    print("\nProcessing validation split...")
    copy_and_save_data(val_images, val_data, 'val')
    
    # UFO 포맷으로 저장
    print("\nSaving annotations...")
    with open(merged_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    
    with open(merged_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
    
    return {
        'train_size': len(train_images),
        'val_size': len(val_images)
    }

def main(args):
    print("Preparing dataset...")
    
    # 기본 languages 리스트
    languages = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt', 'english_receipt']
    
    # external_data가 True일 때만 english_receipt 추가
    if args.external_data:
        languages.append('english_receipt')
    
    result = merge_and_split_dataset(
        root_dir=args.data_dir,
        languages=languages,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print(f"\nDataset preparation completed!")
    print(f"Train set size: {result['train_size']}")
    print(f"Validation set size: {result['val_size']}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--external_data', action='store_true', help='Include english receipt data')
    
    args = parser.parse_args()
    main(args)