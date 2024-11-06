import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

def copy_images(src_dir: str, dst_dir: str, annotations: List[str]):
    """
    Copy images from source directory to destination directory while preserving filenames.
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # Get all image paths from annotations
    image_paths = set()
    for ann_file in annotations:
        if not os.path.exists(ann_file):
            continue
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                image_paths.add(data['file_name'])
    
    # Copy each image with new naming format
    for img_path in image_paths:
        src_path = os.path.join(src_dir, img_path)
        filename = os.path.basename(img_path)
        new_filename = f"wild_receipt_extractor.en.in_external_{filename}"
        dst_path = os.path.join(dst_dir, new_filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Could not find image at {src_path}")

def convert_wild_to_ufo(input_files: List[str], output_file: str):
    """Convert WILD format annotations to UFO format."""
    ufo_dict = {}
    
    # Process all input files
    for input_file in input_files:
        if not os.path.exists(input_file):
            continue
            
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line.strip())
            filename = os.path.basename(data['file_name'])
            # Update filename format in UFO annotation
            new_filename = f"wild_receipt_extractor.en.in_external_{filename}"
            
            # Initialize UFO format for this image
            word_instances = {}
            
            # Convert each annotation
            for idx, ann in enumerate(data['annotations']):
                points = ann['box']
                # Convert box format [x1,y1,x2,y2,x3,y3,x4,y4] to points format
                word_instances[f'word_{idx}'] = {
                    'points': [[points[0], points[1]],
                             [points[2], points[3]],
                             [points[4], points[5]],
                             [points[6], points[7]]],
                    'transcription': ann['text'],
                    'language': ['en'],
                    'tags': [f'label_{ann["label"]}']
                }
            
            # Create image entry with new filename
            ufo_dict[new_filename] = {
                'img_h': data['height'],
                'img_w': data['width'],
                'words': word_instances,
                'chars': {},
                'tags': []
            }
    
    # Write output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'images': ufo_dict}, f, indent=2, ensure_ascii=False)

def main():
    # Use relative paths instead of absolute paths
    project_root = Path(__file__).parent.parent
    
    # Define paths
    base_dir = project_root / 'data/wild_receipt'
    img_dir = base_dir / 'img'
    ufo_dir = base_dir / 'ufo'
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ufo_dir, exist_ok=True)
    
    # Define input files using relative paths
    src_img_dir = project_root / 'wildreceipt'
    annotation_files = [
        str(project_root / 'wildreceipt/train.txt'),
        str(project_root / 'wildreceipt/test.txt')
    ]
    
    # Copy images from both train and test
    if os.path.exists(src_img_dir):
        copy_images(str(src_img_dir), str(img_dir), annotation_files)
    
    # Convert both train and test annotations to a single UFO file
    convert_wild_to_ufo(annotation_files, str(ufo_dir / 'train.json'))
    
    print("Conversion completed!")
    print(f"Images copied to: {img_dir}")
    print(f"Annotations saved to: {ufo_dir / 'train.json'}")

if __name__ == '__main__':
    main()