import json
import glob
import os
from pathlib import Path
import shutil
from PIL import Image

def convert_txt_to_ufo(txt_dir, img_dir, target_img_train_dir):
    ufo_json = {"images": {}}
    
    # Retrieve .txt files
    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    
    for txt_path in txt_files:
        # Extract base name and set image file names
        base_name = Path(txt_path).stem
        orig_image = f"{base_name}.jpg"
        new_image = f"english_receipt_extractor.en.in_external_{base_name}.jpg"
        
        # Copy image and get dimensions
        src_path = os.path.join(img_dir, orig_image)
        dst_path = os.path.join(target_img_train_dir, new_image)  # Place image in img/train only
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            print(f"Image copied: {src_path} -> {dst_path}")
            
            with Image.open(src_path) as img:
                img_w, img_h = img.size
                print(f"Image dimensions: {img_w}x{img_h}")
        else:
            print(f"Image not found: {src_path}")
            continue
        
        # Initialize UFO format data
        ufo_json["images"][new_image] = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": img_w,
            "img_h": img_h,
            "tags": [],
            "relations": []
        }
        
        # Parse bounding box data from .txt
        word_count = 1
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                
                if len(parts) < 9:
                    print(f"Invalid format in line: {line}")
                    continue
        
                coords = list(map(int, parts[:8]))
                word = " ".join(parts[8:])
                
                word_id = f"{word_count:04d}"
                word_count += 1
                
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
    # Base directories
    base_dir = "../data/sroie_receipt"
    txt_dir = os.path.join(base_dir, "train/box")
    img_dir = os.path.join(base_dir, "train/img")
    
    # Target directories
    target_img_train_dir = os.path.join(base_dir, "img/train")  # Save only to img/train under english_receipt
    target_ufo_dir = os.path.join(base_dir, "ufo")
    
    os.makedirs(target_img_train_dir, exist_ok=True)
    os.makedirs(target_ufo_dir, exist_ok=True)
    
    print("Converting TXT format to UFO format...")
    ufo_json = convert_txt_to_ufo(txt_dir, img_dir, target_img_train_dir)
    total_count = len(ufo_json["images"])
    print(f"Conversion complete: {total_count} images processed.")
    
    # Save UFO JSON
    output_path = os.path.join(target_ufo_dir, "train.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_json, f, indent=2, ensure_ascii=False)
    print(f"UFO format saved to {output_path}.")

    # Clean up any folders or files under english_receipt except img and ufo
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if item not in ["img", "ufo"]:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
            else:
                os.remove(item_path)
                print(f"Deleted file: {item_path}")

if __name__ == "__main__":
    main()
