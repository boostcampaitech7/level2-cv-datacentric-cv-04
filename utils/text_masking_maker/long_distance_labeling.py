import json
import numpy as np

def detect_horizontal_lines(json_path, image_folder, output_json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = {'images': {}}
    
    for image_name, image_data in data['images'].items():
        filtered_words = {}
        
        for word_id, word_info in image_data['words'].items():
            points = np.array(word_info['points'])
            
            box_width = int(abs(points[1][0] - points[0][0]))
            box_height = int(abs(points[2][1] - points[1][1]))
            
            aspect_ratio = box_width / box_height if box_height != 0 else 0
            
            if aspect_ratio > 20.0:
                filtered_words[word_id] = word_info
        
        if filtered_words:
            filtered_data['images'][f"Only_line_{image_name}"] = {
                'words': filtered_words
            }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"필터링된 데이터가 {output_json_path}에 저장되었습니다.")

if __name__ == "__main__":
    json_path = "../data/merged_receipts/train.json"
    image_folder = "../data/merged_receipts/images/train/"
    output_json_path = "../data/merged_receipts/train_long_distance.json"
    detect_horizontal_lines(json_path, image_folder, output_json_path)