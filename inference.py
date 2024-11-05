import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect

# dataset.py의 모든 전처리 함수들을 여기로 복사
from dataset import resize_img, adjust_height, _apply_mask_numba, preprocess_receipt, lab_processing, hsv_processing, _apply_morphology, background_removal
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange
import numpy as np

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args
    
def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    # 체크포인트에서 model_state_dict만 추출하여 로드
    checkpoint = torch.load(ckpt_fpath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []
    batch_count = 0
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))
        
        img = cv2.imread(image_fpath)
        img = preprocess_receipt(img)  # PIL Image 반환
        # PIL Image를 numpy array로 변환 후 RGB로 변환
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) == 2:  # grayscale인 경우
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        images.append(img)
        if len(images) == batch_size:
            batch_count += 1
            bboxes = detect(model, images, input_size)
            by_sample_bboxes.extend(bboxes)
            
            # 5번째 배치마다 첫 번째 이미지 시각화
            if batch_count % 5 == 0:
                vis_img = images[0].copy()
                # bbox 그리기
                for bbox in bboxes[0]:
                    points = bbox.astype(np.int32)
                    # 박스 그리기
                    cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)
                    # 각 모서리 점 표시
                    for point in points:
                        cv2.circle(vis_img, tuple(point), 3, (0, 0, 255), -1)
                
                # 결과 저장
                save_path = osp.join(args.output_dir, f'batch_{batch_count}_vis.jpg')
                cv2.imwrite(save_path, vis_img)
            
            images = []

    if len(images):
        bboxes = detect(model, images, input_size)
        by_sample_bboxes.extend(bboxes)

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result

def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'full_gradiant_100/epoch_70.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')
    
    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                              args.batch_size, split='test')  # output_dir 인자 추가
    ufo_result['images'].update(split_result['images'])

    output_fname = 'full_gradiant_100_epoch_70.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)
