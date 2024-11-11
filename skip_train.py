import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

# PYTORCH_CUDA_ALLOC_CONF 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import yaml

import wandb  # wandb import 추가
# validation
from deteval import calc_deteval_metrics # DeTEval import 추가
from utils.utils import get_pred_bboxes, get_gt_bboxes
import json


def parse_args():
    parser = ArgumentParser()

    # Conventional args 
    parser.add_argument('--data_dir', type=str,default="./blur/only_blur") # os.environ.get('SM_CHANNEL_TRAIN', 'data')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    parser.add_argument('--pretrained_model', type=str, default='./pretrained/Cord_#11_epoch_70.pth',
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=1024) # 2048 # resize
    parser.add_argument('--input_size', type=int, default=1024) # 1024 # crop
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--accumulation_steps' , type = int, default = 8) # gradient accumulation ## new 
    parser.add_argument('--learning_rate', type=float, default=1e-4)#1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    # Arguments related to Recall, Precision, F1-Score Evaluation
    parser.add_argument('--valid_json_file', type=str, default='val.json', help='Validation Json File for calculating F1 score. This will be joined with args.data_dir')
    parser.add_argument('--start_evaluation', type=int, default=50, help='Evaluation of Recall, Precision, F1-Score is started from this epoch')
    parser.add_argument('--evaluation_interval', type=int, default=2, help='Sets interval for calculating Recall, Precision, F1-Score. Calculated from args.start_evaluation epoch')
    # Wandb 관련 인자 추가
    parser.add_argument('--wandb_project', type=str, default='Fine-Tuning')
    parser.add_argument('--wandb_entity', type=str, default='cv_04_data_centric')
    parser.add_argument('--run_name', type=str, default='base_onlyline_epoch150_adamW_1024_1024')
    

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def save_checkpoint(model, optimizer, scheduler, epoch, loss, model_dir, f1=None, is_best=False):
    """체크포인트 저장 및 wandb에 업로드하는 함수"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }

    if is_best and f1 is not None:
        save_path = os.path.join(model_dir, f'best_epoch_{epoch+1}_f1_{f1:.3f}.pth')
        artifact_name = 'model-best'
    elif is_best:
        save_path = os.path.join(model_dir, f'best_epoch_loss_{loss:.3f}.pth')
        artifact_name = 'model-best'
    else:
        save_path = os.path.join(model_dir, f'epoch_{epoch+1}.pth')
        artifact_name = f'model-epoch-{epoch+1}'
    
    torch.save(checkpoint, save_path)
    
    # wandb에 아티팩트 업로드
    artifact = wandb.Artifact(
        name=artifact_name,
        type='model',
        description=f'Model checkpoint from epoch {epoch+1}'
    )
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

def check_loss_errors(extra_info, img, epoch, phase="train"):
    """
    Loss가 None인지 체크하고 에러를 로깅하는 함수
    Returns: True if there was an error, False otherwise
    """
    for loss_type in ['cls_loss', 'angle_loss', 'iou_loss']:
        if extra_info[loss_type] is None:
            wandb.log({
                "error_type": loss_type,
                "error_batch_shape": img.shape,
                "error_phase": phase,
                # "error_image": wandb.Image(img[0].cpu().numpy(), caption=f"Error Image - {loss_type} ({phase})"),
                "epoch": epoch + 1
            })
            print(f"Error in {loss_type} loss at {phase} phase")
            print(f"Batch shape: {img.shape}")
            print(f"Epoch: {epoch + 1}")
            return True
    return False

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, pretrained_model, **kwargs):
    
    # wandb 설정 파일 로드
    # wandb 설정 파일 로드
    try:
        with open('wandb_config.yaml', 'r') as f:
            wandb_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("wandb_config.yaml 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        wandb_config = {
            "project": "EAST_t1",#project name,
            #"name": "EAST-training1", # each experiment name 
            "entity": "cv_04_data_centric"  # team name
        }
    # run_name 설정
    run_name = wandb_config.get("name", None) or f"EAST_bs{batch_size}_lr{learning_rate}_{time.strftime('%Y%m%d_%H%M%S')}"

    # wandb 초기화
    wandb.init(
        project=args.wandb_project or wandb_config.get("project", "EAST-Text-Detection"),
        entity=args.wandb_entity or wandb_config.get("entity", None),
        name=args.run_name or run_name,
        tags=wandb_config.get("tags", []),
        group=wandb_config.get("group", None),
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "image_size": image_size,
            "input_size": input_size,
            "optimizer": "AdamW",
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [max_epoch // 2],
            "scheduler_gamma": 0.1,
        }
    )

    # 데이터셋 준비
    train_data_dir = os.path.join(data_dir) #'merged_receipts')
    if not os.path.exists(train_data_dir):
        raise ValueError(f"Merged dataset not found at {train_data_dir}. Please run prepare_dataset.py first.")
    
    # 데이터셋 경로 확인
    required_paths = [
        os.path.join(train_data_dir, 'images', 'train'),
        os.path.join(train_data_dir, 'train.json')
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise ValueError(f"Required path not found: {path}")
    
    # Train/Val 데이터셋 및 데이터로더 설정
    train_dataset = SceneTextDataset(
        train_data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size
    )
    train_dataset = EASTDataset(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataset = SceneTextDataset(
        train_data_dir,
        split='val',
        image_size=image_size,
        crop_size=input_size
    )
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 학습 관련 변수 초기화
    best_loss = float('inf')  # best_loss 초기화를 여기로 이동
    num_batches = math.ceil(len(train_dataset) / batch_size)
    num_val_batches = math.ceil(len(val_dataset) / batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = EAST()
    model.to(device)
    # 사전 학습된 모델 가중치 로드
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Loading pretrained model from {pretrained_model}")
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Loading base model')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # 0.001 -> 0.00025 -> 0.0000625
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        # Train phase
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0

        if device=="cuda": torch.cuda.empty_cache()
        # print(f"== memory : {torch.cuda.memory_allocated()}")
        # print(torch.cuda.memory_reserved())
        with tqdm(total=num_batches) as pbar:
            for i , (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description(f'[Epoch {epoch + 1}]')
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                # 에러 체크
                if check_loss_errors(extra_info, img, epoch, "train"):
                    continue
                
                
                loss.backward()
                # if (i+1) % args.accumulation_steps == 0 :
                    # optimizer.step()
                    # optimizer.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
                
                loss_val = loss.item()
                epoch_loss += loss_val
                # print("###",loss,extra_info['cls_loss'] , extra_info['angle_loss'],extra_info['iou_loss'])
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
        # 트레이닝 평균 손실 계산
        mean_epoch_loss = epoch_loss / num_batches
        mean_epoch_cls_loss = epoch_cls_loss / num_batches
        mean_epoch_angle_loss = epoch_angle_loss / num_batches
        mean_epoch_iou_loss = epoch_iou_loss / num_batches

        print(f'\nTraining metrics - Loss: {mean_epoch_loss:.4f}, Cls: {mean_epoch_cls_loss:.4f}, '
              f'Angle: {mean_epoch_angle_loss:.4f}, IoU: {mean_epoch_iou_loss:.4f}')

        # 트레이닝 메트릭 로깅
        wandb.log({
            "train_loss": mean_epoch_loss,
            "train_cls_loss": mean_epoch_cls_loss,
            "train_angle_loss": mean_epoch_angle_loss,
            "train_iou_loss": mean_epoch_iou_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })

        # Validation phase
        model.eval()
        val_loss = 0
        val_cls_loss = 0
        val_angle_loss = 0
        val_iou_loss = 0
        
        with torch.no_grad():
            with tqdm(total=num_val_batches, desc='Validation') as val_pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    img, gt_score_map = img.to(device), gt_score_map.to(device)
                    gt_geo_map, roi_mask = gt_geo_map.to(device), roi_mask.to(device)

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    
                    # 에러 체크
                    if check_loss_errors(extra_info, img, epoch, "validation"):
                        continue
                    
                    val_loss += loss.item()
                    val_cls_loss += extra_info['cls_loss']
                    val_angle_loss += extra_info['angle_loss']
                    val_iou_loss += extra_info['iou_loss']

                    val_pbar.update(1)
                    val_pbar.set_postfix({
                        'Val Cls loss': extra_info['cls_loss'],
                        'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    })

        # Validation 평균 손실 계산
        mean_val_loss = val_loss / num_val_batches
        mean_val_cls_loss = val_cls_loss / num_val_batches
        mean_val_angle_loss = val_angle_loss / num_val_batches
        mean_val_iou_loss = val_iou_loss / num_val_batches

        print(f'Validation metrics - Loss: {mean_val_loss:.4f}, Cls: {mean_val_cls_loss:.4f}, '
              f'Angle: {mean_val_angle_loss:.4f}, IoU: {mean_val_iou_loss:.4f}')


        # Validation 메트릭 로깅
        wandb.log({
            "val_loss": mean_val_loss,
            "val_cls_loss": mean_val_cls_loss,
            "val_angle_loss": mean_val_angle_loss,
            "val_iou_loss": mean_val_iou_loss,
            "epoch": epoch + 1
        })    

                
        # 트레이닝 평균 손실 계산
        mean_epoch_loss = epoch_loss / num_batches
        mean_epoch_cls_loss = epoch_cls_loss / num_batches
        mean_epoch_angle_loss = epoch_angle_loss / num_batches
        mean_epoch_iou_loss = epoch_iou_loss / num_batches

        # 트레이닝 메트릭 로깅
        wandb.log({
            "train_loss": mean_epoch_loss,
            "train_cls_loss": mean_epoch_cls_loss,
            "train_angle_loss": mean_epoch_angle_loss,
            "train_iou_loss": mean_epoch_iou_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        if device=="cuda": torch.cuda.empty_cache()
        # Validation phase
        model.eval()
        val_loss = 0
        val_cls_loss = 0
        val_angle_loss = 0
        val_iou_loss = 0

        #validation
        pred_bboxes_dict = {}
        gt_bboxes_dict = {}
        

        with torch.no_grad():
            with tqdm(total=num_val_batches, desc='Validation') as val_pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    
                    # 에러 체크
                    if check_loss_errors(extra_info, img, epoch, "validation"):
                        continue
                    
                    val_loss += loss.item()
                    val_cls_loss += extra_info['cls_loss']
                    val_angle_loss += extra_info['angle_loss']
                    val_iou_loss += extra_info['iou_loss']

                    # 예측 결과 저장
                    preds = extra_info['geo_map'].cpu().numpy()  # 예측 결과
                    gt_boxes = gt_geo_map.cpu().numpy()  # 실제 라벨

                    # 예측 결과와 실제 라벨을 딕셔너리에 추가
                    pred_bboxes_dict[f'img_{val_pbar.n}'] = preds
                    gt_bboxes_dict[f'img_{val_pbar.n}'] = gt_boxes

                    val_pbar.update(1)
                    val_pbar.set_postfix({
                        'Val Cls loss': extra_info['cls_loss'],
                        'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    })

        # Validation 평균 손실 계산
        mean_val_loss = val_loss / num_val_batches
        mean_val_cls_loss = val_cls_loss / num_val_batches
        mean_val_angle_loss = val_angle_loss / num_val_batches
        mean_val_iou_loss = val_iou_loss / num_val_batches
        # DeTEval을 사용하여 Precision, Recall, F1 Score 계산
        if epoch+1 >= args.start_evaluation and (epoch + 1 - args.start_evaluation) % args.evaluation_interval == 0:
            print("Calculating validation results...")
            valid_json_file = args.valid_json_file
            # 에러 체크

            with open(osp.join(args.data_dir, valid_json_file), 'r', encoding='utf-8') as file:
                data = json.load(file)
            valid_images = list(data['images'].keys())
            pred_bboxes_dict = get_pred_bboxes(model, args.data_dir, valid_images, args.input_size, args.batch_size)    
            gt_bboxes_dict = get_gt_bboxes(args.data_dir, json_file=valid_json_file, valid_images=valid_images)
            results = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict)

            # Precision, Recall, F1 Score 추출
            precision = results['total']['precision']
            recall = results['total']['recall']
            f1 = results['total']['hmean']
            # Validation 메트릭 출력
            print(f'Validation metrics - Loss: {mean_val_loss:.4f}, Cls: {mean_val_cls_loss:.4f}, '
                    f'Angle: {mean_val_angle_loss:.4f}, IoU: {mean_val_iou_loss:.4f}, '
                    f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            # Validation 메트릭 로깅
            wandb.log({
                "val_loss": mean_val_loss,
                "val_cls_loss": mean_val_cls_loss,
                "val_angle_loss": mean_val_angle_loss,
                "val_iou_loss": mean_val_iou_loss,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "epoch": epoch + 1
            })
        else:
            # Validation 메트릭 로깅
            wandb.log({
                "val_loss": mean_val_loss,
                "val_cls_loss": mean_val_cls_loss,
                "val_angle_loss": mean_val_angle_loss,
                "val_iou_loss": mean_val_iou_loss,
                "epoch": epoch + 1
            })
            f1 = None # 체크포인트 저장할 때, 예외처리를 위함

        scheduler.step()

        # 체크포인트 저장 (validation loss 기준)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                mean_val_loss, model_dir, f1,
                is_best=True
            )
            print(f'New best model saved! (Val Loss: {best_loss:.4f})')

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                mean_val_loss, model_dir,
                is_best=False
            )

        print('Time elapsed: {}'.format(timedelta(seconds=time.time() - epoch_start)))
    
    wandb.finish()


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)