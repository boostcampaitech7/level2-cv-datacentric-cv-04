import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

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

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    
    # # Wandb 관련 인자 추가
    # parser.add_argument('--wandb_project', type=str, default='EAST-Receipt-Detection')
    # parser.add_argument('--wandb_entity', type=str, default='your_entity_name')
    # parser.add_argument('--run_name', type=str, default=None)
    

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def save_checkpoint(model, optimizer, scheduler, epoch, loss, model_dir, is_best=False):
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

    if is_best:
        save_path = os.path.join(model_dir, 'best.pth')
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

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    
    # wandb 설정 파일 로드
    try:
        with open('wandb_config.yaml', 'r') as f:
            wandb_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("wandb_config.yaml 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        wandb_config = {
            "project": "EAST-Text-Detection",
            "name": "EAST-training"
        }

    # run_name 설정
    run_name = wandb_config.get("name", None) or f"EAST_bs{batch_size}_lr{learning_rate}_{time.strftime('%Y%m%d_%H%M%S')}"

    # wandb 초기화
    wandb.init(
        project=wandb_config.get("project", "EAST-Text-Detection"),
        entity=wandb_config.get("entity", None),
        name= run_name,
        tags=wandb_config.get("tags", []),
        group=wandb_config.get("group", None),
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "image_size": image_size,
            "input_size": input_size,
            "optimizer": "Adam",
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [max_epoch // 2],
            "scheduler_gamma": 0.1,
        }
    )

    # 데이터셋 준비
    train_data_dir = os.path.join(data_dir, 'merged_receipts')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 0.001 -> 0.00025 -> 0.0000625
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 4, max_epoch // 2, max_epoch // 4 * 3], gamma=0.25)

    model.train()
    for epoch in range(max_epoch):
        # Train phase
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description(f'[Epoch {epoch + 1}]')
                
                img, gt_score_map = img.to(device), gt_score_map.to(device)
                gt_geo_map, roi_mask = gt_geo_map.to(device), roi_mask.to(device)
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
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

        scheduler.step()

        # 체크포인트 저장 (validation loss 기준)
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                mean_val_loss, model_dir, 
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