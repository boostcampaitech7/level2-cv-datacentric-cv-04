import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import psutil  # 시스템 모니터링용
import GPUtil  # GPU 모니터링용
import numpy as np

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default='./data')
    parser.add_argument('--model_dir', type=str, default='./trained_models')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    wandb.init(
        project="EAST-Text-Detection",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "image_size": image_size,
            "input_size": input_size,
            "optimizer": "Adam",
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [max_epoch // 4, max_epoch // 2, max_epoch -1],
            "scheduler_gamma": 0.1,
        }
    )
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 4, max_epoch // 2, max_epoch -1], gamma=0.1)

    # 성능 메트릭 추적을 위한 변수들
    best_loss = float('inf')
    running_loss = []
    batch_times = []
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        epoch_cls_loss, epoch_angle_loss, epoch_iou_loss = 0, 0, 0
        batch_start = time.time()
        
        with tqdm(total=num_batches) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                
                # 배치 시작 시간 기록
                batch_start_time = time.time()
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss']
                epoch_angle_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']
                running_loss.append(loss_val)
                
                # 배치 처리 시간 계산
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # GPU 상태 확인
                gpu = GPUtil.getGPUs()[0]
                
                # 배치별 상세 로깅
                wandb.log({
                    # 손실값 관련 메트릭
                    "batch_loss": loss_val,
                    "cls_loss": extra_info['cls_loss'],
                    "angle_loss": extra_info['angle_loss'],
                    "iou_loss": extra_info['iou_loss'],
                    "running_avg_loss": np.mean(running_loss[-100:]),  # 최근 100개 배치의 평균 손실
                    
                    # 학습 진행 상황
                    "epoch_progress": (batch_idx + 1) / num_batches * 100,
                    "total_progress": (epoch * num_batches + batch_idx + 1) / (max_epoch * num_batches) * 100,
                    
                    # 성능 메트릭
                    "batch_time": batch_time,
                    "samples_per_second": batch_size / batch_time,
                    "running_avg_batch_time": np.mean(batch_times[-100:]),
                    
                    # 시스템 메트릭
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_util": gpu.memoryUtil * 100,
                    "gpu_temperature": gpu.temperature,
                    "cpu_percent": psutil.cpu_percent(),
                    "ram_percent": psutil.virtual_memory().percent,
                })

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss'],
                    'Batch time': f'{batch_time:.3f}s'
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        
        # 에폭별 평균 계산
        mean_epoch_loss = epoch_loss / num_batches
        mean_epoch_cls_loss = epoch_cls_loss / num_batches
        mean_epoch_angle_loss = epoch_angle_loss / num_batches
        mean_epoch_iou_loss = epoch_iou_loss / num_batches
        
        # best loss 업데이트
        is_best = mean_epoch_loss < best_loss
        if is_best:
            best_loss = mean_epoch_loss
            # best 모델 저장
            best_ckpt_fpath = osp.join(model_dir, 'best.pth')
            torch.save(model.state_dict(), best_ckpt_fpath)
            wandb.save(best_ckpt_fpath)
        
        # 에폭별 상세 로깅
        wandb.log({
            "epoch": epoch + 1,
            "mean_epoch_loss": mean_epoch_loss,
            "mean_epoch_cls_loss": mean_epoch_cls_loss,
            "mean_epoch_angle_loss": mean_epoch_angle_loss,
            "mean_epoch_iou_loss": mean_epoch_iou_loss,
            "best_loss": best_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_time": time.time() - epoch_start,
            "is_best_epoch": is_best,
        })

        print('Mean loss: {:.4f} | Best loss: {:.4f} | Elapsed time: {}'.format(
            mean_epoch_loss, best_loss, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            # 에폭 번호가 포함된 체크포인트 파일명으로 저장
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            latest_fpath = osp.join(model_dir, 'latest.pth')
            
            # 모델 상태 저장
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': mean_epoch_loss,
                'best_loss': best_loss
            }
            
            # 두 버전으로 저장 (numbered version과 latest)
            torch.save(checkpoint, ckpt_fpath)
            torch.save(checkpoint, latest_fpath)
            
            # wandb에 체크포인트 업로드
            artifact = wandb.Artifact(
                name=f'model-checkpoint-epoch-{epoch+1}',
                type='model',
                description=f'Model checkpoint from epoch {epoch+1}'
            )
            artifact.add_file(ckpt_fpath)
            wandb.log_artifact(artifact)
            
            # latest 버전도 따로 업로드
            latest_artifact = wandb.Artifact(
                name='model-checkpoint-latest',
                type='model',
                description='Latest model checkpoint'
            )
            latest_artifact.add_file(latest_fpath)
            wandb.log_artifact(latest_artifact)

    wandb.finish()


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)