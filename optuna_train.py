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
## for optuna 
import optuna
from optuna.trial import Trial
from functools import partial
from torch.optim import Adam, AdamW
from torch import optim

import logging
import sys
import time
import pandas as pd



def parse_args():
    parser = ArgumentParser()

    # Conventional args
    # data dir should be set in 
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data')) ## you need to change
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'Optuna_models')) ## you need to change

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)# you can change
    parser.add_argument('--input_size', type=int, default=1024) # you can change
    parser.add_argument('--batch_size', type=int, default=8) #you can change
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=100)#150
    
    parser.add_argument('--study_name', type=str, default='your optuna_project name') ## you need to change
    parser.add_argument('--n_trials', type=int, default=5) ## optuna 실행 횟수 ##you need to change

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def objective(trial : Trial , args) :

    params = args.__dict__.copy()
    # optuna에서 탐색할 하이퍼파라미터만 업데이트(하이퍼파라미터 탐색 범위 지정
    #탐색할 범위 지정 
    # you need to change
    params.update( {
        # 'batch_size' : trial.suggest_int('batch_size', 8, 32, step=8), # fix 8
        'learning_rate' : trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        # 'max_epoch' : trial.suggest_int('max_epoch', 70, 150, step=20), #
        #'image_size' : trial.suggest_int('image_size', 1024, 2048, step=128),
        #'input_size' : trial.suggest_int('input_size', 512, 1024, step=64), 
        'optimizer_name' : trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW']),
    })
    try : 
        final_loss = do_training(**params , trial=trial , save_final=False)
        return final_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()
      

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, optimizer_name , trial , save_final = False, study_name = None, n_trials = None): #  optimizer_name, trial 추가
    
   
    
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


    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate) # optimizer_name 추가
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        avg_loss = epoch_loss / num_batches # 평균 손실 계산 

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            avg_loss, timedelta(seconds=time.time() - epoch_start)))
    
        # optuna 탐색 시 필요
        if trial is not None:
            trial.report(avg_loss, epoch)
            ## Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
        ## Save model
        if save_final:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'final_best_model.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        # if (epoch + 1) % save_interval == 0:
        #     if not osp.exists(model_dir):
        #         os.makedirs(model_dir)

        #     ckpt_fpath = osp.join(model_dir, 'latest.pth')
        #     torch.save(model.state_dict(), ckpt_fpath)

    return avg_loss 
# 추가 
def setup_logging(model_dir):
    """로깅 설정을 위한 함수"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 로그 디렉토리 생성
    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 파일 핸들러 설정
    log_file = os.path.join(log_dir, f'optuna_train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Optuna 로깅 설정
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    
    return logger

def main(args):

    
    ## optuna 실행
    # logging 설정
    logger = setup_logging(args.model_dir)
    logger.info("=== Training Start ===")
    logger.info(f"Arguments: {vars(args)}") #
    # Run optimization
    # optuna -dashboard 사용 시 필요
    study_name = f"{args.study_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        storage=f"sqlite:///{study_name}.db",
        study_name=study_name,
        direction = "minimize",
        pruner=optuna.pruners.MedianPruner(), #Sampling algorithm
    )
    
    
    # objective_with_args = partial(objective,args=args)
    study.optimize(
                lambda trial : objective(trial , args),
                n_trials=args.n_trials, # 실험횟수
                timeout=None, # 시간제한(sec)
    )

    df = study.trials_dataframe()
    df.to_csv(osp.join(args.model_dir, f"{study_name}.csv"), index=False)
    # 최적의 하이퍼파라미터 출력
    trial = study.best_trial
    logger.info("\nBest trial:")
    logger.info(f"  Value (Best Loss): {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
  
    ## 최적의 하이퍼파라미터로 최종 학습 진행
    logger.info("\nStarting final training with best parameters...")
    final_params = args.__dict__.copy()
    final_params.update(study.best_params)
    
    final_loss = do_training(
        **final_params,
        trial=None,  # 최종 학습에서는 trial 없음
        save_final=True  # 최종 모델 저장
    )
    logger.info(f"Final training completed with loss: {final_loss:.4f}")
    logger.info("=== Training End ===")

   
if __name__ == '__main__':
    args = parse_args()
    main(args)