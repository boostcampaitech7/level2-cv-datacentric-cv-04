#!/bin/bash

# wandb 설치
pip install wandb

# wandb 로그인
echo "Weights & Biases(wandb) 로그인을 시작합니다..."
echo "wandb.ai에서 발급받은 API 키를 입력해주세요."
wandb login 