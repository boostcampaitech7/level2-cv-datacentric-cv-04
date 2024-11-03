#!/bin/bash


# 패키지 업데이트 및 wget 설치
echo "패키지 목록 업데이트 및 wget 설치 중..."
apt-get update && apt-get install -y wget

# build-essential 설치
echo "build-essential 설치 중..."
apt-get install -y build-essential

# 코드 다운로드
git clone https://github.com/boostcampaitech7/level2-cv-datacentric-cv-04.git
cd level2-cv-datacentric-cv-04

# requirements.txt 파일 설치
echo "requirements.txt 파일 설치 중..."
pip install -r requirements.txt
cd ..

# opencv 관련 오류 해결
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0


# 데이터 다운로드
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20240912160112/data.tar.gz
tar -zxvf data.tar.gz

# 코드 일부 추가
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20241030035533/code.tar.gz
tar -zxvf code.tar.gz

# 코드 일부 와 데이터 이동
mv code/detect.py code/loss.py code/model.py code/east_dataset.py ./level2-cv-datacentric-cv-04/
mv code/pths ./level2-cv-datacentric-cv-04/

mv data/ ./level2-cv-datacentric-cv-04/




echo "설치가 완료되었습니다." 


# 사용법
# sh server_setting.sh