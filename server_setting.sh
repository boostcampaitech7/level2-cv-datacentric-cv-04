#!/bin/bash


# 패키지 업데이트 및 wget 설치
echo "패키지 목록 업데이트 및 wget 설치 중..."
apt-get update && apt-get install -y wget

# build-essential 설치
echo "build-essential 설치 중..."
apt-get install -y build-essential

# requirements.txt 파일 설치
echo "requirements.txt 파일 설치 중..."
pip install -r requirements.txt

# opencv 관련 오류 해결
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0



echo "설치가 완료되었습니다." 


# 사용법
# chmod +x server_setting.sh
# ./server_setting.sh