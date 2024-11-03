# 데이터 다운로드
pip install gdown
pip install unzip

gdown 1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI

# 데이터 압축 해제
unzip CORD-1k-001.zip

# CORD 데이터를 UFO 포맷으로 변환
python utils/cord2ufo.py

# train, validation에 영어 데이터 추가
python utils/prepare_dataset.py --external_data

# 압축 파일 삭제
rm -rf CORD-1k-001.zip

echo "영어 영수증(CORD) 데이터 준비가 완료되었습니다."

# 사용법
# sh cord_dataset.sh