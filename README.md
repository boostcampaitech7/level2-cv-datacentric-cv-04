## Image Object detection competition

카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 

이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행합니다.

본 대회에서는 글자 검출만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

본 대회는 제출된 예측 (prediction) 파일로 평가합니다.

## Result
Public: 0.9030
Private: 0.8983

### Wrap-up Report

[level2_CV기초대회3_CV_팀 리포트(04조).docx.pdf](https://github.com/user-attachments/files/17736758/level2_CV.3_CV_.04.docx.pdf)


## Data

데이터 셋은 AI Stage 제공 데이터를 사용하였으며 라이센스에 따라 외부 공개가 어렵습니다.

데이터셋 디렉토리 구조는 아래와 같습니다.

해당 파일에는 chinese_receipt, japanses_receipt, thai_receipt, vietnamese_receipt 이라는 하위 경로에 4개 국어로 작성된 다국어 영수증 데이터셋이 제공됩니다.

각각 폴더의 하위 img/train 경로에 학습 이미지가, 하위 img/test 경로에는 테스트 이미지가 있습니다.

각각 폴더의 하위 ufo 경로에는 학습 이미지에 대한 UFO 형식의 annotation 파일인 train.json, 테스트 이미지에 대한 UFO 형식의 test.json이 있습니다.

마지막으로, 캠퍼 분들이 참고하실 수 있는 제출 예시 파일인 sample_submission.csv가 있습니다. 참고로 이 파일의 확장자는 csv이지만, 내부 구조는 test.json과 동일한 UFO 형식입니다.

데이터셋 세부 정보
학습 데이터셋은 언어당 100장, 총 400장의 영수증 이미지 파일 (.jpg)과 해당 이미지에 대한 주석 정보를 포함한 UFO (.json) 파일로 구성되어 있습니다. 각 UFO의 ‘images’키에 해당하는 JSON 값은 각 이미지 파일의 텍스트 내용과 텍스트 좌표를 포함하고 있습니다.




## Model 

[EAST](https://arxiv.org/abs/1704.03155) 모델을 사용하였으며 대회 규정 상 다른 모델은 사용하지 않았습니다.

## Usage

### Installation

1. Clone the repository & download data:
   ```
   git clone https://github.com/boostcampaitech7/level2-datacentric-cv-04.git
   cd level2-datacentric-cv-04
   ```
2. setup
   ```
   chmod +x server_setting.sh
   ./server_setting.sh
   ```

### Training

```
python train.py --data_dir=./data --model_dir=./trained_models --image_size=2048 --input_size=1024 --batch_size=8 learning_rate=1e-3 --max_epoch=100
```

hyperparameter는 원하는 만큼 수정할 수 있습니다.

### Inference


```
python inference.py --data_dir=./data --output_dir --input_size=2048 --batch_size=8
```

## Project Structure

프로젝트 폴더 구조를 맞춰야 상대경로로 인식됩니다. 

또한 east_dataset.py, loss.py, model.py, detect.py는 대회 종료에 따라 더이상 제공하지 못할 수 있습니다.

```
** project structure **

level2-cv-datacentric-cv-04/
├── data/
│   ├──chinese_receipt/
│   │   ├── img
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── ufo
│   │       ├── train.json
│   │       └── test.json
│   │
│   ├──japanese_receipt/ 
│   │   ├── img
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── ufo
│   │       ├── train.json
│   │       └── test.json
│   │
│   ├──thai_receipt/ 
│   │   ├── img
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── ufo
│   │       ├── train.json
│   │       └── test.json
│   │
│   ├── vietnamese_receipt/
│   │   ├── img
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── ufo
│   │       ├── train.json
│   │       └── test.json 
│   │
│   └── merged_receipts/
│       ├── images
│       │   ├── train/
│       │   └── val/
│       ├── train.json
│       └── val.json
│
│
├── streamlit/
│   ├── pages/
│   │   ├── data_EDA.py
│   │   └── data_viewer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── streamlit_data_loader.py
│   ├── main.py
│   └── st_requierment.txt
│
├──utils/
│   ├── add_language.py
│   └── prepare_dataset.py
│
├── pths/
├── predictions/
├── trained_models/
│
├── dataset.py
├── deteval.py
├── east_dataset.py
├── train.py
├── inference.py
├── requirements.txt
├── loss.py
├── model.py
├── detect.py
└── README.md
```

   
