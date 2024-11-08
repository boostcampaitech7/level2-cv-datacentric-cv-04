#!/bin/bash

# wildreceipt dataset 다운로드 및 압축 해제
wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar
tar -xvf wildreceipt.tar

cd utils

python wild2ufo.py

python prepare_dataset.py --external_data

cd ..
