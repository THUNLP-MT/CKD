#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# PATH=/usr/local/cuda-10.1/bin:$PATH 
# LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 

module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
source activate thumtc

CODE_PATH=/data/home/scv0107/run/zyc/THUMTA

DATA_PATH=/data/home/scv0107/run/zyc/125w-zip 
TRAIN_PATH=${DATA_PATH}
TEST_PATH=${DATA_PATH}/dev_test 
OUTPUT_PATH=/data/home/scv0107/run/zyc/output/test1
mkdir -p ${OUTPUT_PATH}

OUTPUT_NAME=nist02.32k.en.trans

PYTHONPATH=$CODE_PATH \
python3 -u ${CODE_PATH}/thumt/bin/translator.py \
  --models transformer \
  --input ${TEST_PATH}/nist02/nist02.32k.zh \
  --output ${OUTPUT_PATH}/${OUTPUT_NAME} \
  --vocabulary ${TRAIN_PATH}/vocab.32k.zh.txt ${TRAIN_PATH}/vocab.32k.en.txt \
  --checkpoints ${OUTPUT_PATH}/eval \
  --parameters=device_list=[0,1,2,3],decode_alpha=1.2

sed -r 's/(@@ )|(@@ ?$)//g' < ${OUTPUT_PATH}/${OUTPUT_NAME} > ${OUTPUT_PATH}/${OUTPUT_NAME}.rbpe
perl multi-bleu.perl ${TEST_PATH}/nist02/nist02.en < ${OUTPUT_PATH}/${OUTPUT_NAME}.rbpe > ${OUTPUT_PATH}/${OUTPUT_NAME}.bleu