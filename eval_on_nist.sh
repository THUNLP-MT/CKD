#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# PATH=/usr/local/cuda-10.1/bin:$PATH 
# LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 

module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
source activate thumtc

OUTPUT_PATH=/data/home/scv0107/run/zyc/output/base_75w_b
mkdir -p ${OUTPUT_PATH}

OUTPUT_NAME=nist04.32k.en.trans

PYTHONPATH=. \
python3 -u ./thumt/bin/translator.py \
  --models transformer \
  --input ../125w-zip/dev_test/nist04/nist04.32k.zh \
  --output ${OUTPUT_PATH}/${OUTPUT_NAME} \
  --vocabulary /data/home/scv0107/run/zyc/125w-zip/vocab.32k.zh.txt /data/home/scv0107/run/zyc/125w-zip/vocab.32k.en.txt \
  --checkpoints ${OUTPUT_PATH}/eval \
  --parameters=device_list=[0,1,2,3,4,5,6,7],decode_alpha=1.2

sed -r 's/(@@ )|(@@ ?$)//g' < ${OUTPUT_PATH}/${OUTPUT_NAME} > ${OUTPUT_PATH}/${OUTPUT_NAME}.rbpe
perl multi-bleu.perl /data/home/scv0107/run/zyc/125w-zip/dev_test/nist04/nist04.en < ${OUTPUT_PATH}/${OUTPUT_NAME}.rbpe > ${OUTPUT_PATH}/${OUTPUT_NAME}.bleu