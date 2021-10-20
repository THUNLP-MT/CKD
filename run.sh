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
LOGDIR=${OUTPUT_PATH}/`date '+%Y-%m-%d-%H-%M-%S'`

PYTHONPATH=$CODE_PATH \
python3 -u ${CODE_PATH}/thumt/bin/trainer.py \
  --input ${TRAIN_PATH}/train.zh.32k.shuf ${TRAIN_PATH}/train.en.32k.shuf \
  --vocabulary ${TRAIN_PATH}/vocab.32k.zh.txt ${TRAIN_PATH}/vocab.32k.en.txt \
  --model transformer \
  --validation ${TEST_PATH}/nist06/nist06.32k.zh \
  --references "${TEST_PATH}/nist06/nist06.en*" \
  --output ${OUTPUT_PATH} \
  --parameters=batch_size=16384,device_list=[0],update_cycle=2,keep_top_checkpoint_max=1,eval_steps=500 \
  --hparam_set bases 2>&1 | tee ${LOGDIR}