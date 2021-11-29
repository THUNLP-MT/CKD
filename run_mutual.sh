#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# PATH=/usr/local/cuda-10.1/bin:$PATH 
# LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 

module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
source activate thumtc


PYTHONPATH=. \
python3 -u ./thumt/bin/trainer_mutual.py \
  --input ../125w-zip/train.zh.32k.shuf ../125w-zip//train.en.32k.shuf \
  --vocabulary ../125w-zip/vocab.32k.zh.txt ../125w-zip/vocab.32k.en.txt \
  --model transformer \
  --validation ../125w-zip/dev_test/nist06/nist06.32k.zh \
  --references "../125w-zip/dev_test/nist06/nist06.en*" \
  --output dadada \
  --parameters=batch_size=8192,device_list=[0,1,2,3,4,5,6,7],update_cycle=2,keep_top_checkpoint_max=1,eval_steps=10 \
  --hparam_set bases \
  2>&1 #| tee ../output/test1/`date '+%Y-%m-%d-%H-%M-%S'`