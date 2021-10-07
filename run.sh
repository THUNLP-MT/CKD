# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PATH=/usr/local/cuda-10.1/bin:$PATH 
LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 
CODE_PATH=/data/disk5/private/zyc/THUMTD/thumt

DATA_PATH=/data/disk5/private/zyc/en_zh_data  
TRAIN_PATH=${DATA_PATH}/train 
TEST_PATH=${DATA_PATH}/dev_test 

PYTHONPATH=$CODE_PATH:$PYTHONPATH \
python3 ${CODE_PATH}/bin/trainer.py \
  --input ${TRAIN_PATH}/train.32k.zh.shuf ${TRAIN_PATH}/train.32k.en.shuf \
  --vocabulary ${TRAIN_PATH}/vocab.32k.zh.txt ${TRAIN_PATH}/vocab.32k.en.txt \
  --model transformer \
  --validation ${TEST_PATH}/nist06/nist06.32k.zh \
  --references ${TEST_PATH}/nist06/nist06.32k.en0 \
  --output /data/disk5/private/zyc/output \
  --parameters=batch_size=4096,device_list=[0,1,2,3,4,5,6,7],update_cycle=2,keep_top_checkpoint_max=1 \
  --hparam_set base