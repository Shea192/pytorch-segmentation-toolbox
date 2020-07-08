#!/bin/bash
uname -a
#date
#env
date
ps -ef|grep train|awk '{print $2}'|xargs -i kill -9 {}

CMD=$1
CS_PATH=/dev/shm/citys
MODEL=$2
EDGE=$3
EXP=$4
LR=1e-2
WD=5e-4
BS=8
STEPS=40000
INPUT_SIZE=769,769
OHEM=0
GPU_IDS=0,1,2,3

#variable ${LOCAL_OUTPUT} dir can save data of you job, after exec it will be upload to hadoop_out path 
#resnet50
mkdir ./snapshots/${EXP}

if [[ "$1"x == "train"x ]]; then
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/resnet50-imagenet.pth --input-size ${INPUT_SIZE} --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --ohem ${OHEM} --with-edge ${EDGE} --edge-weight 10 --snapshot-dir ./snapshots/${EXP} 2>&1 | tee ./snapshots/${EXP}/${EXP}.log 
elif [[ "$1"x == "resume"x ]]; then
echo "resume traning"
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from $5 --start-iter $6 --input-size ${INPUT_SIZE} --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --ohem ${OHEM} --with-edge ${EDGE} --edge-weight 10 --snapshot-dir ./snapshots/${EXP} 2>&1 | tee ./snapshots/${EXP}/${EXP}.log 
else
  echo "$1"x" is invalid..."
fi
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS}
#CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 evaluate.py --data-dir ${CS_PATH} --model ${MODEL} --input-size ${INPUT_SIZE} --batch-size 4 --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0
