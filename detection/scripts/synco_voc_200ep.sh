#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=256gb:ngpus=8
#PBS -lwalltime=24:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate detectron2_cuda10

python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 8 MODEL.WEIGHTS ./resnet50_synco_200ep.pkl
