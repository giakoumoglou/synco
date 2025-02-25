#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=256gb:ngpus=8
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate detectron2_cuda10

python train_net.py --config-file configs/coco_R_50_C4_2x_synco.yaml --num-gpus 8 MODEL.WEIGHTS ./resnet50_synco_200ep.pkl OUTPUT_DIR ./output/coco_200ep/
