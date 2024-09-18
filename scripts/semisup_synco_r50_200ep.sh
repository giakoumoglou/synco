#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4
#PBS -lwalltime=48:00:00

cd $PBS_O_WORKDIR

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch

python main_semisup.py -a resnet50 --lr-backbone 0.005 --lr-classifier 0.5 --train-percent 1 --weights finetune --batch-size 256 --pretrained ./results/synco_r50_200ep/checkpoint_0199.pth.tar --dist-url 'tcp://localhost:16001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/

python main_semisup.py -a resnet50 --lr-backbone 0.005 --lr-classifier 0.5 --train-percent 10 --weights finetune --batch-size 256 --pretrained ./results/synco_r50_200ep/checkpoint_0199.pth.tar --dist-url 'tcp://localhost:16001' --multiprocessing-distributed --world-size 1 --rank 0 ../../datasets/imagenet/