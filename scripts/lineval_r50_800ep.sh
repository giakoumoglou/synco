#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=64gb:ngpus=4
#PBS -lwalltime=48:00:00

cd $PBS_O_WORKDIR

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch

python main_lincls.py -a resnet50 --lr 30.0 --batch-size 256 --pretrained $EPHEMERAL/synco/synco_r50_800ep/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:19001' --multiprocessing-distributed --world-size 1 --rank 0 $HOME/datasets/imagenet/