#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=64gb:ngpus=4
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch

python main_synco.py -a resnet50 --lr 0.03 --batch-size 256 --epochs 200 --resume ./checkpoints/synco_r50_200ep/checkpoint_0168.pth.tar --save-dir ./checkpoints/synco_r50_200ep/ --dist-url 'tcp://localhost:10201' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos --n-hard 1024 --n1 256 --n2 256 --n3 256 --n4 64 --n5 64 --n6 64 ./datasets/imagenet/
