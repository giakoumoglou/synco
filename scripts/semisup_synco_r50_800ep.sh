#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=96gb:ngpus=4
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch

learning_rates=(0.01 0.02 0.05 0.1 0.005)

for lr in "${learning_rates[@]}"; do
    echo "========== LR: $lr, Percentage 1% ==========="
    python main_semisup.py -a resnet50 --lr-backbone $lr --lr-classifier 0.5 --epochs 60 --train-percent 1 --weights finetune --batch-size 1024 --pretrained $EPHEMERAL/synco/synco_r50_800ep/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:17002' --multiprocessing-distributed --world-size 1 --rank 0 $HOME/datasets/imagenet/
    
    echo "========== LR: $lr, Percentage 10% ==========="
    python main_semisup.py -a resnet50 --lr-backbone $lr --lr-classifier 0.5 --epochs 30 --train-percent 10 --weights finetune --batch-size 1024 --pretrained $EPHEMERAL/synco/synco_r50_800ep/checkpoint_0799.pth.tar --dist-url 'tcp://localhost:17002' --multiprocessing-distributed --world-size 1 --rank 0 $HOME/datasets/imagenet/
done