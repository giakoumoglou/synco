#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=64gb:ngpus=4
#PBS -lwalltime=48:00:00

cd $PBS_O_WORKDIR

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate torch

PORT=$((10000 + RANDOM % 10000))

python main_lincls.py \
	-a resnet50 \
	--lr 30.0 \
	--batch-size 256 \
	--pretrained $EPHEMERAL/synco/syncov2_r50_200ep/checkpoint_0199.pth.tar \
	--dist-url 'tcp://localhost:10501' \
	--multiprocessing-distributed \
	--world-size 1 --rank 0 \
	$HOME/datasets/imagenet/