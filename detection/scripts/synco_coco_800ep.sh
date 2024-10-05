#!/bin/bash
#PBS -lselect=1:ncpus=32:mem=256gb:ngpus=4
#PBS -lwalltime=72:00:00

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate detectron2_cuda10

echo "Checking NVIDIA GPU status with nvidia-smi:"
nvidia-smi
echo "Checking CUDA compiler version with nvcc:"
nvcc --version

python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 4 MODEL.WEIGHTS ./resnet50_synco_800ep.pkl OUTPUT_DIR ./output/coco_800ep/