#!/bin/bash
#PBS -lselect=1:ncpus=2:mem=8gb
#PBS -lwalltime=48:00:00
cd $PBS_O_WORKDIR

mkdir coco
cd coco

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
