#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR

# Creating directory structure
mkdir -p VOC2007 VOC2012

# Download VOC 2007 TrainVal
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P VOC2007
# Download VOC 2007 Test
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P VOC2007

# Download VOC 2012 TrainVal
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P VOC2012

# Extracting archives
tar -xf VOC2007/VOCtrainval_06-Nov-2007.tar -C VOC2007
tar -xf VOC2007/VOCtest_06-Nov-2007.tar -C VOC2007
tar -xf VOC2012/VOCtrainval_11-May-2012.tar -C VOC2012

echo "Dataset downloads and extraction complete."