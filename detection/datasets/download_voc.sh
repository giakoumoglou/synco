#!/bin/bash

mkdir -p VOC2007 VOC2012

wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P VOC2007
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P VOC2007
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P VOC2012

tar -xf VOC2007/VOCtrainval_06-Nov-2007.tar -C VOC2007
tar -xf VOC2007/VOCtest_06-Nov-2007.tar -C VOC2007
tar -xf VOC2012/VOCtrainval_11-May-2012.tar -C VOC2012
