#!/bin/bash
#python3 super.py -d humaneva15 -k detectron_pt_coco -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -b 128 -e 1000 -lrd 0.996 -lr 0.0005 -a Walk,Jog,Box --by-subject --export-training-curves 
#CMD="python3 run.py -d humaneva15 -k detectron_pt_coco -str Train/S1,Train/S2,Train/S3 -ste Validate/S1,Validate/S2,Validate/S3 -a Walk,Jog,Box --by-subject -c checkpoint --evaluate pretrained_humaneva15_detectron.bin"
CMD="python3 run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin"

echo $CMD
$CMD
