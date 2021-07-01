#!/bin/bash
# eval
#CMD="python3 run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin"

# training from scratch
CMD='python3 run.py -e 80 -k cpn_ft_h36m_dbb -arc 1'
echo $CMD
$CMD
