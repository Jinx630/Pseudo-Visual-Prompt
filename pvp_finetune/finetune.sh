#!/bin/bash
cd Dassl.pytorch-master
python setup.py develop
cd ../scripts
sh main_coco2014.sh