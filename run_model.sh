#!/bin/sh
model="1 2 3 4 5 6 7 8"

for mdls in $model;
do
    CUDA_VISIBLE_DEVICES=1 python 0_main.py --model $mdls 
    #python 1_sample_gen.py --model $mdls 
done
