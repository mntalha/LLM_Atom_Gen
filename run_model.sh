#!/bin/sh
model="0 1 2 3 4 5 6 7 8"
for mdls in $model;
do
    python 0_main.py --model $mdls 
done
