# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fcn_fpn_2048 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname fcn_fpn_2048_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_2048 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fcn_fpn_2048/fcn_fpn_2048_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_2048 --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/fcn_fpn_2048/fcn_fpn_2048_res101_pcontext/model_best.pth.tar --split val --mode testval --ms