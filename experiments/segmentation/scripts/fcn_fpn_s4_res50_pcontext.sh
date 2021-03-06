# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fcn_fpn_s4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname fcn_fpn_s4_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_s4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fcn_fpn_s4/fcn_fpn_s4_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_s4 --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fcn_fpn_s4/fcn_fpn_s4_res50_pcontext/model_best.pth.tar --split val --mode testval --ms