# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fcn_fpn_nobn --aux --base-size 520 --crop-size 480 \
    --backbone resnet50 --checkname fcn_fpn_nobn_480_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_nobn --aux --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fcn_fpn_nobn/fcn_fpn_nobn_480_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fcn_fpn_nobn --aux --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fcn_fpn_nobn/fcn_fpn_nobn_480_res50_pcontext/model_best.pth.tar --split val --mode testval --ms