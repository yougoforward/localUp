# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model pano_fpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname pano_fpn_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model pano_fpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/pano_fpn/pano_fpn_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model pano_fpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/pano_fpn/pano_fpn_res101_pcontext/model_best.pth.tar --split val --mode testval --ms