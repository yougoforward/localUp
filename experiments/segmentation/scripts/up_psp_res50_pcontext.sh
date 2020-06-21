# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model up_psp --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname up_psp_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model up_psp --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/up_psp/up_psp_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset pcontext \
    --model up_psp --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/up_psp/up_psp_res50_pcontext/model_best.pth.tar --split val --mode testval --ms