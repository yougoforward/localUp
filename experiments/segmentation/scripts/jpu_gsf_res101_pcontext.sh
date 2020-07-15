# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model jpu_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname jpu_gsf_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model jpu_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/jpu_gsf/jpu_gsf_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model jpu_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/jpu_gsf/jpu_gsf_res101_pcontext/model_best.pth.tar --split val --mode testval --ms