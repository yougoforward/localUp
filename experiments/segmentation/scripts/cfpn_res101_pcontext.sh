# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname cfpn_res101_pcontext --batch-size 16

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn/cfpn_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/cfpn/cfpn_res101_pcontext/model_best.pth.tar --split val --mode testval --ms