# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname dfpn8_gsf_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/dfpn8_gsf/dfpn8_gsf_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/dfpn8_gsf/dfpn8_gsf_res101_pcontext/model_best.pth.tar --split val --mode testval --ms