# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset cocostuff \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname dfpn7_gsf_res101_cocostuff

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset cocostuff \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/cocostuff/dfpn7_gsf/dfpn7_gsf_res101_cocostuff/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset cocostuff \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/cocostuff/dfpn7_gsf/dfpn7_gsf_res101_cocostuff/model_best.pth.tar --split val --mode testval --ms