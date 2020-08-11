# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset ade20k \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname dfpn7_gsf_res101_ade20k

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset ade20k \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/ade20k/dfpn7_gsf/dfpn7_gsf_res101_ade20k/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset ade20k \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/ade20k/dfpn7_gsf/dfpn7_gsf_res101_ade20k/model_best.pth.tar --split val --mode testval --ms