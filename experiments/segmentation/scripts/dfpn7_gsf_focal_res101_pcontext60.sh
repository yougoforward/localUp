# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext60 \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname dfpn7_gsf_res101_pcontext60

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext60 \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext60/dfpn7_gsf/dfpn7_gsf_res101_pcontext60/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext60 \
    --model dfpn7_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext60/dfpn7_gsf/dfpn7_gsf_res101_pcontext60/model_best.pth.tar --split val --mode testval --ms