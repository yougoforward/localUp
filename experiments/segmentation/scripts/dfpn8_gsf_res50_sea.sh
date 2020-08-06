# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset sea \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname dfpn8_gsf_res50_sea

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/dfpn8_gsf/dfpn8_gsf_res50_sea/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model dfpn8_gsf --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/dfpn8_gsf/dfpn8_gsf_res50_sea/model_best.pth.tar --split val --mode testval --ms