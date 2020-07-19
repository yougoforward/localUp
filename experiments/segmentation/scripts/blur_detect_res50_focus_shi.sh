# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset focus_shi \
    --model blur_detect --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --checkname blur_detect_res50_focus_shi

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset focus_shi \
    --model blur_detect --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/focus_shi/blur_detect/blur_detect_res50_focus_shi/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset focus_shi \
    --model blur_detect --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/focus_shi/blur_detect/blur_detect_res50_focus_shi/model_best.pth.tar --split val --mode testval --ms

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset focus_shi \
    --model blur_detect --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/focus_shi/blur_detect/blur_detect_res50_focus_shi/model_best.pth.tar --split val --mode test --ms --save-folder experiments/segmentation/results/blur_detect_res50