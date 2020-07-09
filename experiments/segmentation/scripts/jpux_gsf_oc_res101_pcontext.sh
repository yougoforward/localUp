# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_oc --dataset pcontext \
    --model jpux_gsf_oc --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname jpux_gsf_oc_res101_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model jpux_gsf_oc --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/jpux_gsf_oc/jpux_gsf_oc_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model jpux_gsf_oc --aux --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume experiments/segmentation/runs/pcontext/jpux_gsf_oc/jpux_gsf_oc_res101_pcontext/model_best.pth.tar --split val --mode testval --ms