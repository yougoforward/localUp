# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pcontext \
    --model fpn_enc --aux --se-loss --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname fpn_enc_res50_pcontext

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_enc --aux --se-loss --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fpn_enc/fpn_enc_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pcontext \
    --model fpn_enc --aux --se-loss --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pcontext/fpn_enc/fpn_enc_res50_pcontext/model_best.pth.tar --split val --mode testval --ms