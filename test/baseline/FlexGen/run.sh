#!/bin/bash

for ((i=0;i<20;i+=1))
do
    CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 8 --gen-len 2 --gpu-batch-size 1
    CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 16 --gen-len 2 --gpu-batch-size 1
    CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 32 --gen-len 2 --gpu-batch-size 1
    CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 64 --gen-len 2 --gpu-batch-size 1
    CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 128 --gen-len 2 --gpu-batch-size 1
done

# CUDA_VISIBLE_DEVICES=1 python3 -m flexgen.flex_opt --model $MODEL_OPT_PATH --local --percent 75 25 100 0 100 0 --prompt-len 512 --gen-len 512 --gpu-batch-size 1