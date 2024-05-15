CUDA_VISIBLE_DEVICES=0 python nvidia/opt_LAMBADA.py --model_path $MODEL_OPT_PATH --lut_path /share/huangshan/masks/opt_lut_density_26.pt --w_bit 4 --tasks lambada_standard --fake
