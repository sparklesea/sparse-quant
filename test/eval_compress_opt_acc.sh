# CUDA_VISIBLE_DEVICES=0 python nvidia/opt_LAMBADA.py --model_path $MODEL_OPT_PATH --lut_path /share/huangshan/masks/opt_lut_density_26.pt --w_bit 4 --w_group_size 64 --rep_file /share/huangshan/rep_file/facebook_opt-6.7b-smooth.pt --tasks lambada_standard --fake

# CUDA_VISIBLE_DEVICES=0 python nvidia/opt_LAMBADA.py --model_path $MODEL_OPT_PATH --mask_path /share/huangshan/masks/opt.pt --w_bit 4 --w_group_size 64 --tasks lambada_standard --fake --rep_file /share/huangshan/rep_file/facebook_opt-6.7b-smooth.pt

CUDA_VISIBLE_DEVICES=0 python nvidia/opt_LAMBADA.py --model_path $MODEL_OPT_PATH --mask_path /share/huangshan/masks/opt.pt --w_bit 4 --w_group_size 64 --tasks lambada_standard --rep_file /share/huangshan/rep_file/facebook_opt-6.7b-smooth.pt --fake