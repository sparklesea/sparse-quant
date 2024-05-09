# CUDA_VISIBLE_DEVICES=6 python nvidia/opt_LAMBADA_fake_quant.py --model_path /share/huangshan/opt-6.7b --mask_path /share/huangshan/masks/opt.pt --w_bit 4 --tasks lambada_standard

CUDA_VISIBLE_DEVICES=6 python nvidia/opt_LAMBADA.py --model_path /share/huangshan/opt-6.7b --lut_path masks/opt_lut_density_26.pt --w_bit 4 --tasks lambada_standard --fake

# CUDA_VISIBLE_DEVICES=6 python nvidia/opt_LAMBADA_fake_quant.py --model_path /share/huangshan/opt-6.7b --mask_path /share/huangshan/masks/opt.pt --w_bit 4 --tasks lambada_standard --fake
