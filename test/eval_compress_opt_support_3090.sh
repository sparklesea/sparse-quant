
CUDA_VISIBLE_DEVICES=6 python opt_infer.py --model_path quantized_model/opt --lut_path masks/opt_lut_density_26.pt --w_bit 4 --quantized