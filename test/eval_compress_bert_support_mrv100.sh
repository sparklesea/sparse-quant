
CUDA_VISIBLE_DEVICES=6 python bert_infer.py --model_path quantized_model/bert --w_bit 4 --lut_path masks/bert_large_lut.pt --quantized
