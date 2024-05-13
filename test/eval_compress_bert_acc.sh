CUDA_VISIBLE_DEVICES=7 python nvidia/bert_LAMBADA.py --model_path $MODEL_BERT_PATH --lut_path /share/huangshan/masks/bert_large_lut.pt --w_bit 4 --eval --fake
