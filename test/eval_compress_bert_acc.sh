CUDA_VISIBLE_DEVICES=7 python nvidia/bert_LAMBADA.py --model_path bert_model/bert-large-cased-lambada --lut_path masks/bert_large_lut.pt --w_bit 4 --eval --fake
