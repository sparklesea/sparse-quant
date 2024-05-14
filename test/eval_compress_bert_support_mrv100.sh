
rm -rf ~/.triton

python tianshu/bert_infer.py --model_path quantized_model/bert --w_bit 4 --lut_path /share/huangshan/masks/bert_large_lut.pt --quantized --eval 
