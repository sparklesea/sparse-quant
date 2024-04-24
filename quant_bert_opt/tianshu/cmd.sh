# for bert
# fp16
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada
# quant+sparse
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada --w_bit 4 --a_bit 8 --mask_path masks/bert.pt

CUDA_VISIBLE_DEVICES=1 python main_bert.py --model_path bert_model/bert-large-cased-lambada --w_bit 4 --w_group_size 64 --lut_path masks/bert_large_lut.pt --output_path quantized_model/bert --w_group_size 64

CUDA_VISIBLE_DEVICES=1 python main_bert.py --model_path quantized_model/bert --w_bit 4 --w_group_size 64 --lut_path masks/bert_large_lut.pt --quantized --eval
# for opt
#fp16
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path /share/huangshan/opt-6.7b --tasks lambada_standard
#quant+sparse
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path /share/huangshan/opt-6.7b --tasks lambada_standard --mask_path masks/opt.pt --rep_file rep_file/facebook_opt-6.7b-smooth.pt --w_bit 4 --a_bit 8 --w_group_size 64 --a_group_size 64

CUDA_VISIBLE_DEVICES=1 python main_opt.py --model_path /share/huangshan/opt-6.7b --w_bit 4 --w_group_size 128 --output_path /home/huangshan/project/sparse-quant/quant_bert_opt/tianshu/quantized_model

CUDA_VISIBLE_DEVICES=1 python main_opt.py --model_path /home/huangshan/project/sparse-quant/quant_bert_opt/tianshu/quantized_model --w_bit 4 --w_group_size 128 --quantized --tasks lambada_standard