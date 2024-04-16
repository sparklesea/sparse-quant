# for bert
# fp16
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada
# quant+sparse
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada --w_bit 4 --a_bit 8 --mask_path masks/bert.pt

# for opt
#fp16
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path facebook/opt-6.7b --tasks lambada_standard
#quant+sparse
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path facebook/opt-6.7b --tasks lambada_standard --mask_path masks/opt.pt --rep_file rep_file/facebook_opt-6.7b-smooth.pt --w_bit 4 --a_bit 8 --w_group_size 64 --a_group_size 64