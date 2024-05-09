# optbert


## sparse-attention
triton >= 2.1.0
### pre-installation
1. lm_eval
    ```
    git clone -b v0.3.0 git@github.com:EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness/
    pip install -e .
    ```
### Installation
1.  cd your GPU environment
    ```
    cd sparse-attention/{your GPU environment}/
    ```
2.  install the look up table attention "playground"
    ```
    python setup.py install
    ```

### Run
1.  cd test path
    ```
    cd quant_bert_opt/{your GPU environment}/
    ```
For original model
```
# for opt
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path {your opt-6.7b model path} --tasks lambada_standard
# for bert
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada
```

For look up table, add `--lut_path` option
```
# for opt
CUDA_VISIBLE_DEVICES=0 python main_opt.py --model_path {your opt-6.7b model path} --tasks lambada_standard --lut_path masks/opt_lut_density_26.pt
# for bert
CUDA_VISIBLE_DEVICES=0 python main_bert.py --model_path bert_model/bert-large-cased-lambada --lut_path masks/bert_large_lut.pt
```
