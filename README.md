# optbert

### Requirements
torch == 2.2.1
transformers == 4.36.2
triton >= 2.1.0
### Pre-installation
1. lm_eval
    ```
    git clone -b v0.3.0 git@github.com:EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness/
    pip install -e .
    ```
2. FlexGen
    ```
    cd test/baseline/FlexGen
    pip install -e .
    ```
### Installation
#### install sparse kernel
1.  cd your GPU environment
    ```
    cd sparse-attention/{your GPU environment}/
    ```
2.  install the look up table attention "playground"
    ```
    python setup.py install
    ```
#### install quant kernel
1.  cd your GPU environment
    ```
    cd quantization/{your GPU environment}/
    ```
2.  install the look up table attention "playground"
    ```
    python setup.py install
    ```

## test
<!-- ### 1. First check `test` folder has 

    bert model: `bert_model/bert-large-cased-lambada`

    lut mask: `masks/` -->

### 1. Set the env variables
```
# if in the test machine

#nvidia or tianshu
export MODEL_OPT_PATH=/share/huangshan/opt-6.7b
export MODEL_BERT_PATH=/share/huangshan/bert-large-cased-lambada

#muxi
export MODEL_OPT_PATH=/home/public/models/opt-6.7b
export MODEL_BERT_PATH=/home/public/models/bert-large-cased-lambada

#else
export MODEL_OPT_PATH={opt model path}
export MODEL_BERT_PATH={bert model path}
```
### 2. Generate quantized models

```
# for opt
## nvidia
CUDA_VISIBLE_DEVICES=0 python nvidia/opt_infer.py --model_path $MODEL_OPT_PATH --w_bit 4 --output_path quantized_model/opt

## muxi
CUDA_VISIBLE_DEVICES=0 python muxi/opt_infer.py --model_path $MODEL_OPT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/opt

## tianshu
CUDA_VISIBLE_DEVICES=0 python tianshu/opt_infer.py --model_path $MODEL_OPT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/opt
```

```
# for bert
## nvidia
CUDA_VISIBLE_DEVICES=0 python nvidia/bert_infer.py --model_path $MODEL_BERT_PATH --w_bit 4 --output_path quantized_model/bert

## muxi
CUDA_VISIBLE_DEVICES=0 python muxi/bert_infer.py --model_path $MODEL_BERT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/bert

## tianshu
CUDA_VISIBLE_DEVICES=0 python tianshu/bert_infer.py --model_path $MODEL_BERT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/bert
```
### 3. Now you can start your test using 
```
bash {your_expected}.sh 
```
if you want tot choose which sentences to sample
```
bash {your_expected}_support_{your_env}.sh --sample {Sequence number separated by commas, e.g., 0,2,4,8,9} 
```