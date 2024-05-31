# optbert

### Requirements
1. anaconda
    ```
    # 设置代理
    vim ~/.bashrc
    # 然后在末尾写入：
        # 3090
        export HTTP_PROXY='http://10.10.20.100:1088'
        export HTTPS_PROXY='http://10.10.20.100:1089'
        # muxi, tianshu
        export HTTP_PROXY='http://172.18.10.12:1089'
        export HTTPS_PROXY='http://172.18.10.12:1089'
    # 保存退出后
    source ~/.bashrc
    
    # 用清华源下载会快一些，官方源下载很慢
    wget -U NoSuchBrowser/1.0 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
    # 安装Anaconda
    sh Anaconda3-2023.07-2-Linux-x86_64.sh
    # 需要接受许可证
    Do you accept the license terms? [yes|no]
    [no] >>> yes
    # 最好选择yes，否则要手动设置环境变量
    Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]
    [no] >>> yes
    # 重新登录终端，就有conda命令了，可以查看一下版本
    conda -V

    # 编辑conda配置文件
    vim ~/.condarc
    # 将下面的代码块中的内容粘贴进去，然后清除相关缓存
    channels:
        - defaults
    show_channel_urls: true
    default_channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
    custom_channels:
        conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
        deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
        nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud

    conda clean -i
    conda clean -p
    conda clean -a

    #创建test环境
    conda create -n test python=3.9
    conda activate test
    ```
3. cuda for 3090
    ```
    conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit
    ```
4. torch
    ```
    # for 3090
    pip install torch

    # for muxi
    pip install /home/public/mxc500-2.19.2.23/wheel/torch-2.0.0+mc2.19.2.23-cp38-cp38-linux_x86_64.whl 

    # for tianshu
    pip install /share/huangshan/torch-2.1.1+corex.4.0.0-cp39-cp39-linux_x86_64.whl 
    ```
5. triton for tianshu
    ```
    pip install /share/huangshan/triton-2.1.0+corex.4.0.0-cp39-cp39-linux_x86_64.whl 
    ```
6. requirements
    ```
    pip install -r requirements.txt
    ```
7. patch
    python test/patch.py

### Pre-installation
1. lm_eval
    ```
    conda install git
    git clone -b v0.3.0 https://github.com/EleutherAI/lm-evaluation-harness.git
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
2.  install the quantization kernel "quant"
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
CUDA_VISIBLE_DEVICES=0 python muxi/opt_infer.py --model_path $MODEL_OPT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/opt --rep_file /home/public/rep_file/facebook_opt-6.7b-smooth.pt

## tianshu
CUDA_VISIBLE_DEVICES=0 python tianshu/opt_infer.py --model_path $MODEL_OPT_PATH --w_bit 4 --w_group_size 64 --output_path quantized_model/opt --rep_file /share/huangshan/rep_file/facebook_opt-6.7b-smooth.pt
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
if you want to choose which sentences to sample
```
bash {your_expected}_support_{your_env}.sh --sample {Sequence number separated by blank space, e.g., 0 2 4 8 9} 
```
