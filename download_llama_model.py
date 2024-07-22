
from huggingface_hub import snapshot_download

repo_id = "Linly-AI/Chinese-LLaMA-2-7B-hf"                                 # 模型在huggingface上的名称
local_dir = "/share/yuzhen/llama2-chinese"                               # 本地模型存储的地址
local_dir_use_symlinks = False               # 本地模型使用文件保存，而非blob形式保存
revision = ""                                # 模型的版本号
snapshot_download(repo_id=repo_id, 
                  local_dir=local_dir,
                  local_dir_use_symlinks=local_dir_use_symlinks)

