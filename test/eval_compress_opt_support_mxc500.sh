
# 初始化变量  
samples=()
  
# 解析命令行参数  
while [ "$#" -gt 0 ]; do  
    case "$1" in  
        --sample)  
            shift  # 移除 --sample  
            # 读取所有后续参数作为samples，直到没有参数或遇到另一个选项为止  
            while [ "$#" -gt 0 ] && [ "${1:0:1}" != "-" ]; do  
                samples+=("$1")  
                shift  
            done  
            # 因为我们已经读取了所有后续的样本值，所以不需要继续解析选项  
            break  
            ;;  
        *)  
            echo "Unknown option: $1"  
            exit 1  
            ;;  
    esac  
done  
  
# 准备传递给py_test.py的参数  
py_args=()  
  
# 如果samples数组不为空，则添加--sample参数和样本值  
if [ ${#samples[@]} -gt 0 ]; then  
    py_args+=("--sample")  
    py_args+=("${samples[@]}")  
fi  


python muxi/opt_infer.py --model_path quantized_model/opt --lut_path /home/public/masks/opt_lut_density_26.pt --w_bit 4 --quantized --eval "${py_args[@]}"
