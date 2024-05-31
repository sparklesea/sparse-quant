import os
import torch.utils.cpp_extension

def patch_cpp_extension():
    # 找到 cpp_extension.py 文件路径
    cpp_extension_path = os.path.join(torch.utils.__path__[0],'cpp_extension.py')

    # 打印找到的文件路径
    print(f"cpp_extension.py path: {cpp_extension_path}")

    # 读取文件内容
    with open(cpp_extension_path, 'r') as file:
        lines = file.readlines()

    # 检查并注释掉第 280 到 282 行的代码
    start_line = 279  # 第 280 行（Python 的行数从 0 开始）
    end_line = 282    # 包含第 282 行
    for i in range(start_line, end_line):
        if not lines[i].strip().startswith("#"):
            lines[i] = "# " + lines[i]

    # 将修改后的内容写回文件
    with open(cpp_extension_path, 'w') as file:
        file.writelines(lines)

    print(f"Successfully patched {cpp_extension_path}")

if __name__ == "__main__":
    patch_cpp_extension()
