import torch
import torch.utils
import torch.utils.cpp_extension
print("当前版本:", torch.__version__) # 打印 PyTorch 版本
print("CUDA是否可用:", torch.cuda.is_available()) # 检查 CUDA 是否可用
print("CUDA版本:", torch.version.cuda) # 打印 CUDA 版本
print("CUDA安装路径:", torch.utils.cpp_extension.CUDA_HOME) # 打印 CUDA 安装路径
print("可用CUDA设备数量:", torch.cuda.device_count()) # 打印可用的 CUDA 设备数量
print("第一个CUDA设备名称:", torch.cuda.get_device_name(0)) # 打印第一个 CUDA 设备的名称
print("当前CUDA设备索引:", torch.cuda.current_device()) # 打印当前 CUDA 设备的索引
print("CUDA是否已初始化:", torch.cuda.is_initialized()) # 检查 CUDA 是否已初始化
# print(torch.cuda.memory_allocated()) # 打印当前分配的 CUDA 内存
# print(torch.cuda.memory_reserved()) # 打印当前保留的 CUDA 内存
# print(torch.cuda.memory_stats()) # 打印 CUDA 内存统计信息
# print(torch.cuda.memory_summary()) # 打印 CUDA 内存摘要
# print(torch.cuda.empty_cache()) # 清空 CUDA 缓存
print("当前设备:", torch.cuda.current_device()) # 打印当前设备索引