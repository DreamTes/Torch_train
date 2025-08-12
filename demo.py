import torch
print("当前版本:", torch.__version__) # 打印 PyTorch 版本
print("CUDA是否可用:", torch.cuda.is_available()) # 检查 CUDA 是否可用
print("CUDA版本:", torch.version.cuda) # 打印 CUDA 版本
print("可用CUDA设备数量:", torch.cuda.device_count()) # 打印可用的 CUDA 设备数量
print("第一个CUDA设备名称:", torch.cuda.get_device_name(0)) # 打印第一个 CUDA 设备的名称

# 检查当前使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备是: {device}")

print(torch.rand(3,3).to(device))  # 在当前设备上创建一个随机张量,放入GPU