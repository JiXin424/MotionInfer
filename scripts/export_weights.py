import torch
import torch.nn as nn
import struct

# 1. 定义一个简单的模型 (或者加载你训练好的模型)
class SimpleMotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)  # 输入128维，输出64维
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)   # 输出10维动作参数

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 初始化模型并保存权重
model = SimpleMotionNet()
model.eval()

# 2. 导出函数：把参数写入二进制文件
def export_model(model, filename):
    with open(filename, 'wb') as f:
        # 遍历所有参数 (weights 和 bias)
        for name, param in model.named_parameters():
            print(f"Exporting {name}: shape={param.shape}")
            
            # 将 Tensor 转为一维 numpy 数组
            data = param.detach().numpy().flatten()
            
            # 写入数据长度 (方便 C++ 知道要读多少)
            # 'i' 代表 int (4字节), 'f' 代表 float (4字节)
            # 这里我们简单粗暴，直接写数据，不写 header，靠硬编码顺序读 (最简单起步)
            # 或者更规范一点：先写维度，再写数据。
            
            # 为了第一步最简单，我们直接把纯浮点数写进去
            # pack 格式: 'f' * 数量
            binary_data = struct.pack(f'{len(data)}f', *data)
            f.write(binary_data)
            
    print(f"Model saved to {filename}")

export_model(model, "../data/model.bin")