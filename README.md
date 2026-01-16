# MotionInfer-Cpp: High-Performance C++ Inference Engine for Generative Motion Models

![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## 📖 项目简介 (Introduction)

**MotionInfer-Cpp** 是一个轻量级、无依赖的深度学习推理引擎，专为**动作生成模型 (VAE / Diffusion Models)** 设计。

本项目旨在验证**边缘设备上的实时生成能力**。与依赖庞大运行时（如 PyTorch/TensorFlow）的方案不同，本项目从零构建了底层的计算图与内存管理系统。通过手动实现核心算子（Matrix Multiplication, Convolution, Attention）并结合 **SIMD (AVX2/NEON)** 指令集优化，实现了比 Python 原生推理快 **X倍** 的性能（待最终测试填入）。

## 🚀 核心特性 (Key Features)

* **Zero-Dependency**: 不依赖 LibTorch、ONNX Runtime 等第三方重型库，纯 C++17 实现。
* **Performance Optimization**: 
    * 针对 CPU 缓存优化的内存布局 (Row-Major)。
    * 手动实现的 SIMD 加速算子 (支持 Intel AVX2 及 ARM NEON)。
    * 基于 OpenMP 的多线程并行计算。
* **Memory Efficiency**: 实现简单的内存池 (Arena Allocator)，最小化扩散模型迭代过程中的内存分配开销。
* **Model Support**: 支持从 PyTorch 导出的权重文件 (`.bin`)，专注于 MLP、CNN 及 Transformer 架构的推理。

## 📂 项目结构 (File Structure)

```text
MotionInfer-Cpp/
├── CMakeLists.txt          # 构建脚本
├── data/                   # 模型权重文件存放区
├── scripts/                # Python 辅助脚本 (权重导出、结果验证)
├── include/                # C++ 头文件 (接口定义)
│   ├── tensor.hpp          # N维数组容器与内存管理
│   ├── ops.hpp             # 深度学习核心算子声明
│   ├── model.hpp           # 网络结构组装
│   └── utils.hpp           # 工具函数 (日志、计时、文件读取)
├── src/                    # C++ 源文件 (具体实现)
│   ├── tensor.cpp
│   ├── ops.cpp             # 包含 SIMD 优化的算子实现
│   ├── model.cpp
│   └── main.cpp            # 推理入口
└── tests/                  # 单元测试

***

### 3. 详细开发清单：每个文件都要写什么？

接下来，我为你拆解每个文件具体要填入的代码逻辑。你可以把这个当作你的**开发任务清单**。

#### A. 构建系统
* **`CMakeLists.txt`**
    * **要做什么**：告诉编译器怎么编译你的代码。
    * **核心内容**：设置 C++ 标准为 17；设置包含目录 (`include/`)；把 `src/` 下的所有 `.cpp` 文件编译成一个可执行文件 `MotionInfer-Cpp`；开启编译器优化选项 (`-O3`, `-march=native` 以启用 AVX/NEON)。

#### B. 数据与内存 (The Foundation)
* **`include/tensor.hpp` / `src/tensor.cpp`**
    * **要做什么**：造一个 C++ 版的 `torch.Tensor`。
    * **核心内容**：
        * 一个 `std::vector<float>` 存储实际数据。
        * 一个 `std::vector<int>` 存储形状 (`shape`)，比如 `[1, 128]`。
        * **关键方法**：`ones()`, `zeros()`, `randn()` (初始化)；`data()` (获取指针用于计算)。
        * **为什么需要它**：因为 C++ 原生数组没有形状概念，我们需要封装它以便传递给算子。

#### C. 算子库 (The Engine)
* **`include/ops.hpp` / `src/ops.cpp`**
    * **要做什么**：这是项目的**技术含量担当**。
    * **核心内容**：
        * `matmul(A, B, C)`: 矩阵乘法 $C = A \times B$。这是 AI 模型 90% 的计算量所在。
        * `relu(A)`, `sigmoid(A)`: 激活函数。
        * `add(A, B)`: 对应元素相加 (Element-wise add)。
    * **进阶内容 (后面做)**：在这里写 SIMD 指令（Intrinsic functions）来加速上述循环。

#### D. 模型定义 (The Architecture)
* **`include/model.hpp` / `src/model.cpp`**
    * **要做什么**：把算子像搭积木一样搭起来。
    * **核心内容**：
        * 定义一个 `SimpleModel` 类。
        * 成员变量：`w1`, `b1`, `w2`, `b2` (类型都是 `Tensor`)。
        * `load(path)` 函数：打开 `.bin` 文件，依次读取字节流填充到 `w1`, `b1`... 中。
        * `forward(input)` 函数：调用 `ops::matmul(input, w1)` -> `ops::relu(...)` -> 返回结果。

#### E. 辅助脚本
* **`scripts/export_weights.py`**
    * **要做什么**：如前所述，把 PyTorch 模型的参数“压扁”并按顺序写入文件。
* **`scripts/verify_output.py`** (稍后创建)
    * **要做什么**：读取 C++ 程序的输出结果，和 PyTorch 的输出结果做减法。如果差值小于 `1e-5`，说明你的 C++ 引擎写对了。

---

### 下一步行动

Jixin，现在请你完成以下操作：
1.  在项目根目录下创建好上述的 **文件夹结构**。
2.  创建并粘贴 **README.md**。
3.  创建 **CMakeLists.txt**（如果你不会写，告诉我，我给你一个可以直接用的模版）。

准备好后，告诉我“目录已就绪”，我们就开始写最核心的 **`Tensor` 类**。