# PyTorch 深度学习快速入门（小土堆教程配套学习仓库）

- **B站视频**：[PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】](https://www.bilibili.com/video/BV1hE411t7RN/)
- **参考源码**：[xiaotudui/pytorch-tutorial](https://github.com/xiaotudui/pytorch-tutorial)

---

## 📁 目录结构说明

为了养成良好的工程规范，我们将**代码（按章节模块）**与**数据、日志、模型权重**分离开来：

```text
pytorch-learning/
├── 01_dataset/                 # P5-P6: Dataset 类代码实战（蚂蚁蜜蜂分类数据集读取）
├── 02_tensorboard/             # P7-P8: TensorBoard 的使用（SummaryWriter 画标量与图像）
├── 03_transforms/              # P9-P12: Transforms 图像变换（ToTensor, Normalize, Resize 等）
├── 04_torchvision_dataset/     # P13: torchvision 标准数据集（CIFAR-10 下载与 Transform 联动）
├── 05_dataloader/              # P14: DataLoader 数据加载器（batch_size, shuffle, drop_last）
├── 06_nn_module/               # P15-P21: nn.Module 神经网络骨架、卷积、池化、非线性激活、全连接与 Sequential
├── 07_loss_and_optim/          # P22-P23: 损失函数（Loss Function）、反向传播（backward）与优化器（Optimizer）
├── 08_model_save_load/         # P24-P25: 现有网络模型修改（VGG16）与模型的保存/加载（torch.save / torch.load）
├── 09_full_training_cifar10/   # P26-P31: 完整模型训练套路、测试评估、GPU 加速训练（.to(device)）与实战验证
├── dataset/                    # 存放数据集文件（如 hymenoptera_data、CIFAR-10 等，通常不提交到 Git）
├── logs/                       # 存放 TensorBoard 生成的事件日志
└── models/                     # 存放训练保存的 .pth 模型权重文件
```

---

## 🗺️ 学习路线图与核心知识点对照表

### 第一阶段：环境搭建与工具认知（P1 - P4）
- [x] **P1**：PyTorch 环境的配置及安装（Conda 虚拟环境与 PyTorch 安装理解）
- [x] **P2**：Python 编辑器的选择、安装及配置
- [x] **P3**：Python 学习中的两大法宝函数（`dir()` 与 `help()`）
- [x] **P4**：PyCharm 及 Jupyter 使用及对比（理解 `.py` 脚本与交互式块的区别）
- **💡 转码核心基本功**：理解什么是环境变量（`PATH`）、什么是虚拟环境（为什么不同项目需要隔离依赖）、如何看懂终端报错栈（Traceback）。

### 第二阶段：数据加载与预处理三剑客（P5 - P14）
- [x] **P5 - P6（`01_dataset/`）**：《PyTorch加载数据初认识》与《Dataset类代码实战》
  - 掌握面向对象继承：`class MyData(Dataset)`
  - 掌握两个魔术方法：`__getitem__(self, idx)` 与 `__len__(self)`
  - 掌握 Python 文件路径操作：`os.path.join`、`os.listdir`
- [ ] **P7 - P8（`02_tensorboard/`）**：《TensorBoard的使用（一）/（二）》
  - `SummaryWriter` 的 `add_scalar()`（画函数曲线/Loss曲线）与 `add_image()`（看图像输入）
  - 终端启动命令：`tensorboard --logdir=logs`
- [ ] **P9 - P12（`03_transforms/`）**：《Transforms的使用》与《常见的Transforms》
  - 理解 `PIL.Image`、`numpy.ndarray`、`torch.Tensor` 三种图像数据格式的区别与维度顺序（`HWC` vs `CHW`）
  - 掌握 `__call__` 魔术方法（为什么对象可以像函数一样被调用 `trans(img)`）
  - 常用变换：`ToTensor`、`Normalize`、`Resize`、`Compose`、`RandomCrop`
- [ ] **P13 - P14（`04_torchvision_dataset/` & `05_dataloader/`）**：《torchvision中的数据集使用》与《DataLoader的使用》
  - `Dataset` 负责“怎么取单个样本”，`DataLoader` 负责“怎么把样本打包成 Batch 送进模型”

### 第三阶段：神经网络核心积木 `torch.nn`（P15 - P21）
- [ ] **P15（`06_nn_module/`）**：《神经网络的基本骨架 - nn.Module的使用》（`__init__` 与 `forward`）
- [ ] **P16 - P17**：《土堆说卷积操作》（`F.conv2d`）与《神经网络 - 卷积层》（`nn.Conv2d`）
- [ ] **P18**：《神经网络 - 最大池化的使用》（`nn.MaxPool2d`）与下采样作用
- [ ] **P19**：《神经网络 - 非线性激活》（`ReLU`、`Sigmoid`）——为什么没有激活函数多层网络就等价于单层线性变换？
- [ ] **P20 - P21**：《线性层及其他层介绍》（`nn.Linear`、`nn.Flatten`）与《搭建小实战和Sequential的使用》

### 第四阶段：训练闭环、模型存取与 GPU 实战（P22 - P31）
- [ ] **P22 - P23（`07_loss_and_optim/`）**：《损失函数与反向传播》与《优化器》（`zero_grad()`、`backward()`、`step()`）
- [ ] **P24 - P25（`08_model_save_load/`）**：《现有网络模型的使用及修改》与《网络模型的保存与读取》（完整模型 vs `state_dict`）
- [ ] **P26 - P31（`09_full_training_cifar10/`）**：《完整的模型训练套路（一/二/三）》、《利用GPU训练（一/二）》与《完整的模型验证套路》
  - 完整训练/验证循环编写
  - `model.train()` 与 `model.eval()`、`with torch.no_grad():` 的作用
  - 分类准确率计算：`argmax(1)`
  - GPU 训练的两种写法（`.cuda()` vs 更现代通用的 `.to(device)`）
  - 拿一张网上的真实图片输入自己训练好的模型进行推理验证！

---

## ⚠️ 视频教程（旧版 PyTorch）与现代 PyTorch 2.x 常见差异提示

小土堆视频录制于 PyTorch 1.8 左右，使用新版 PyTorch 学习时注意以下几个常见变化：
1. **预训练模型参数（P24）**：视频中的 `torchvision.models.vgg16(pretrained=True)` 在新版中会报警告，现代标准写法为 `vgg16(weights=VGG16_Weights.DEFAULT)`。
2. **加载自定义模型安全限制（P25）**：在 PyTorch 2.4+ / 2.6+ 中，`torch.load()` 默认开启了 `weights_only=True` 安全限制，如果加载整个模型类对象需要显式传入 `torch.load("xxx.pth", weights_only=False)`，或者使用更规范的 `state_dict` 方式。
3. **GPU 写法（P29-P30）**：相比于在模型和数据上零散地加 `if torch.cuda.is_available(): xxx = xxx.cuda()`，工程上更推荐统一定义 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 并使用 `.to(device)`。
