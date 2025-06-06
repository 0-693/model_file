# self-VLA 项目

## 项目结构

self-VLA 是在pi0基础上修改实现的简化（可能吧）版本，能够完成paligemma+action expert的训练，不加载pi0预训练模型
但是还是有很多问题，比如训练起来还是会有不知名报错，本人对jax实在是不太懂qwq。训练后的保存权重不兼容现有pi0模型（不是pth模板）。
😔，代码能力还是不行啊

### models/

核心模型实现目录，包含以下关键文件：

- **gemma.py**: 基于big_vision的Gemma语言模型适配实现，采用多头注意力机制，支持高效的序列处理。主要特点包括：
  - 支持可配置的模型宽度、深度和注意力头数
  - 实现了RoPE位置编码
  - 集成了LoRA低秩适应能力
  - 支持KV缓存以提高推理效率

- **siglip.py**: 视觉模型实现，基于ViT（Vision Transformer）架构，主要功能：
  - 实现了2D位置编码（sincos_2d）
  - 支持可学习和固定位置编码
  - 优化的图像特征提取能力

- **pi0.py**: 多模态模型实现，整合了视觉和语言模型的能力：
  - 实现了注意力掩码机制
  - 支持不同类型的注意力模式（因果注意力、前缀-LM注意力等）
  - 集成了位置编码的实现

- **lora.py**: 低秩适应（LoRA）实现，用于高效模型微调：
  - 支持可配置的LoRA秩和缩放因子
  - 实现了rank-stabilized LoRA优化
  - 提供了与Einsum兼容的接口

### shared/

共享工具类目录，包含以下实用工具：

- **array_typing.py**: 类型检查和数组类型定义工具：
  - 提供了JAX数组的类型注解支持
  - 实现了运行时类型检查装饰器
  - 优化了数据类型检查的性能

- **image_tools.py**: 图像处理工具集：
  - 实现了保持宽高比的图像缩放功能
  - 支持填充和裁剪操作
  - 处理uint8和float32格式图像

- **nnx_utils.py**: 神经网络工具类：
  - 提供了模块JIT编译优化
  - 实现了模块状态管理
  - 支持路径正则匹配功能
