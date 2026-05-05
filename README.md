# LDM 文本生成图像 — COCO 2017

基于 **潜扩散模型 (LDM)** 的文本条件图像生成，在 MS COCO 2017 数据集上训练。

## 架构

```
VAE (runwayml/stable-diffusion-v1-5)
CLIP ViT-L/14 (openai/clip-vit-large-patch14)
条件 U-Net (UNet2DConditionModel + DDPMScheduler)
```

| 项目 | 参数 |
|------|------|
| 输入分辨率 | 256×256 RGB |
| Latent 空间 | 32×32×4 |
| 文本编码维度 | 77×768 |
| U-Net 通道 | (128, 256, 512, 512) |
| 训练步数 | 1000 (DDPM) |
| 推理步数 | 50 (DDIM) |

## 目录结构

```
coco4.ipynb              # 主 Notebook（预处理 + 训练 + 推理）
coco2017/
  images/train2017/      # COCO 训练图片
  images/val2017/        # COCO 验证图片
  annotations/           # Caption 标注 JSON
preproc_ldm/
  train/latents/         # VAE 编码结果 (.npy)
  train/texts/           # CLIP 编码结果 (.npy)
  val/latents/
  val/texts/
checkpoints_ldm/         # U-Net 权重 (unet_epochXXXX.pth)
training_logs_ldm.csv    # 训练损失日志
```

## 阶段流程

### 阶段一：数据预处理
将 COCO 图片经 VAE 编码为 Latent，Caption 经 CLIP 编码为 Text Embeddings，保存为 `.npy` 文件。训练时直接读取，无需重复加载 VAE/CLIP。

> 空间优化：`.npy` 格式比 `.pt` 节省约 93% 磁盘空间（~14 GB ← ~218 GB）

### 阶段二：U-Net 训练

- **混合精度**：FP16 autocast + GradScaler
- **梯度累加**：等效 Batch Size $= 25 \times 4 = 100$
- **CFG 训练**：10% 概率随机丢弃文本条件
- **预取优化**：后台线程预取，GPU 利用率最大化
- 每 5 个 epoch 保存一次权重

DDPM 前向加噪（将干净 Latent $x_0$ 在时间步 $t$ 处混入噪声）：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

训练目标（MSE 噪声预测损失）：

$$
\mathscr{L} = \mathbb{E}\left[\left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2\right]
$$

### 阶段三：CFG 推理

在每个去噪步骤中，用 CFG 引导强度 $w$ 加权组合有条件与无条件预测：

$$
\hat{\epsilon} = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})
$$

默认 `guidance_scale=7.5`，DDIM 50 步生成 $256 \times 256$ 图像。

## 主要超参数

| 超参数 | 值 |
|--------|----|
| Batch Size | $100$ |
| 学习率 | $1\times 10^{-4}$ |
| Epochs | $50$ |
| CFG Drop Prob | $0.1$ |
| Guidance Scale | $7.5$ |

## 依赖

```
torch >= 2.0
diffusers
transformers
numpy
Pillow
tqdm
matplotlib
pandas
```

## 运行环境

| 项目 | 配置 |
|------|------|
| OS | Windows 11 Pro 64-bit |
| CPU | Intel Core i9-10900K @ 3.70 GHz (10 cores / 20 threads) |
| Memory | 64 GB RAM |
| GPU | NVIDIA Quadro RTX 4000 (8 GB GDDR6) |
| Python | 3.10.19 |
| CUDA | 12.1 |
