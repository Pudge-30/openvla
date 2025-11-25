# OpenVLA: 一个开源的视觉-语言-动作模型

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**快速开始**](#快速开始) | [**预训练 VLA**](#预训练-vla) | [**安装**](#安装) | [**通过 LoRA 微调 OpenVLA**](#通过-lora-微调-openvla) | [**全量微调 OpenVLA**](#全量微调-openvla) |
[**从头训练 VLA**](#从头训练-vla) | [**评估 OpenVLA**](#评估-openvla) | [**项目网站**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

## 最新更新
- [2025-03-03] OFT (VLA 的优化微调配方) 最近发布了！与普通的 OpenVLA 微调相比，OFT 实现了 25-50 倍的推理速度提升，更高的任务成功率，支持多输入图像以及高频双臂机器人控制。与 FAST 不同，OFT 使用连续动作以获得更高的模型质量。请参阅项目网站 [此处](https://openvla-oft.github.io/)。
- [2025-01-16] FAST 动作分词器 (Action Tokenizer) 最近发布了！与普通的 OpenVLA 风格的 256-bin 动作离散化相比，FAST 允许将动作块压缩成更少的 token，在使用离散机器人动作时将推理速度提高多达 15 倍。请参阅项目网站 [此处](https://www.physicalintelligence.company/research/fast)。
- [2024-10-15] 在 README 中添加了 [VLA 性能故障排除](#vla-性能故障排除) 部分，提供了微调后 VLA 性能不佳时的调试最佳实践。
- [2024-09-04] 在论文中添加了 LIBERO 仿真基准测试的微调实验（见 [arXiv](https://arxiv.org/abs/2406.09246) 上的 v2 版本）；
  并在 [LIBERO 仿真基准测试评估](#libero-仿真基准测试评估) 部分添加了复现 OpenVLA 结果的说明。
- [2024-08-14] 添加了新章节 [评估 OpenVLA](#评估-openvla)，包含运行 BridgeData V2 WidowX 机器人评估的说明。
- [2024-07-08] 添加了新章节：[通过 LoRA 微调 OpenVLA](#通过-lora-微调-openvla)，[全量微调 OpenVLA](#全量微调-openvla)。
- [2024-06-13] 首次发布。

<hr style="border: 2px solid gray;"></hr>

这是一个简单且可扩展的代码库，用于训练和微调通用的机器人操作视觉-语言-动作模型 (VLA)：

- **不同的数据集混合**：我们原生支持 RLDS 格式的任意数据集，包括来自 [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) 的任意数据混合。
- **易于扩展**：基于 PyTorch FSDP 和 Flash-Attention，我们可以快速高效地训练 1B - 34B 参数的模型，并具有易于调整的模型架构。
- **原生微调支持**：内置支持（附带示例）各种形式的微调（全量、部分、LoRA）。

构建在 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 之上。

## 快速开始

要开始加载和运行 OpenVLA 模型进行推理，我们提供了一个轻量级接口，利用 HuggingFace `transformers` AutoClasses，依赖项极少。

例如，要在 WidowX 机器人上的 [BridgeData V2 环境](https://rail-berkeley.github.io/bridgedata/) 中加载 `openvla-7b` 进行零样本指令跟随：

```python
# 安装最小依赖 (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# 加载处理器和 VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [可选] 需要 `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# 获取图像输入并格式化提示词
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# 预测动作 (7-DoF; 针对 BridgeData V2 进行反归一化)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 执行...
robot.act(action, ...)
```

我们还提供了一个[针对新任务和具身微调 OpenVLA 模型的示例脚本](./vla-scripts/finetune.py)；该脚本支持不同的微调模式——包括 [HuggingFace PEFT 库](https://huggingface.co/docs/peft/en/index) 支持的（量化）低秩适应 (LoRA)。

对于部署，我们提供了一个轻量级脚本，用于[通过 REST API 提供 OpenVLA 模型服务](./vla-scripts/deploy.py)，提供了一种简单的方法将 OpenVLA 模型集成到现有的机器人控制栈中，无需强大的端侧计算能力。

## 预训练 VLA

我们发布了作为我们工作一部分训练的两个 OpenVLA 模型，相关的检查点、配置和模型卡可在 [我们的 HuggingFace 页面](https://huggingface.co/openvla) 上找到：
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b)：我们论文中的旗舰模型，基于 Prismatic `prism-dinosiglip-224px` VLM（基于融合的 DINOv2 和 SigLIP 视觉骨干，以及 Llama-2 LLM）训练。在 Open X-Embodiment 的大型数据集混合上进行训练，涵盖 970K 条轨迹（[混合详情 - 参见 "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)）。
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01)：开发过程中使用的早期模型，基于 Prismatic `siglip-224px` VLM（单一 SigLIP 视觉骨干和 Vicuña v1.5 LLM）训练。在与 [Octo](https://github.com/octo-models/octo) 相同的数据集混合上训练，但 GPU 小时数远少于我们的最终模型（[混合详情 - 参见 "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)）。

**关于模型许可和商业使用的明确说明**：虽然此仓库中的所有代码均在 MIT 许可下发布，但我们的预训练模型可能会继承我们使用的底层基础模型的限制。具体而言，上述两个模型均派生自 Llama-2，因此受 [Llama 社区许可](https://ai.meta.com/llama/license/) 的约束。

---

## 安装

> **注意**：这些安装说明适用于全规模预训练（和分布式微调）；如果只是想运行 OpenVLA 模型进行推理（或执行轻量级微调），请参阅上面的说明！

此仓库是使用 Python 3.10 构建的，但也应向后兼容任何 Python >= 3.8。我们需要 PyTorch 2.2.* —— 安装说明[可以在这里找到](https://pytorch.org/get-started/locally/)。此仓库的最新版本已在以下环境中开发并经过全面测试：
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, 和 flash-attn 2.5.5

**[5/21/24] 注意**：由于 `transformers`、`timm` 和 `tokenizers` 的后续版本报告了回归和破坏性更改，我们明确固定了上述依赖项的版本。我们正在努力实施全面的测试，并计划尽快放宽这些限制。

使用以下设置命令开始：

```bash
# 创建并激活 conda 环境
conda create -n openvla python=3.10 -y
conda activate openvla

# 安装 PyTorch。下面是一个示例命令，但您应该查看以下链接
# 以找到特定于您的计算平台的安装说明：
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # 请根据情况更新！

# 克隆并安装 openvla 仓库
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 安装 Flash Attention 2 用于训练 (https://github.com/Dao-AILab/flash-attention)
#   =>> 如果遇到困难，请先尝试 `pip cache remove flash_attn`
pip install packaging ninja
ninja --version; echo $?  # 验证 Ninja --> 应该返回退出代码 "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

如果在安装过程中遇到任何问题，请提交 GitHub Issue。

**注意：** 有关 OpenVLA 模型的完整训练和验证脚本，请参阅 `vla-scripts/`。请注意，`scripts/` 主要是原始（基础）`prismatic-vlms` 仓库的遗留物，支持训练和评估视觉条件语言模型；虽然您可以使用此仓库训练 VLM 和 VLA，但请注意，尝试使用现有的 OpenVLA 模型生成语言（通过 `scripts/generate.py`）将不起作用（因为我们仅训练当前的 OpenVLA 模型生成动作，且仅生成动作）。

## 通过 LoRA 微调 OpenVLA

**(2025-03-03 更新：我们建议尝试新的 OFT 配方来微调 OpenVLA，以产生更快更成功的策略。请参阅项目网站 [此处](https://openvla-oft.github.io/)。)**

在本节中，我们将讨论通过 Hugging Face `transformers` 库使用低秩适应 (LoRA) 微调 OpenVLA，如果您没有足够的计算资源来全量微调 7B 参数的模型，建议使用此方法。主要的 LoRA 微调脚本是 `vla-scripts/finetune.py`。（如果您希望进行全量微调，请参阅 [全量微调 OpenVLA](#全量微调-openvla) 部分。）

下面我们展示如何通过 LoRA 微调主 OpenVLA 检查点 ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b)) 的示例。在这里，我们使用单个具有 80 GB 显存的 A100 GPU 在 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 上进行微调。（只要显存至少有约 27 GB，您也可以通过修改批量大小使用更小的 GPU 进行微调。）

首先，下载 BridgeData V2 数据集：

```bash
# 切换到您的基础数据集目录
cd <基础数据集目录路径>

# 下载完整数据集 (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 将数据集重命名为 `bridge_orig` (注意：省略此步骤可能会导致后续运行时错误)
mv bridge_dataset bridge_orig
```

现在，启动 LoRA 微调脚本，如下所示。请注意，`--batch_size==16` 和 `--grad_accumulation_steps==1` 需要约 72 GB GPU 内存。如果您有更小的 GPU，应减小 `--batch_size` 并增加 `--grad_accumulation_steps` 以保持足够大的有效批量大小以进行稳定训练。如果您有多个 GPU 并希望通过 PyTorch 分布式数据并行 (DDP) 进行训练，只需在下面的 `torchrun` 命令中将 `--nproc-per-node` 设置为可用 GPU 的数量。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <基础数据集目录路径> \
  --dataset_name bridge_orig \
  --run_root_dir <日志/检查点目录路径> \
  --adapter_tmp_dir <保存适配器权重的临时目录路径> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True 或 False> \
  --wandb_project <项目名> \
  --wandb_entity <实体名> \
  --save_steps <每个检查点保存的梯度步数>
```

注意：如果您在上述命令中设置 `--image_aug==False`，您将在训练日志中观察到接近 100% 的 `action_accuracy`，因为 [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) 模型已经在包含 BridgeData V2 的数据集超集上进行了预训练（无增强）。

要在不同的数据集上进行 LoRA 微调，您可以从 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) 混合中下载数据集（参见 [此自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) 了解如何从 OXE 下载数据集的示例）。或者，如果您有不属于 OXE 的自定义数据集，您可以 (a) 将数据集转换为与我们的微调脚本兼容的 RLDS 格式（有关说明，请参阅 [此仓库](https://github.com/kpertsch/rlds_dataset_builder)），或 (b) 使用您自己的自定义 PyTorch Dataset 包装器（有关说明，请参阅 `vla-scripts/finetune.py` 中的注释）。我们建议大多数用户选择选项 (a)；RLDS 数据集和数据加载器经过了更广泛的测试，因为我们将它们用于所有的预训练和微调实验。

对于选项 (a)，将数据集转换为 RLDS 后，您需要将其注册到我们的数据加载器中，方法是在 [此处](prismatic/vla/datasets/rlds/oxe/configs.py#L54) 注册数据集配置，并在 [此处](prismatic/vla/datasets/rlds/oxe/transforms.py#L828) 注册数据集转换函数。

集成新数据集后，您可以使用上面的同一个 `vla-scripts/finetune.py` 脚本启动 LoRA 微调。如果遇到任何问题，请访问 [VLA 故障排除](#vla-性能故障排除) 部分或在 [OpenVLA GitHub Issues 页面](https://github.com/openvla/openvla/issues?q=)（包括“已关闭”的问题）中搜索类似问题。如果在那里找不到类似问题，请随时创建新 issue。

## 全量微调 OpenVLA

**(2025-03-03 更新：我们建议尝试新的 OFT 配方来微调 OpenVLA，以产生更快更成功的策略。请参阅项目网站 [此处](https://openvla-oft.github.io/)。)**

在本节中，我们讨论使用 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 训练脚本通过原生 PyTorch 完全分片数据并行 (FSDP) <ins>全量微调</ins> OpenVLA（所有 75 亿参数）。全量微调更高级/复杂，仅在您有足够的计算资源（例如，一个包含 8 个 A100 GPU 的完整节点）且 LoRA 微调不足以满足您的用例（例如，如果微调分布与预训练分布差异巨大）时才建议使用。否则，我们建议您尝试通过 LoRA 进行参数高效微调，如 [通过 LoRA 微调 OpenVLA](#通过-lora-微调-openvla) 部分所述。

对于全量微调，您需要下载[不同版本的 OpenVLA 模型检查点](https://huggingface.co/openvla/openvla-7b-prismatic)，该版本与 Prismatic VLMs 代码库兼容，我们在其之上开发了 OpenVLA 模型。您可以使用下面的 git 命令下载此兼容 Prismatic 的 OpenVLA 检查点（或者，您可以通过 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 下载）：

```bash
# 切换到您的基础模型检查点文件夹
cd <基础模型检查点目录路径>

# 下载检查点 (30 GB) -- 可能需要几分钟
git clone git@hf.co:openvla/openvla-7b-prismatic

# 如果上述命令没有下载完整的检查点，
# 请通过 git Large File Storage (LFS) 手动获取
# 注意：您可能需要配置 SSH 密钥才能使其工作
cd openvla-7b-prismatic
git lfs fetch --all
```

我们展示了如何使用具有 8 个 GPU 的单个节点在 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 上全量微调 OpenVLA。如果您希望使用不同数量的 GPU（或节点），可以在 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中修改 VLA 训练配置。

下载 BridgeData V2 数据集：

```bash
# 切换到您的基础数据集文件夹
cd <基础数据集目录路径>

# 下载完整数据集 (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 将数据集重命名为 `bridge_orig` (注意：省略此步骤可能会导致后续运行时错误)
mv bridge_dataset bridge_orig
```

接下来，创建一个 [Hugging Face 用户访问令牌](https://huggingface.co/docs/hub/en/security-tokens) 并将令牌值（以 `hf_...` 开头的字符串）复制到此仓库根目录下名为 `.hf_token` 的文件中 (`openvla/.hf_token`)。

```bash
# 进入 openvla 根目录
cd openvla

# 将 HF 令牌值复制到令牌文件中。将 "hf_..." 替换为您自己的令牌值！
# 参见：https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```

现在，启动训练脚本。如果您希望使用不同数量的节点或 GPU，请修改 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中的 VLA 训练配置，然后相应地更改下面的 `--nnodes` 和 `--nproc-per-node` 参数。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <openvla/openvla-7b-prismatic 检查点文件路径: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <基础数据集目录路径> \
  --run_root_dir <日志/检查点目录路径> \
  --run_id <WANDB 日志的可选运行 ID> \
  --image_aug <True 或 False> \
  --wandb_project <项目名> \
  --wandb_entity <实体名> \
  --save_interval <每个检查点保存的梯度步数> \
  --is_resume False
```

请注意，上面的 `--is_resume` 参数设置为 `False`，因为我们是在微调预训练的检查点，而不是恢复暂停的训练运行。

如果您的训练运行暂停并且您希望从最新的检查点恢复，请将 `--pretrained_checkpoint` 更改为最新的检查点路径，然后设置 `--is_resume==True` 并分别指定 `--resume_step` 和 `--resume_epoch` 为步数和轮数。例如，如果您希望从名为 `step-010000-epoch-20-loss=0.0160.pt` 的检查点恢复训练，则应设置 `is_resume==True`，`resume_step==10000`，和 `resume_epoch==20`。

注意：如果您运行上面的 BridgeData V2 微调命令，您应该在训练日志中观察到接近 100% 的动作 Token 准确率，因为 [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) 模型已经在包含 BridgeData V2 的数据集超集上进行了预训练。

要在不同的数据集上全量微调 OpenVLA，您可以从 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) 混合中下载数据集（参见 [此自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) 了解如何从 OXE 下载数据集的示例）。或者，如果您有不属于 OXE 的自定义数据集，您可以将数据集转换为 RLDS 格式，该格式与我们的微调脚本兼容（有关说明，请参阅 [此仓库](https://github.com/kpertsch/rlds_dataset_builder)）。下载/转换数据集后，您需要修改以下文件：

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py)：通过创建一个实验类添加新的训练配置，然后在文件底部的 `VLARegistry` 中注册它。
  * 确保为您的微调运行创建一个新的唯一 `vla_id`，并根据需要调整一些配置变量——例如，`expected_world_size`（GPU 数量），`per_device_batch_size`（每个 GPU 的批量大小），`global_batch_size`（总批量大小），`shuffle_buffer_size`（每个 GPU 的洗牌缓冲区样本数）等。请参阅文件顶部 `VLAConfig` 类下的注释以了解每个变量的用途。
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py)：在 `OXE_NAMED_MIXTURES` 字典中为您的微调混合定义一个新的混合。
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py)：为您的微调数据集定义一个新的数据集转换函数，并将其添加到文件底部的 `OXE_STANDARDIZATION_TRANSFORMS` 注册表中。
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py)：将指定微调数据集观察和动作空间的新配置添加到 `OXE_DATASET_CONFIGS` 字典中。

完成上述步骤后，您可以使用 `vla-scripts/train.py` 脚本开始全量微调。确将 `--vla.type` 参数设置为您在 `prismatic/conf/vla.py` 中添加的新 `vla_id`。

微调完成后，您需要将最终模型检查点转换为与 Hugging Face `transformers` 库兼容的版本。有关说明，请参阅 [将 Prismatic 模型转换为 Hugging Face](#将-prismatic-模型转换为-hugging-face) 部分。

如果遇到任何问题，请访问 [VLA 故障排除](#vla-性能故障排除) 部分或在 [OpenVLA GitHub Issues 页面](https://github.com/openvla/openvla/issues?q=)（包括“已关闭”的问题）中搜索类似问题。如果在那里找不到类似问题，请随时创建新 issue。

### 将 Prismatic 模型转换为 Hugging Face

如果您使用 Prismatic VLMs 代码库训练了您的模型（例如，如果您在通过 Prismatic 对 OpenVLA 进行了全量微调），您需要将最终检查点转换为与 Hugging Face `transformers` AutoClasses 兼容的版本。我们在本节中讨论如何执行此操作。

假设您的训练运行目录是 `PRISMATIC_RUN_DIR`（例如，`prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`）。在此目录中，应该有一个名为 `checkpoints` 的目录，其中包含保存的模型检查点（例如，`step-295000-epoch-40-loss=0.2200.pt`）。Prismatic 到 Hugging Face 的转换脚本（[convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)）期望一个名为 `latest-checkpoint.pt` 的检查点文件。因此，您应该首先创建一个名为 `latest-checkpoint.pt` 的符号链接，指向您希望转换的检查点文件：

```bash
# 进入您的 Prismatic 训练运行的 `checkpoints` 目录
cd PRISMATIC_RUN_DIR/checkpoints

# 创建指向您的检查点文件的符号链接
ln -s <您的检查点文件名> latest-checkpoint.pt
```

然后，启动转换脚本将检查点从 Prismatic VLMs 格式转换为 Hugging Face 格式：

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <转换后检查点的输出目录>
```

上面的命令会将兼容 HF 的检查点保存在 `output_hf_model_local_path` 中。现在您可以像往常一样使用 HF AutoClasses 加载检查点，如下所示。请注意，在加载 OpenVLA 模型之前，需要将其注册到 HF AutoClasses，因为您加载的是本地保存的检查点，而不是推送到 HF Hub 的检查点（有关详细信息，请参阅 [此处](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)）。

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# 将 OpenVLA 模型注册到 HF AutoClasses（如果已将模型推送到 HF Hub 则不需要）
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# 加载处理器和 VLA
processor = AutoProcessor.from_pretrained("<转换后检查点目录的路径>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<转换后检查点目录的路径>",
    attn_implementation="flash_attention_2",  # [可选] 需要 `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

## 从头训练 VLA

我们提供有关在 [Open X-Embodiment (OXE) 数据集](https://robotics-transformer-x.github.io/)（的任意子集）上训练 VLA 模型的完整说明和配置。如果在执行以下操作时遇到任何问题，请参阅下面的 [VLA 故障排除](#vla-性能故障排除)（或提交 GitHub Issue）。

### VLA 预训练数据集

我们按照[此自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh)以 [RLDS 格式](https://github.com/google-research/rlds)下载并预处理 Open X-Embodiment 中的各个数据集。请参阅 [mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) 获取我们用于训练 `openvla-7b` 的组件数据集（和混合权重）的完整列表。
- **重要**：对于 BridgeData V2 组件，OXE 中的版本已过时（截至 2023/12/20）。相反，您应该从[官方网站](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/)下载数据集，并将其放置在子目录 `bridge_orig/` 下。将 OXE 代码中对 `bridge` 的任何引用替换为 `bridge_orig`。

### VLA 配置和训练脚本

VLA 训练的入口点是 [`vla-scripts/train.py`](vla-scripts/train.py)。我们使用 [`draccus`](https://pypi.org/project/draccus) 提供模块化、基于数据类的接口来指定 VLA 训练配置；现有的 VLA 配置位于 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中。您可以添加自己的训练配置，并使用 `--vla.type` 命令行参数引用它。

我们使用 PyTorch 完全分片数据并行 (FSDP) 在 GPU 之间分发训练。通过 `torchrun` 启动训练：

```bash
# 在单个节点（8 个 GPU）上使用 Prismatic DINO-SigLIP 224px 骨干在 BridgeData V2 上训练 VLA
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <OXE 数据根目录路径> \
  --run_root_dir <日志/检查点根目录路径> \
  --wandb_project "<项目名>" \
  --wandb_entity "<实体名>"
```

### VLA 性能故障排除

以下是一系列已知问题和相应的修复方法：

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **修复**：通过 `pip install tensorflow-datasets==4.9.3` 降级 `tensorflow-datasets`。


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **修复**：将 `dlimp` 升级到最新版本。您可能需要像这样 `--force-reinstall`：
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---

## 评估 OpenVLA

### BridgeData V2 WidowX 评估

#### 设置

克隆 [BridgeData V2 WidowX 控制器仓库](https://github.com/rail-berkeley/bridge_data_robot) 并安装 `widowx_envs` 包：

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
pip install -e widowx_envs
```

此外，安装 [`edgeml`](https://github.com/youliangtan/edgeml) 库：
```bash
git clone https://github.com/youliangtan/edgeml.git
cd edgeml
pip install -e .
```

按照 `bridge_data_robot` README 中的说明创建 Bridge WidowX Docker 容器。

#### 启动 BridgeData V2 评估

有多种运行 BridgeData V2 评估的方法。我们在下面描述服务器-客户端方法。

在一个终端窗口中（例如，在 tmux 中），启动 WidowX Docker 容器：

```bash
cd bridge_data_robot
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

在第二个终端窗口中，运行 WidowX 机器人服务器：

```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

在第三个终端窗口中，运行 OpenVLA 策略评估脚本：

```bash
cd openvla
python experiments/robot/bridge/run_bridgev2_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b
```

如果遇到诸如 `ModuleNotFoundError: No module named 'moviepy.editor'` 之类的错误，可以通过在 bridge_data_robot 仓库的 requirements.txt 文件[这里](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/requirements.txt)将 moviepy 版本固定为旧版本 v1.0.3 来解决。即，只需将 requirements.txt 文件中的 `moviepy` 替换为 `moviepy==1.0.3`。然后，返回第一步重新启动 WidowX Docker 容器；它应该会使用旧的 moviepy 版本重新构建。


### LIBERO 仿真基准测试评估

在 [更新后的 OpenVLA 论文 (v2)](https://arxiv.org/abs/2406.09246) 中，我们在附录 E 中讨论了在仿真基准测试 [LIBERO](https://libero-project.github.io/main.html) 上微调 OpenVLA。
请参阅论文了解详细信息，例如我们如何修改提供的演示数据集以提高所有方法的整体性能。

我们将结果复制到下面的部分，然后讨论如何复现 OpenVLA 的结果。

#### OpenVLA 微调结果

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy 从头训练 | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo 微调 | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA 微调 (我们的) | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

每个成功率是 3 个随机种子 x 每次 500 次 rollout 的平均值（10 个任务 x 每个任务 50 次 rollout）。

#### LIBERO 设置

克隆并安装 [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

此外，安装其他所需的包：
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

（可选）要下载我们在微调实验中使用的修改版 LIBERO 数据集，请运行以下命令。这将下载 RLDS 数据格式的 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-10 数据集（总共约 10 GB）。您可以使用这些数据集微调 OpenVLA 或训练其他方法。此步骤是可选的，因为我们在下面提供了预训练的 OpenVLA 检查点。
（此外，您可以找到我们用于生成原始 HDF5 格式修改版数据集的脚本 [此处](experiments/robot/libero/regenerate_libero_dataset.py)，以及我们用于将这些数据集转换为 RLDS 格式的代码 [此处](https://github.com/moojink/rlds_dataset_builder)。）
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

#### 启动 LIBERO 评估

我们通过 LoRA (r=32) 在四个 LIBERO 任务套件上独立微调了 OpenVLA：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-10（也称为 LIBERO-Long）。
这四个检查点可在 Hugging Face 上获得：
* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

要使用这些检查点之一开始评估，请运行下面的命令之一。每个命令都会自动下载上面列出的相应检查点。

```bash
# 启动 LIBERO-Spatial 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# 启动 LIBERO-Object 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# 启动 LIBERO-Goal 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# 启动 LIBERO-10 (LIBERO-Long) 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

注意：
* 评估脚本默认将运行 500 次试验（10 个任务 x 每个 50 集）。您可以通过设置 `--num_trials_per_task` 修改每个任务的试验次数。您还可以通过 `--seed` 更改随机种子。
* **注意：设置 `--center_crop True` 很重要**，因为我们使用随机裁剪增强微调了 OpenVLA（我们在每个训练样本中采用了 90% 面积的随机裁剪，因此在测试时我们只需采用中心 90% 裁剪）。
* 评估脚本在本地记录结果。您还可以通过设置 `--use_wandb True` 并指定 `--wandb_project <项目名>` 和 `--wandb_entity <实体名>` 将结果记录在 Weights & Biases 中。
* 我们论文中报告的结果是使用 **Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, 和 flash-attn 2.5.5** 在 **NVIDIA A100 GPU** 上获得的，且为三个随机种子的平均值。请坚持使用这些包版本。请注意，如果您在评估时使用不同的 GPU，结果可能会略有不同，这是由于大模型中的 GPU 非确定性（尽管我们已测试结果在具有 A100 GPU 的不同机器上是一致的）。

如果遇到任何问题，请提交 GitHub Issue。

---

## 仓库结构

仓库/项目文件树的高级概览：

+ `prismatic` - 包源码；提供模型加载、训练、数据预处理等核心实用程序。
+ `vla-scripts/` - 用于训练、微调和部署 VLA 的核心脚本。
+ `experiments/` - 用于在机器人环境中评估 OpenVLA 策略的代码。
+ `LICENSE` - 所有代码均在 MIT 许可下提供；祝编程愉快！
+ `Makefile` - 顶层 Makefile（默认支持 linting - 检查和自动修复）；根据需要扩展。
+ `pyproject.toml` - 完整的项目配置详情（包括依赖项），以及工具配置。
+ `README.md` - 您在这里！

---


# VLA 性能故障排除

在本节中，我们涵盖了在目标域机器人数据集上微调后调试 VLA 性能不佳的最佳实践。

**注意**：OpenVLA 通常需要在来自目标域机器人的小型演示数据集（约 100 个演示）上进行微调。开箱即用时，它仅在训练数据集的域上表现良好。

**健全性检查**：
- 重放微调数据集中的演示动作，并确保机器人可以成功执行任务（这确保您的数据收集管道是正确的）
- 微调模型后，将模型加载到推理管道中（就像您运行它来控制机器人一样），但将微调数据集中的图像输入模型（假装它们来自机器人），并验证您可以复现训练时的 token 准确率 / L1 误差（这确保您的推理管道是正确的）

**微调数据收集的最佳实践**：
如果您的设置通过了上述两项健全性检查，则问题可能不在于模型训练，而在于您微调模型的数据。一些数据收集的最佳实践：
- *以 5-10Hz 左右的控制频率进行收集。* OpenVLA 未使用动作分块 (action chunking) 进行训练，经验表明该模型在高频数据上表现不佳。如果您的机器人设置使用高频控制器（例如 50 Hz），请考虑将动作下采样到 5Hz。首先验证您的机器人是否可以在使用 5Hz 动作时仍能完成任务（即使用 5Hz 动作重复上面的健全性检查 (1)）
- *在数据收集期间避免停顿/小动作。* 由于 OpenVLA 是在没有动作分块的情况下训练的，因此模型可能对微调数据中的空闲动作敏感。如果您的数据包含机器人几乎不移动的步骤，则模型可能会在推理时“卡在”这些步骤中。尝试通过连续、缓慢的移动来收集微调演示。
- *确保足够的数据覆盖率。* 如果您计划在某些变化下测试模型，例如物体的不同初始位置，请确保您的微调数据也包含此类条件的足够多样性，例如展示具有不同初始条件的演示。
- *在数据收集过程中使用一致的任务策略。* 这不是硬性约束，但可能会让您的生活更轻松。尝试以一致的方式演示任务，例如从同一侧接近物体，即使子步骤可以按任意顺序执行，也按相同的顺序执行。保持一致性可以为您提供一个模态较少的微调数据集，从而使建模问题.


---

#### 引用

如果您发现我们的代码或模型对您的工作有用，请引用[我们的论文](https://arxiv.org/abs/2406.09246)：

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```

