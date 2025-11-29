import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

# 1. 加载模型和处理器
# 注意：openvla-7b 需要较大显存 (约 16GB+ for bfloat16)
# 如果显存不足，可以尝试 8-bit 量化 (需安装 bitsandbytes)
print("Loading model...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "openvla/openvla-7b", 
#     attn_implementation="flash_attention_2",  # 如果已安装 flash_attn，推荐开启
#     torch_dtype=torch.bfloat16, 
#     low_cpu_mem_usage=True, 
#     trust_remote_code=True
# ).to("cuda:0")

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    # attn_implementation="flash_attention_2",  # 如果已安装 flash_attn，推荐开启
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# 2. 准备输入
# 读取真实图片 (确保 'coke_can_real.jpg' 在当前目录下)
print("Loading image...")
image = Image.open("coke_can_real.jpg").convert("RGB")
instruction = "pick up the coke can"

# 构建 Prompt
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# 3. 预测动作
print(f"Predicting action for: '{instruction}'")
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

# 预测 7 维动作 (x, y, z, roll, pitch, yaw, gripper)
# unnorm_key="bridge_orig" 表示使用 BridgeData V2 的统计数据反归一化
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Predicted Action: {action}")
# 结果是一个 shape 为 (7,) 的 numpy 数组
