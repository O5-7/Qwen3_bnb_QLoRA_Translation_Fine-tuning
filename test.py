import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

model_name = "../dl_models/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
# model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(device)

text = "我是测试内容。" + tokenizer.eos_token
print(tokenizer(text, return_tensors="pt"))