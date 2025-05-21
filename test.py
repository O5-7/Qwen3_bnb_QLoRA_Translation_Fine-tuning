import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from tqdm import tqdm

from os.path import join
import torch
# from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch.utils.data import Dataset, DataLoader, Sampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from bitsandbytes.optim import PagedAdamW8bit
from transformers import BitsAndBytesConfig

Q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "../dl_models/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)

model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=Q_config
).to(device)

prompt = "你好\n请介绍一下自己。"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
)
print(text)
print("============================================================================")
text = """<|im_start|>user\n文件：MayaEvents\n上下文：<||>.........<||>......<||>...<||>I tap on Maya’s name in my phone and think about how many other versions of me have been able to narrate that.<||>Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.<||>And hopefully soon, she will be as well.<||>As I stare down at a name that is perhaps the most important to me (Barring the recent intrusion of another girl I’ve known for far too long), I think about what I’m going to say when she picks up.<||>But then she picks up.<||>And I still have absolutely nothing.\n目标原文：<||>Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n翻译："""
print(text)
print("============================================================================")

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
exit()

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
