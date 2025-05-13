import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

model_name = "../dl_models/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(device)

prompt = "我不是唐熙文"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

for k, v in model_inputs.items():
    print(k, v)

generated_ids = model(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    labels=model_inputs.input_ids,
    return_dict=True,
)

print(generated_ids)
