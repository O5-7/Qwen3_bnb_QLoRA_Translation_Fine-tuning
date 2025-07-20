import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")

from random import shuffle
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

model_name = "../dl_models/Qwen3-0.6B"
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)


prompt_list = []

with open("lil.txt", "r", encoding="utf-8") as f:
    prompt_list += [s[s.find("翻译：") + 3:-1] for s in tqdm(f.readlines())]

seq_len_list = []
for prompt in tqdm(prompt_list):
    ids = tokenizer(prompt, return_tensors="pt")
    ids_len = ids.input_ids.shape[1]
    if ids_len > 300:
        continue
    if ids_len > 100:
        print(prompt)
    else:
        seq_len_list.append(ids_len)

plt.figure(0)
plt.hist(seq_len_list, bins=100)
plt.show()
