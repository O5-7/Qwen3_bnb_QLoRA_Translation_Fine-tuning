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

tokens = tokenizer(["i am test text."], return_tensors="pt")

prompt_list = []
with open("./translation_dataset/tr.txt", "r", encoding="utf-8") as f:
    prompt_list += [s[:-1] for s in f.readlines()]
with open("./translation_dataset/lil.txt", "r", encoding="utf-8") as f:
    prompt_list += [s[:-1] for s in f.readlines()]
with open("./translation_dataset/mc.txt", "r", encoding="utf-8") as f:
    prompt_list += [s[:-1] for s in f.readlines()]

# shuffle(prompt_list)
prompt_list = prompt_list[:50000]

lens_list = []
for i in range(len(prompt_list)):
    tokens = tokenizer(prompt_list[i:i+1], return_tensors="pt")
    tokens_len = tokens.input_ids.shape[1]
    lens_list.append(tokens_len)
    if tokens_len >= 512:
        print(prompt_list[i])

lens_list = np.array(lens_list)

plt.figure(0)
plt.hist(lens_list, bins=200, log=True)
plt.show()