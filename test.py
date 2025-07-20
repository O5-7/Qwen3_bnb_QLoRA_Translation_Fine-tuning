import random
import warnings

from sympy.physics.units import temperature

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.optim import AdamW
from bitsandbytes.optim import PagedAdamW8bit
from transformers import BitsAndBytesConfig

Q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "../dl_models/Qwen3-1.7B"
lora_model_name = "Qwen3-1.7B-bnb4-LIL-16-30000"
print(model_name[model_name.rfind("/") + 1:])
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)

model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=Q_config
).to(device)

lora_model = PeftModel.from_pretrained(model, lora_model_name)

print(lora_model._get_name())
lora_model.eval()
text = ["文件：MakiEvents\n上下文：<||>.........<||>......<||>...<|Maki|>Hey. Sorry it took me so long. Makoto had a question about blowjobs that I needed to answer.<|Maki|>Hey. Sorry it took me so long. I was continuously running into the wall in hopes that all of its atoms aligned at the perfect time, allowing me to pass through it.<|Sensei|>That’s probably the strangest intro to a date I’ve ever had the pleasure of hearing. <|Maki|>Such is the life of the newly proclaimed “Beautiful Porn Salesman.”<|Sensei|>Oh, right. I forgot that was your nickname now.<|Maki|>Don’t worry. So did I until five seconds ago.\n目标原文：<|Maki|>Hey. Sorry it took me so long. I was continuously running into the wall in hopes that all of its atoms aligned at the perfect time, allowing me to pass through it.\n翻译："]
model_inputs = tokenizer(
    text, return_tensors="pt", padding=True, padding_side="left"
).to(lora_model.device)

print(model_inputs.input_ids.shape)

with tqdm(ncols=120) as pbar:
    while (True):
        generated_ids = lora_model.generate(
            **model_inputs,
            temperature=1.7,
            max_new_tokens=100,
            num_beams=3,
            length_penalty=1.5,
            early_stopping=True
        )


        ress = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        # for res in ress:
        #     print(res)
        pbar.update(1)
