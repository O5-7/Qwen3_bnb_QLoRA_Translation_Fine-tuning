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

model_name = "../dl_models/Qwen3-1.7B"
print(model_name[model_name.rfind("/") + 1 :])
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)

model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=Q_config
).to(device)

print(model._get_name())
model.eval()
text = [
    """<|im_start|>user\n文件：AmiEvents\n上下文：<|Sensei|>Any idea what you’re going to buy with your first paycheck?<|Ami|>Not a clue. I don’t even know what I’m getting paid yet. <|Ami|>Uta hasn’t told me when I’m starting either so I’m kinda just trying on the costume to see how I look and stuff today.<|Ami|>She was definitely right about it making me feel prettier than I actually am.<|Sensei|>I mean, you’re pretty adorable to begin with. But yeah, this costume is good. I support this look.<|Ami|>Of course {i}you{/i} support it, Master. You and your unhealthy maid addiction.<|Sensei|>I can assure you this addiction is completely healthy for both of us. <|Ami|>Healthy for me? I’m gonna need you to explain why, dearest [uncle].<|Sensei|>Why do I need to explain anything? I think our relationship has progressed enough for you to understand what that means.\n目标原文：<|Sensei|>I mean, you’re pretty adorable to begin with. But yeah, this costume is good. I support this look.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n翻译：""",
    """<|im_start|>user\n文件：AmiEvents\n上下文：<|Ami|>What’s the point of having a vacation at the beach if you’re not going to go swimming?<|Ami|>Plus, the swimsuit I have now is really...childish anyway.<|Ami|>And if I’m gonna start doing grown-up stuff like working and living at the dorm, I’m gonna need to look the part.<|Ami|>The days of the one piece are over, Sensei! The days of bikini Ami are about to begin!<|Sensei|>Are you sure you’re going to be able to find one that...you know...{i}fits?{/i}<|Ami|>...<|Sensei|>Because I think the one piece might actually be a better bet until you...grow in a little more.<|Ami|>...<|Ami|>Sometimes, the sound of the ocean helps me stay focused while filing your returns.\n目标原文：<|Sensei|>Are you sure you’re going to be able to find one that...you know...{i}fits?{/i}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n翻译：""",
    """<|im_start|>user\n文件：AmiEvents\n上下文：<|Ami|>Nope!<|Sensei|>See what I mean?<|Ami|>Nope! Cause I also know you can’t be happy with anyone else.<|Ami|>You and me are gonna be together forever cause that’s how things are meant to be.<|Ami|>And we’re gonna buy tons more bathing suits and eat tons more french fries because we both think those things are fun and we love each other very, very much.<|Ami|>And if anyone ever tries to ruin that, I will do horrible things to them.<|Sensei|>You do realize that if “anyone tries to ruin that” it will be me, right?<|Ami|>You want to ruin me? Your adorable [niece]? Who cooks you breakfast every morning and lets you cum on her face?<|Sensei|>I don’t know what I want. But I know that a relationship isn’t anywhere near the top of the list.\n目标原文：<|Ami|>And we’re gonna buy tons more bathing suits and eat tons more french fries because we both think those things are fun and we love each other very, very much.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n翻译：""",
]

model_inputs = tokenizer(
    text, return_tensors="pt", padding=True, padding_side="left"
).to(model.device)

print(model_inputs.input_ids.shape)

generated_ids = model.generate(
    **model_inputs,
    temperature=0.7,
    num_beams=5,
    max_new_tokens=100,
    length_penalty=1.5,
    early_stopping = False
)

print(generated_ids.shape)

ress = tokenizer.batch_decode(
    generated_ids[:,model_inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
for res in ress:
    print(res)

exit()

output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
    "\n"
)
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
