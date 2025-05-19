import random
import warnings
from math import floor

from mc_dataset import translation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from tqdm import tqdm

from os import listdir
from os.path import join
import json
import re
from collections import defaultdict
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


# generated_ids = model(
#     input_ids=model_inputs.input_ids,
#     attention_mask=model_inputs.attention_mask,
#     labels=model_inputs.input_ids,
#     return_dict=True,
# )


Q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_name = "./Qwen3-0.6B-bnb4-LIL-LoRA-16-50000"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=Q_config
)
model = torch.compile(model).to(device)
model.eval()

"""
{file_name}
<|speaker|>AAA<|speaker|>BBB
<|start|><|speaker|>ORIGIN<|end|>
<|speaker|>CCC<|speaker|>DDD
<|>translation<|>XXXX
"""

name_dict = {"<>": ""}
def_file_path = join(
    "E:\恋爱课程LessonsInLove\LessonsInLove0.42.0-0.42.0-pc", "game/definitions.rpy"
)
with open(def_file_path, mode="r", encoding="utf-8") as F:

    def absolute(_):
        pass

    class Character:
        def __init__(
            self,
            name,
            color="",
            who_outlines=None,
            who_font=None,
        ):
            self.name = name
            self.color = color

        def get_name_color(self):
            return self.name

    for line in F.readlines():
        line: str
        if line.find("Character(") != -1:
            line = line[11:-1]
            ed_index = line.find("=")
            key = line[: ed_index - 1]
            char_obj_str = line[ed_index + 2 :]
            char_obj = eval(char_obj_str)
            name_dict.update({key: char_obj.get_name_color()})


def translate(folder_path: str, max_new_tokens_limit_ratio: float = 1.7):
    file_names = listdir(folder_path)
    translation_seqs = []
    for file_name in tqdm(file_names):
        json_file = json.load(
            open(join(folder_path, file_name), encoding="utf-8", mode="r")
        )
        in_file_name = json_file["name"]

        dialogue_list = []
        for event_name, event in json_file["dialogue"].items():
            for _hex, dialogue in event.items():
                name = (
                    name_dict[dialogue["speaker"]]
                    if dialogue["speaker"] in name_dict
                    else ""
                )
                dialogue_list.append(
                    (
                        f"<|{name}|>",
                        dialogue["origin"],
                        dialogue["translation"],
                        (event_name, _hex),
                    )
                )
        dialogue_len = len(dialogue_list)
        for i in range(dialogue_len):
            if len(dialogue_list[i][2]) == 0:
                pre_text = ""
                next_text = ""
                pre = dialogue_list[max(i - 4, 0) : i]
                next = dialogue_list[i + 1 : min(dialogue_len, i + 5)]
                for p in pre:
                    pre_text += f"{p[0]}{p[1]}"
                for n in next:
                    next_text += f"{n[0]}{n[1]}"
                prompt = f"{in_file_name}\n{pre_text}\n<|start|>{dialogue_list[i][0]}{dialogue_list[i][1]}<|end|>\n{next_text}\n<|translation|>"
                translation_seqs.append((in_file_name, 0, *dialogue_list[i][3], prompt))

        string_list = []
        if len(json_file["strings"]) == 0:
            continue
        for script in json_file["strings"][0]:
            string_list.append((f"<||>", script["old"], script["new"]))
        string_len = len(string_list)
        for i in range(string_len):
            if len(string_list[i][2]) == 0:
                pre_text = ""
                next_text = ""
                pre = string_list[max(i - 4, 0) : i]
                next = string_list[i + 1 : min(string_len, i + 5)]
                for p in pre:
                    pre_text += f"{p[0]}{p[1]}"
                for n in next:
                    next_text += f"{n[0]}{n[1]}"
                prompt = f"{in_file_name}\n{pre_text}\n<|start|>{string_list[i][0]}{string_list[i][1]}<|end|>\n{next_text}\n<|translation|>"
                translation_seqs.append((in_file_name, 1, string_list[i][1], prompt))

    out_file_json = {}
    for file_name in file_names:
        json_file = json.load(
            open(join(folder_path, file_name), encoding="utf-8", mode="r")
        )
        out_file_json.update({json_file["name"]: json_file})
    for seq in tqdm(translation_seqs, ncols=120):
        input = tokenizer(seq[-1], return_tensors="pt").to(device)
        print(input)
        output_ids = model.generate(**input)
        translation_ids = output_ids[0].tolist()
        translation_res = tokenizer.decode(translation_ids)
        print(translation_res+"\n")
        exit(0)



if __name__ == "__main__":
    translate("./049_json")
