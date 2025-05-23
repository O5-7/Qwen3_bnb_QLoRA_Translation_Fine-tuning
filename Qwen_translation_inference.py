import warnings
from math import floor

from sympy.physics.units import temperature

from mc_dataset import translation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")

from os import listdir
from os.path import join
import json
from tqdm import tqdm
from os.path import join
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

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

model_name = "./Qwen3-0.6B-bnb4-LIL-LoRA-16-20000"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the tokenizer and the model
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=Q_config
)

model.eval()
model = torch.compile(model).to(device)

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
            char_obj_str = line[ed_index + 2:]
            char_obj = eval(char_obj_str)
            name_dict.update({key: char_obj.get_name_color()})


def translate(folder_path: str, batch_size:int=4):
    file_names = listdir(folder_path)
    translation_seqs = []

    for file_name in file_names:
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
                pre = dialogue_list[max(i - 4, 0): i]
                next = dialogue_list[i + 1: min(dialogue_len, i + 5)]
                for p in pre:
                    pre_text += f"{p[0]}{p[1]}"
                for n in next:
                    next_text += f"{n[0]}{n[1]}"

                target_text = f"{dialogue_list[i][0]}{dialogue_list[i][1]}"
                prompt = f"<|im_start|>user\\n文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n翻译："
                translation_seqs.append(
                    (in_file_name, 0, *dialogue_list[i][3], prompt)
                )  # in_file_name 0 event_name _hex prompt

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
                pre = string_list[max(i - 4, 0): i]
                next = string_list[i + 1: min(string_len, i + 5)]
                for p in pre:
                    pre_text += f"{p[0]}{p[1]}"
                for n in next:
                    next_text += f"{n[0]}{n[1]}"

                target_text = f"{string_list[i][0]}{string_list[i][1]}"
                prompt = f"<|im_start|>user\\n文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n翻译："
                translation_seqs.append((in_file_name, 1, string_list[i][1], prompt))

    out_file_json = {}
    for file_name in file_names:
        json_file = json.load(
            open(join(folder_path, file_name), encoding="utf-8", mode="r")
        )
        out_file_json.update({json_file["name"]: json_file})

    batch_running = batch_size
    start_index = 0
    out_memory = False
    with tqdm(total=len(translation_seqs), ncols=120) as pbar:
        while True:
            if start_index >= len(translation_seqs):
                break
            try:
                seq_batch = translation_seqs[
                            start_index: min(
                                start_index + batch_running, len(translation_seqs)
                            )
                            ]
                input_text_batch = [seq[-1] for seq in seq_batch]
                model_input_batch = tokenizer(
                    input_text_batch,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                ).to(device)
                output_ids = model.generate(
                    **model_input_batch,
                    temperature=1.7,
                    num_beams=3,
                    max_new_tokens=100,
                    length_penalty=1.5,
                    early_stopping = True
                )
                ress = tokenizer.batch_decode(
                    output_ids[:, model_input_batch.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                out_memory = False

                for i in range(len(seq_batch)):
                    seq = seq_batch[i]
                    if seq[1] == 0:
                        # in_file_name 0 event_name _hex prompt
                        in_file_name, _, event_name, _hex, prompt = seq
                        out_file_json[in_file_name]["dialogue"][event_name][_hex]["translation"] = ress[i]
                    if seq[1] == 1:
                        # in_file_name 1 script["old"] prompt
                        in_file_name, _, old, prompt = seq
                        for i in range(len(out_file_json[in_file_name]["strings"][0])):
                            if out_file_json[in_file_name]["strings"][0][i]["old"] == old:
                                out_file_json[in_file_name]["strings"][0][i]["new"] = ress[i]
                                break
                pbar.update(len(seq_batch))

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f'out of memory in batch = {batch_running}')
                    out_memory = True
                else:
                    raise e

            torch.cuda.empty_cache()

            if out_memory:
                if batch_running == 1:
                    start_index += 1 # jump
                    pbar.update(1)
                batch_running = max(floor(batch_running / 2), 1)
            else:
                start_index += batch_running
                batch_running = min(floor(batch_running * 2), batch_size)

    for file_name,file_json in out_file_json.items():
        with open(join(folder_path, "out", f"{file_name}.json"), "w", encoding="utf-8") as f:
            json.dump(file_json, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    translate("./049_json", batch_size=24)
