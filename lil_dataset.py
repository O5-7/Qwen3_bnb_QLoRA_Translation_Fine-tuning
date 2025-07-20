from os import listdir
from os.path import join
import json
import re
from collections import defaultdict
from tqdm import tqdm

file_names = listdir("./lil_json")


def remove_flag(input_: str):
    # 为原文lore准备
    input_ = re.sub(r"{size=[-+]?\d+}", "", input_).replace("{/size}", "")
    input_ = re.sub(r"{lore=.*?}", "", input_).replace("{/lore}", "")
    input_.replace("{rb}", "").replace("{/rb}", "")
    input_ = re.sub(r"{rt}.*?{/rt}", "", input_)
    return input_

# 截至048

"""
{file_name}
<|speaker|>AAA<|speaker|>BBB
<|start|><|speaker|>ORIGIN<|end|>
<|speaker|>CCC<|speaker|>DDD
<|translation|>XXXX
"""

"""
文件：{file_name}
上下文：<|speaker|>AAA <|speaker|>BBB <|speaker|>ORIGIN <|speaker|>CCC <|speaker|>DDD
目标原文：<|speaker|>ORIGIN
翻译：XXXX
"""

name_dict = {"<>": ""}
name_dict.setdefault("")
def_file_path = join(
    "D:\恋爱课程LessonsInLove\LessonsInLove0.42.0-0.42.0-pc", "game/definitions.rpy"
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

is_write = True

print(name_dict)
with open("lil.txt", mode="w", encoding="utf-8") as F:
    for file_name in tqdm(file_names):
        json_file = json.load(
            open(join("./lil_json/", file_name), encoding="utf-8", mode="r")
        )
        in_file_name = json_file["name"]

        dialogue_list = []
        for event in json_file["dialogue"].values():
            for _hex, dialogue in event.items():
                name = (
                    name_dict[dialogue["speaker"]]
                    if dialogue["speaker"] in name_dict
                    else ""
                )
                dialogue_list.append(
                    (f"<|{name}|>", dialogue["origin"], dialogue["translation"])
                )
        dialogue_len = len(dialogue_list)
        for i in range(dialogue_len):
            pre_text = ""
            next_text = ""
            pre = dialogue_list[max(i - 4, 0) : i]
            next = dialogue_list[i + 1 : min(dialogue_len, i + 5)]
            for p in pre:
                pre_text += f"{p[0]}{p[1]}"
            for n in next:
                next_text += f"{n[0]}{n[1]}"
            target_text = f"{dialogue_list[i][0]}{dialogue_list[i][1]}"
            # prompt = f"<|im_strat|>user\\n{in_file_name}\\n{pre_text}\\n<|start|>{target_text}<|end|>\\n{next_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<|translation|>{remove_flag(dialogue_list[i][2])}<|im_end|>"
            # prompt = f"文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}\\n翻译：{remove_flag(dialogue_list[i][2])}<|im_end|>"
            # prompt = f"<|im_start|>user\\n文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n翻译：{remove_flag(dialogue_list[i][2])}<|im_end|>"
            prompt = f"文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}\\n翻译：{remove_flag(dialogue_list[i][2])}"
            if is_write:
                F.write(prompt + "\n")

        string_list = []
        if len(json_file["strings"]) == 0:
            continue
        for script in json_file["strings"][0]:
            string_list.append((f"<||>", script["old"], script["new"]))
        string_len = len(string_list)
        for i in range(string_len):
            pre_text = ""
            next_text = ""
            pre = string_list[max(i - 4, 0) : i]
            next = string_list[i + 1 : min(string_len, i + 5)]
            for p in pre:
                pre_text += f"{p[0]}{p[1]}"
            for n in next:
                next_text += f"{n[0]}{n[1]}"
            target_text = f"{string_list[i][0]}{string_list[i][1]}"
            # prompt = f"<|im_strat|>user\\n{in_file_name}\\n{pre_text}\\n<|start|>{target_text}<|end|>\\n{next_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<|translation|>{string_list[i][2]}<|im_end|>"
            # prompt = f"文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}\\n翻译：{string_list[i][2]}<|im_end|>"
            # prompt = f"<|im_start|>user\\n文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n翻译：{string_list[i][2]}<|im_end|>"
            prompt = f"文件：{in_file_name}\\n上下文：{pre_text}{target_text}{next_text}\\n目标原文：{target_text}\\n翻译：{string_list[i][2]}"
            if is_write:
                F.write(prompt + "\n")
