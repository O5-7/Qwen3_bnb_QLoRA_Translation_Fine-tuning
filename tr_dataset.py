import numpy as np
import os


def can_cut(text: str) -> bool:
    if not (
        (
            (text[0] == text[-1] == '"')
            or (text[0] == "“" and text[-1] == "”")
            or (text[0] == "‘" and text[-1] == "’")
        )
    ):
        return False
    quotation_list = ""
    quotation_count = 0
    for s in text:
        if s in '"“”‘’':
            quotation_count += 1
            quotation_list += s
            while len(quotation_list) >= 2 and quotation_list[-2:] in [
                '""',
                "“”",
                "‘’",
            ]:
                quotation_list = quotation_list[:-2]
    return quotation_list == "" and quotation_count != 0


def remove_quotation(text: str):
    if text[0] not in '"“”‘’':
        return text
    if '"' not in text and "“" not in text and "‘" not in text:
        return text
    while can_cut(text):
        text = text[1:-1]
    return text


with open("../Data_set/Translation-Data-240804.csv", encoding="utf-8") as f:
    with open("./translation_dataset/tr.txt", mode="w", encoding="utf-8") as w:
        lines = f.readlines()[1:]
        train_data = {}
        for line in lines:
            if line.startswith("French,"):
                continue
            line = line.strip()
            if line[0] == "C":
                split_index = line.rfind(",")
                zh = line[split_index + 1 :]
                en = line[8:split_index]
                if " " in en:
                    continue
                en = remove_quotation(en)
                zh = remove_quotation(zh)
                w.write(f"<|start|><||>{en}<|end|>\\n<|translation|>{zh}" + "\n")
