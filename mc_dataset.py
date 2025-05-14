import pandas as pd
import json

# 读取 parquet 文件
df_1 = pd.read_parquet("../Data_set/mc-translation/train-00000-of-00002.parquet")
df_2 = pd.read_parquet("../Data_set/mc-translation/train-00001-of-00002.parquet")
df = pd.concat([df_1, df_2], ignore_index=True)

# 显示前几行
EZ_results = df[(df['in_lang'] == 'English') & (df['out_lang'] == 'Chinese (Simplified)')]['conversations'].tolist()
ZE_results = df[(df['in_lang'] == 'Chinese (Simplified)') & (df['out_lang'] == 'English')]['conversations'].tolist()

with open("./translation_dataset/mc.txt",mode="w",encoding="utf-8") as f:
    for result in EZ_results:
        in_put:str = result[0]["value"]
        out_put:str = result[1]["value"]

        start = in_put[in_put.find("<question>")+10:in_put.find("</question>")]
        translation = out_put[out_put.find("<question>")+10:out_put.find("</question>")]
        start += in_put[in_put.find("<options>")+9:in_put.find("</options>")].replace("\n  <option>","(").replace("</option>",")")
        translation += out_put[out_put.find("<options>")+9:out_put.find("</options>")].replace("\n  <option>","(").replace("</option>",")")

        write_str = f"<|start|><||>{start}<|end|>\\n<|translation|>{translation}"
        write_str = write_str.replace("\n","") + "\n"
        f.write(write_str)
    for result in ZE_results:
        in_put:str = result[0]["value"]
        out_put:str = result[1]["value"]

        translation = in_put[in_put.find("<question>")+10:in_put.find("</question>")]
        start = out_put[out_put.find("<question>")+10:out_put.find("</question>")]
        translation += in_put[in_put.find("<options>")+9:in_put.find("</options>")].replace("\n  <option>","(").replace("</option>",")")
        start += out_put[out_put.find("<options>")+9:out_put.find("</options>")].replace("\n  <option>","(").replace("</option>",")")

        write_str = f"""<|start|><||>{start}<|end|>\\n<|translation|>{translation}"""
        write_str = write_str.replace("\n","") + "\n"
        f.write(write_str)
