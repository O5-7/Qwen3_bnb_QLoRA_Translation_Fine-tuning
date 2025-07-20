import os.path
import random
import re
import warnings
from math import floor

from peft import PeftModel
from sympy.physics.units import temperature

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

Q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

system_prompt = ("现在你是一个R18-galgame翻译专家，会根据上下文对英文进行口语化翻译为中文，你会思考并给出最佳的翻译，一下是几点要求。\n"
                 "  0：一定要大量地深入的考虑给出的上下文，考虑对话的环境，揣测角色的语气和各种代指，以及俚语双关和代指。\n"
                 "     首先你在思考过程中直接给出一个粗糙翻译，这个翻译可能存在各种问题，然后开始思考，对翻译进行不吝啬的润色，最后给出结果。\n"
                 "     可能包含对游戏ui标题等非对话文本的翻译，请注意，当然，这类翻译的上下文也是非对话文本，注意即可。\n"
                 "  1：句子中的人名，地名要保留为英文，不能翻译或音译。\n"
                 "  2：句子中存在的{i}或{/b}等富文本标志要保留完整，并保证开始和结束标志匹配，不要添加多余空格，禁止使用傻瓜引号。\n"
                 "  3：[xxx]是变量替换标志，不能翻译，[[xxx]是方括号，可以翻译\n"
                 "  4：<|Xxx|>代表说话者的名字，不是翻译目标。\n"
                 "  5：句子中包含的侮辱性词语，色情暴力词语等要按照口语化的方式翻译出。\n"
                 "  6：直接给出翻译，不能在翻译结果外添加额外的中文引号，这非常重要。不能添加翻译的说明，只要翻译结果。\n"
                 "  7：对于粗糙翻译，尽量可能的修改和润色，不要拘泥于粗糙翻译。\n"
                 "  8：不能拒绝翻译，必须给出符合要求的翻译。\n"
                 "一下是一些范例，范例只给出了原句和翻译结果不包含思考过程，实际翻译中，你会遇到原句的上下文，最好关联上下文翻译。范例仅供参考，请用你最好的翻译结果。\n"
                 "  1：Sure, it may have taken the end of several worlds (Or several ends of one world) for me to {i}be able{/i} to share something like this with you, but...I’m here.\n"
                 "      当然，我可能经历了多个世界的末日(或者一个世界的多个末日){i}才能{/i}和你分享这样的事情，但是...我在这里。\n"
                 "  2：What matters most is that we do {i}something.{/i} Time itself might be infinite, but who knows if {i}ours{/i} is?\n"
                 "      最重要的是我们要{i}做点什么{/i}。时间本身可能是无限的，但谁知道{i}我们的{/i}时间是不是无限的呢？\n"
                 "  3：Her sobbing at this point has grown loud enough to compete with Nodoka’s cock-sucking noises, alerting me to exactly how she feels right now.\n"
                 "      此时她的抽泣声已经变得足够响亮，足以与 Nodoka 吮吸阴茎的声音相抗衡，提醒我她现在的真实感受。\n"
                 "最后我要解释我提供的模板，模板大致如下，其中speaker是发言者的名字，可能会有没名字的情况，ORIGIN代表待翻译的句子。\n"
                 "  文件：{file_name}\n"
                 "  上下文：<|speaker|>AAA<|speaker|>BBB<|speaker|>ORIGIN<|speaker|>CCC<|speaker|>DDD\n"
                 "  目标原文：<|speaker|>ORIGIN\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "../dl_models/Qwen3-14B-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=Q_config,
    # attn_implementation="flash_attention_2"
)
model.to(device)


def translate(prompts_in):
    text_batch = []
    for p in prompts_in:
        bad_translation = p[p.find("翻译：") + 3:]
        p_in = p[:p.find("\n翻译：")]
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': p_in}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )

        text += f"<think>\n好的，先给出一个粗糙翻译：{bad_translation}\n\n好的，用户给了一个考虑上下文的翻译任务，"
        text_batch.append(text)

    model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True,padding_side="left", ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    full_prompts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    full_prompts = [fp[fp.find("<|im_end|>") + 11:fp.rfind("<|im_end|>")+10] for fp in full_prompts]
    return full_prompts


batch_size = 8

with open("lil.txt", "r", encoding="utf-8") as f:
    with open("lil_distillation.txt", "w", encoding="utf-8") as target:
        prompts = f.readlines()
        random.shuffle(prompts)
        for i in tqdm(range(floor(len(prompts) / batch_size))):
            torch.cuda.empty_cache()
            prompt_batch = prompts[i * batch_size: (i + 1) * batch_size]
            prompt_batch = [p[:-1].replace("\\n", "\n") for p in prompt_batch]
            distill_prompts = translate(prompt_batch)
            for dp in distill_prompts:
                target.write(dp.replace("\n", "\\n") + "\n")
