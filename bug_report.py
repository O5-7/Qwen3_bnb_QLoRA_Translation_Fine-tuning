import time
import warnings

import transformers
from torch import nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from torch.optim import AdamW
import os


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)  # 在训练开始时启用
    print(torch.__version__)
    print(transformers.__version__)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model_name = "../dl_models/Qwen3-0.6B"
    device = "cuda"

    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    config = Qwen3Config.from_pretrained(model_name)
    # model = Qwen3ForCausalLM(config).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-9)
    optimizer.zero_grad()
    model.train()

    train_seq = ["""I decide to spend the night hanging out in my room with Ami.No matter what I suggest doing, she playfully refuses and beckons me into the bed to lie with her and hold her hand.The next few hours pass by with her reciting memories that I wish I could have inherited when I arrived here.Unfortunately, her relationship with me is much more than mine will ever be with her.And all of those memories that mean all of those things are more like anecdotes or poems.One thing that means the world to one person might make absolutely no sense to someone else.I'm caught somewhere in between.I absorb the things she tells me, racking my brain and trying to connect any dots that someone else may have left behind.But I come up with nothing."""]
    inputs = tokenizer(train_seq, return_tensors="pt", padding=True).to(device)
    print(inputs.input_ids.shape)

    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        labels=inputs.input_ids
    )
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    for name,param in model.named_parameters():
        if param is not None:
            print(f"{name:<60}",end='')
            print(torch.isnan(param).any().item())
