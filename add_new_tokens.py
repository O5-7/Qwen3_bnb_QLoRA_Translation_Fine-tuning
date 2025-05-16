import warnings
from typing import List

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

def add_new_tokens_with_mean_embeddings(in_model, in_tokenizer, tokens:List[str]):
    origin_vocabs_len = len(in_tokenizer.get_vocab())
    token_embedding_size = in_model.config.hidden_size
    new_tokens_embedding = torch.zeros(size = (len(tokens), token_embedding_size), dtype=in_model.model.embed_tokens.weight.dtype)
    with torch.no_grad():
        for i in range(len(tokens)):
            token = tokens[i]
            new_token_ids:torch.Tensor = in_tokenizer(token, return_tensors='pt')['input_ids'][0]
            average_embedding = 0
            for id in new_token_ids.numpy():
                average_embedding += in_model.model.embed_tokens.weight[id]
            average_embedding /= len(new_token_ids.numpy())
            new_tokens_embedding[i] = average_embedding

        in_tokenizer.add_tokens(tokens)
        in_model.resize_token_embeddings(len(tokenizer))

        for i in range(len(tokens)):
            in_model.model.embed_tokens.weight[origin_vocabs_len + i] = new_tokens_embedding[i]

model_name = "../dl_models/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
print(model)

new_tokens = ['<|', '|>', '<|start|>', '<|end|>', '<|translation|>', '{i}', '{/i}', '{b}', '{/b}', '{s}', '{/s}']

add_new_tokens_with_mean_embeddings(model, tokenizer, new_tokens)

model.save_pretrained(save_directory = "../dl_models/Qwen3-0.6B-LIL-tokens")
tokenizer.save_pretrained(save_directory = "../dl_models/Qwen3-0.6B-LIL-tokens")