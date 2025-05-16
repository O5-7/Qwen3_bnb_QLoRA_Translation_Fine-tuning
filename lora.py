import random
import warnings

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


class Q3_data(Dataset):
    def __init__(self, file_name: str, tokenizer: Qwen2TokenizerFast, test_limit = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_list = []
        with open(join("./translation_dataset", file_name), "r", encoding="utf-8") as f:
            for prompt in tqdm(f.readlines()[:1000] if test_limit else f.readlines(), ncols=120):
                prompt = prompt.replace("\\n","\n")
                prompt = prompt[:-1]
                prompt += tokenizer.eos_token
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[0]
                if prompt_len <= 512:
                    self.prompt_list.append(prompt)
        self.len = len(self.prompt_list)

    def get_sample(self, num):
        indices = random.sample(range(self.len), num)
        in_batch = [self.prompt_list[i] + self.tokenizer.eos_token for i in indices]
        mask_len_list = []
        for prompt in in_batch:
            tr_start = prompt.find("<|translation|>")
            mask_len = self.tokenizer(
                prompt[: tr_start + 15], return_tensors="pt"
            ).input_ids.shape[0]
            mask_len_list.append(mask_len)
        return mask_len_list, in_batch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.prompt_list[idx]


# generated_ids = model(
#     input_ids=model_inputs.input_ids,
#     attention_mask=model_inputs.attention_mask,
#     labels=model_inputs.input_ids,
#     return_dict=True,
# )

if __name__ == "__main__":
    Q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_name = "../dl_models/Qwen3-0.6B-LIL-tokens"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the tokenizer and the model
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
    model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=Q_config
    )

    model = prepare_model_for_kbit_training(model)

    L_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        # init_lora_weights="pissa",
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(model, L_config).to(device)
    lora_model.print_trainable_parameters()

    step_num = 50000
    optimizer = PagedAdamW8bit(lora_model.parameters(), lr=1e-5)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=step_num,
    )

    lil_data = Q3_data("lil.txt", tokenizer)
    mc_data = Q3_data("mc.txt", tokenizer)
    tr_data = Q3_data("tr.txt", tokenizer)

    # scaler = GradScaler()

    for step in range(step_num):
        mask = []
        batch = []
        data_list = [
            lil_data.get_sample(1),
            mc_data.get_sample(1) if random.random() > 0.5 else tr_data.get_sample(1),
        ]
        for _m, _b in data_list:
            mask += _m
            batch += _b

        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()
        for i, m in enumerate(mask):
            labels[i][:m] = -100
        inputs.to(device)
        labels.to(device)
        optimizer.zero_grad()
        try:
            outputs = lora_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            print(f"step={step:<6} loss={loss.item():.4f}")
            if step % 1000 == 0 and step != 0:
                print(f"step:{step} lora adapter saved=====================================")
                lora_model.save_pretrained(f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter")
                tokenizer.save_pretrained(f"model_saves/{'0' * (6 - len(str(step)))}{step}_lora_adapter")
        except torch.OutOfMemoryError as e:
            print("OutOfMemoryError")
            step -= 1
            continue

    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained("./Qwen3-0.6B-bnb4-LIL-LoRA-16-50000")
    tokenizer.save_pretrained("./Qwen3-0.6B-bnb4-LIL-LoRA-16-50000")
    print("model saved")
